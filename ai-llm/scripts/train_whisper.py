"""
BƯỚC 2: Fine-tune Whisper Model
Training Whisper với dataset đã trộn (tiếng Anh + tiếng Việt)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil
import gc
import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator cho Whisper training"""
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad inputs
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # If decoder_input_ids are provided, use them; otherwise use labels shifted right
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


def prepare_dataset(batch, processor, model):
    """Prepare dataset batch"""
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute log-Mel spectrogram
    input_features = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize text
    batch["input_features"] = input_features
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    # Giải phóng audio thô để giảm RAM
    batch.pop("audio", None)
    
    return batch


def compute_metrics(pred, processor: WhisperProcessor):
    """Compute WER metric"""
    try:
        wer_metric = evaluate.load("wer")
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    except Exception as e:
        print(f"⚠️  Lỗi tính WER: {e}")
        return {"wer": 1.0}  # Return high WER if computation fails


def clear_caches():
    """Xóa cache HuggingFace/torch để giảm chiếm RAM/disk trước khi train"""
    cache_targets = []

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_targets.append(hf_home / "datasets")
    cache_targets.append(hf_home / "transformers")

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    cache_targets.append(torch_home)
    cache_targets.append(Path.home() / ".cache" / "torch_extensions")

    local_cache = Path(".cache")
    cache_targets.append(local_cache / "huggingface" / "datasets")
    cache_targets.append(local_cache / "huggingface" / "transformers")

    print("🧹 Đang dọn cache:")
    for path in cache_targets:
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                print(f"   - Đã xóa {path}")
        except Exception as e:
            print(f"   - Không thể xóa {path}: {e}")

    # Thu hồi bộ nhớ sau khi xóa cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")
    parser.add_argument(
        "--dataset",
        default="data/processed_finetune/whisper_train_mixed.jsonl",
        help="Path to training dataset JSONL"
    )
    parser.add_argument(
        "--model-id",
        default="openai/whisper-small",
        help="Base Whisper model ID"
    )
    parser.add_argument(
        "--output-dir",
        default="models/finetuned/whisper-mixed",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (tối ưu cho RTX 5060 Ti 16GB)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 training (recommended for Ampere+ GPUs, sẽ tự động bật nếu GPU hỗ trợ)"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Tắt BF16 training (dùng FP32)"
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Bỏ qua bước xóa cache HF/torch trước khi train"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (chỉ dùng khi save_strategy='steps')"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (chỉ dùng khi eval_strategy='steps')"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="Save strategy: 'epoch' (khuyến nghị) hoặc 'steps'"
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Evaluation strategy: 'epoch' (khuyến nghị), 'steps', hoặc 'no'"
    )
    
    args = parser.parse_args()
    
    print("🚀 Bắt đầu Fine-tuning Whisper...")
    print(f"   Model: {args.model_id}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output_dir}")

    if args.no_clear_cache:
        print("\n🧹 Bỏ qua dọn cache (đã chọn --no-clear-cache)")
    else:
        clear_caches()
    
    # Load dataset
    print("\n📂 Đang load dataset...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   - Số samples: {len(dataset)}")
    
    # Load processor - QUAN TRỌNG: KHÔNG set language cố định
    print("\n🔧 Đang load processor và model...")
    processor = WhisperProcessor.from_pretrained(
        args.model_id,
        language=None,  # QUAN TRỌNG: Để None để tự detect language từ dataset
        task="transcribe"
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    
    # Gradient checkpointing disabled to avoid backward pass issues
    # Will be enabled via training_args if needed
    
    # Load audio và prepare dataset
    print("\n🎵 Đang prepare dataset (load audio files)...")
    from datasets import Audio
    import librosa
    
    # Load audio và filter empty trong một lần để tối ưu
    # Thêm flag để đánh dấu samples hợp lệ
    def load_audio_with_flag(example):
        audio_path = example["audio"]
        original_path = audio_path
        
        # Resolve path - đường dẫn trong dataset có thể là relative hoặc absolute
        if not Path(audio_path).is_absolute():
            # Nếu đường dẫn bắt đầu với "data/", resolve từ project root
            if audio_path.startswith("data/"):
                project_root = Path(args.dataset).parent.parent.parent
                test_path = project_root / audio_path
                if test_path.exists():
                    audio_path = str(test_path)
                else:
                    # Nếu không tìm thấy, extract filename và tìm trong các split folders
                    filename = Path(audio_path).name
                    base_dir = Path(args.dataset).parent.parent / "full_merged_dataset"
                    found = False
                    for split in ["train", "val", "test"]:
                        test_path = base_dir / split / "audio" / filename
                        if test_path.exists():
                            audio_path = str(test_path)
                            found = True
                            break
                    if not found:
                        audio_path = None
            else:
                # Đường dẫn chỉ là filename, tìm trong các split folders
                base_dir = Path(args.dataset).parent.parent / "full_merged_dataset"
                found = False
                for split in ["train", "val", "test"]:
                    test_path = base_dir / split / "audio" / audio_path
                    if test_path.exists():
                        audio_path = str(test_path)
                        found = True
                        break
                if not found:
                    audio_path = None
        
        if audio_path and Path(audio_path).exists():
            try:
                audio, sr = librosa.load(str(audio_path), sr=16000)
                # Librosa trả về float64, cần ép float32 để khớp schema datasets
                audio = np.asarray(audio, dtype=np.float32)
                example["audio"] = {"array": audio, "sampling_rate": int(sr)}
                example["has_audio"] = len(audio) > 0
            except Exception as e:
                # Return empty audio on error
                example["audio"] = {"array": np.array([], dtype=np.float32), "sampling_rate": 16000}
                example["has_audio"] = False
        else:
            # Return empty audio if not found
            example["audio"] = {"array": np.array([], dtype=np.float32), "sampling_rate": 16000}
            example["has_audio"] = False
        return example
    
    # Load audio và đánh dấu samples hợp lệ - giới hạn CPU để tránh tràn RAM
    num_proc = min(4, os.cpu_count() or 1)
    print(f"   - Sử dụng {num_proc} CPU cores cho preprocessing")
    dataset = dataset.map(
        load_audio_with_flag, 
        load_from_cache_file=False,
        num_proc=num_proc,
        writer_batch_size=1000,
        desc="Loading audio files (limited RAM)"
    )
    # Filter nhanh hơn bằng cách dùng flag thay vì check length mỗi lần
    dataset = dataset.filter(
        lambda x: x["has_audio"], 
        load_from_cache_file=False,
        num_proc=num_proc,
        writer_batch_size=1000,
        desc="Filtering empty audio (limited RAM)"
    )
    # Remove flag column
    dataset = dataset.remove_columns(["has_audio"])
    
    # Prepare features - KHÔNG cache để tiết kiệm dung lượng đĩa
    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor, model),
        remove_columns=dataset.column_names,
        desc="Preparing features",
        load_from_cache_file=False,
        num_proc=num_proc,
        writer_batch_size=500,
    )
    
    # Split train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"   - Train: {len(train_dataset)} samples")
    print(f"   - Eval: {len(eval_dataset)} samples")
    
    # Training arguments - Tối ưu cho 225k samples trên RTX 5060 Ti 16GB
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,  # 3 epochs, có thể dừng sớm sau epoch 1-2
        gradient_checkpointing=False,  # Disabled to avoid backward pass issues
        bf16=(args.bf16 or not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # BF16 mặc định nếu GPU hỗ trợ
        eval_strategy=args.eval_strategy,  # "epoch" - đánh giá sau mỗi epoch
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,  # "epoch" - lưu sau mỗi epoch (để có checkpoint epoch 1, 2, 3)
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"] if torch.cuda.is_available() else [],
        load_best_model_at_end=True,  # Tự động chọn model tốt nhất trong 3 epochs
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=3,  # Giữ 3 checkpoint gần nhất (epoch 1, 2, 3)
        predict_with_generate=True,  # Cho phép generate trong evaluation
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,  # Fixed: use processing_class instead of deprecated tokenizer
        # compute_metrics=compute_metrics,  # Optional: uncomment to compute WER
    )
    
    # Train
    print("\n🏋️ Bắt đầu training...")
    trainer.train()
    
    # Save final model
    print(f"\n💾 Đang lưu model tại {args.output_dir}...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("\n✅ Hoàn thành training!")
    print(f"   Model đã được lưu tại: {args.output_dir}")
    print("\n📝 Để sử dụng model mới, cập nhật .env:")
    print(f"   ASR_MODEL={args.output_dir}")


if __name__ == "__main__":
    main()

