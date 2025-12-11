"""
Evaluate Whisper model trên test set và so sánh với model gốc
"""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import json


def load_audio(example, base_dir):
    """Load audio file"""
    import librosa
    audio_path_str = example["audio"]
    
    # Fix duplicate /audio/audio/ in path
    audio_path_str = audio_path_str.replace("/audio/audio/", "/audio/")
    
    # Try multiple paths
    project_root = Path(__file__).parent.parent
    audio_path = None
    
    # Try 1: Relative to project root
    test_path = project_root / audio_path_str
    if test_path.exists():
        audio_path = test_path
    else:
        # Try 2: Extract filename and find in base_dir
        filename = Path(audio_path_str).name
        # Try to find in test/audio/
        test_audio_path = Path(base_dir) / "test" / "audio" / filename
        if test_audio_path.exists():
            audio_path = test_audio_path
        else:
            # Try 3: Direct path from base_dir
            test_path2 = Path(base_dir) / audio_path_str
            if test_path2.exists():
                audio_path = test_path2
    
    if audio_path and audio_path.exists():
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            example["audio"] = {"array": audio, "sampling_rate": sr}
        except Exception as e:
            print(f"⚠️  Lỗi load audio {audio_path}: {e}")
            example["audio"] = {"array": [], "sampling_rate": 16000}
    else:
        # Don't print too many warnings, just skip
        example["audio"] = {"array": [], "sampling_rate": 16000}
    return example


def evaluate_model(model_path, test_dataset_path, base_audio_dir):
    """Evaluate model trên test set"""
    print(f"\n🔍 Đang evaluate model: {model_path}")
    
    # Load model và processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    # Load test dataset
    print(f"📂 Đang load test dataset: {test_dataset_path}")
    dataset = load_dataset("json", data_files=test_dataset_path, split="train")
    
    # Load audio files
    print("🎵 Đang load audio files...")
    dataset = dataset.map(
        lambda x: load_audio(x, base_audio_dir),
        desc="Loading audio"
    )
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) > 0)
    
    print(f"   - Số samples: {len(dataset)}")
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Evaluate với batch processing để tăng tốc độ
    print("\n📊 Đang evaluate...")
    predictions = []
    references = []
    batch_size = 16  # Tăng batch size để sử dụng GPU tốt hơn
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            # Get batch indices
            batch_end = min(i + batch_size, len(dataset))
            
            # Prepare batch audio
            batch_audios = []
            batch_sampling_rates = []
            batch_texts = []
            
            for idx in range(i, batch_end):
                ex = dataset[idx]
                batch_audios.append(ex["audio"]["array"])
                batch_sampling_rates.append(ex["audio"]["sampling_rate"])
                batch_texts.append(ex["text"])
            
            # Process batch
            batch_input_features = []
            for audio, sr in zip(batch_audios, batch_sampling_rates):
                features = processor.feature_extractor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features
                batch_input_features.append(features)
            
            # Stack và pad batch
            max_len = max(f.shape[-1] for f in batch_input_features)
            padded_features = []
            for f in batch_input_features:
                pad_len = max_len - f.shape[-1]
                if pad_len > 0:
                    f = torch.nn.functional.pad(f, (0, pad_len))
                padded_features.append(f)
            
            batch_input = torch.cat(padded_features, dim=0).to(device)
            
            # Generate batch
            generated_ids = model.generate(
                batch_input,
                max_length=448,
                language=None,  # Auto-detect
                task="transcribe",
                num_beams=1,  # Greedy decoding để nhanh hơn
                do_sample=False
            )
            
            # Decode batch
            batch_predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            predictions.extend(batch_predictions)
            references.extend(batch_texts)
    
    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Compute character error rate (CER) if available
    try:
        cer_metric = evaluate.load("cer")
        cer = cer_metric.compute(predictions=predictions, references=references)
    except:
        cer = None
    
    # Print results
    print("\n" + "="*60)
    print("📊 KẾT QUẢ EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Số samples: {len(predictions)}")
    print(f"WER (Word Error Rate): {wer:.4f} ({wer*100:.2f}%)")
    if cer:
        print(f"CER (Character Error Rate): {cer:.4f} ({cer*100:.2f}%)")
    print("="*60)
    
    # Show some examples
    print("\n📝 Một số ví dụ:")
    print("-"*60)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Reference: {references[i][:100]}...")
        print(f"  Prediction: {predictions[i][:100]}...")
    
    return {
        "model": model_path,
        "wer": wer,
        "cer": cer,
        "num_samples": len(predictions),
        "predictions": predictions[:10],  # Save first 10 for inspection
        "references": references[:10]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper models")
    parser.add_argument(
        "--test-dataset",
        default="data/processed_finetune/whisper_test.jsonl",
        help="Path to test dataset JSONL"
    )
    parser.add_argument(
        "--base-audio-dir",
        default="data/processed/full_merged_dataset",
        help="Base directory for audio files"
    )
    parser.add_argument(
        "--finetuned-model",
        default="models/finetuned/whisper-test",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-small",
        help="Base model ID for comparison"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    # Evaluate fine-tuned model
    print("="*60)
    print("🚀 EVALUATION WHISPER MODELS")
    print("="*60)
    
    finetuned_result = evaluate_model(
        args.finetuned_model,
        args.test_dataset,
        args.base_audio_dir
    )
    results["finetuned"] = finetuned_result
    
    # Evaluate base model
    print("\n" + "="*60)
    base_result = evaluate_model(
        args.base_model,
        args.test_dataset,
        args.base_audio_dir
    )
    results["base"] = base_result
    
    # Comparison
    print("\n" + "="*60)
    print("📊 SO SÁNH KẾT QUẢ")
    print("="*60)
    print(f"Base Model ({args.base_model}):")
    print(f"  WER: {base_result['wer']:.4f} ({base_result['wer']*100:.2f}%)")
    if base_result['cer']:
        print(f"  CER: {base_result['cer']:.4f} ({base_result['cer']*100:.2f}%)")
    
    print(f"\nFine-tuned Model ({args.finetuned_model}):")
    print(f"  WER: {finetuned_result['wer']:.4f} ({finetuned_result['wer']*100:.2f}%)")
    if finetuned_result['cer']:
        print(f"  CER: {finetuned_result['cer']:.4f} ({finetuned_result['cer']*100:.2f}%)")
    
    # Improvement
    wer_improvement = base_result['wer'] - finetuned_result['wer']
    wer_improvement_pct = (wer_improvement / base_result['wer']) * 100
    
    print(f"\n📈 Cải thiện:")
    print(f"  WER giảm: {wer_improvement:.4f} ({wer_improvement_pct:.2f}%)")
    if base_result['cer'] and finetuned_result['cer']:
        cer_improvement = base_result['cer'] - finetuned_result['cer']
        cer_improvement_pct = (cer_improvement / base_result['cer']) * 100
        print(f"  CER giảm: {cer_improvement:.4f} ({cer_improvement_pct:.2f}%)")
    
    print("="*60)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Đã lưu kết quả vào: {args.output}")


if __name__ == "__main__":
    main()

