"""
BƯỚC 3: Fine-tune LLM với QLoRA
Training Qwen model với dataset đã trộn (tiếng Anh + tiếng Việt)
Sử dụng Unsloth cho training nhanh hơn
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

# QUAN TRỌNG: Import unsloth TRƯỚC transformers, trl, peft để đảm bảo tất cả optimizations được áp dụng
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM với QLoRA")
    parser.add_argument(
        "--dataset",
        default="data/processed_finetune/llm_train_mixed.jsonl",
        help="Path to training dataset JSONL"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base LLM model ID"
    )
    parser.add_argument(
        "--output-dir",
        default="models/finetuned/qwen-mixed",
        help="Output directory for fine-tuned adapter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (giảm xuống 2 để tránh OOM trên 16GB VRAM)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (1 epoch là CHUẨN cho model 0.5B với 225k samples để tránh overfitting)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Bắt đầu Fine-tuning LLM với QLoRA...")
    print(f"   Model: {args.model_id}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output_dir}")
    
    # Check if Unsloth is available
    if USE_UNSLOTH:
        print("\n✅ Sử dụng Unsloth cho training nhanh hơn...")
        use_unsloth = True
    else:
        print("\n⚠️  Unsloth không có sẵn, sử dụng transformers thông thường...")
        print("   Cài đặt: pip install unsloth")
        use_unsloth = False
    
    # Load dataset
    print("\n📂 Đang load dataset...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   - Số samples: {len(dataset)}")
    
    if use_unsloth:
        # Sử dụng Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # 4-bit quantization
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )
        
        # Format dataset với Qwen template
        def format_prompts(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            
            texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                if input_text:
                    text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                else:
                    text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                texts.append(text)
            
            return {"text": texts}
        
        dataset = dataset.map(format_prompts, batched=True)
        
        # Trainer với Unsloth - sử dụng adamw_8bit để tiết kiệm VRAM
        # SFTTrainer trong trl 0.24.0 dùng processing_class thay vì tokenizer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            processing_class=tokenizer,  # Sử dụng processing_class thay vì tokenizer
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # BF16 thay vì FP16
                gradient_checkpointing=True,  # Bật để tiết kiệm VRAM
                logging_steps=10,
                optim="adamw_8bit",  # Sử dụng 8-bit optimizer để tiết kiệm 75% VRAM
                output_dir=args.output_dir,
                save_strategy="epoch",
                save_total_limit=3,
            ),
        )
        
        # Train
        print("\n🏋️ Bắt đầu training...")
        trainer.train()
        
        # Save model
        print(f"\n💾 Đang lưu model tại {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
    else:
        # Sử dụng transformers thông thường
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )
        
        # Load model với 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        
        # Format dataset
        def format_prompts(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            
            texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                if input_text:
                    text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                else:
                    text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                texts.append(text)
            
            return tokenizer(texts, truncation=True, max_length=args.max_seq_length, padding="max_length")
        
        dataset = dataset.map(format_prompts, batched=True)
        
        # Trainer với transformers thông thường
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # BF16 thay vì FP16
                gradient_checkpointing=True,  # Bật để tiết kiệm VRAM
                logging_steps=10,
                output_dir=args.output_dir,
                save_strategy="epoch",
                save_total_limit=3,
            ),
            data_collator=lambda x: {
                "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in x]),
                "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in x]),
                "labels": torch.stack([torch.tensor(item["input_ids"]) for item in x]),
            },
        )
        
        # Train
        print("\n🏋️ Bắt đầu training...")
        trainer.train()
        
        # Save model
        print(f"\n💾 Đang lưu model tại {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ Hoàn thành training!")
    print(f"   Adapter đã được lưu tại: {args.output_dir}")
    print("\n📝 Để sử dụng model mới, cập nhật .env:")
    print(f"   GEN_MODEL={args.output_dir}")
    print("\n⚠️  Lưu ý: Với LoRA adapter, bạn cần load base model + adapter khi inference")


if __name__ == "__main__":
    main()


