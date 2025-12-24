#!/usr/bin/env python3
"""
Pipeline tích hợp Whisper (GPU) + LLM với Unsloth LoRA (GPU)
Sử dụng GPU cho cả hai models để tối ưu hiệu suất
"""
import argparse
import os
import sys
from pathlib import Path
import time
import torch


def check_gpu():
    """Kiểm tra GPU và cuDNN"""
    if not torch.cuda.is_available():
        print("⚠️  GPU không khả dụng")
        return False
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Check cuDNN
    try:
        import torch.backends.cudnn as cudnn
        if cudnn.is_available():
            print(f"   cuDNN Available: ✅")
        else:
            print(f"   cuDNN Available: ❌")
            return False
    except Exception as e:
        print(f"   ⚠️  Lỗi check cuDNN: {e}")
        return False
    
    return True


def load_whisper_gpu(model_path: str):
    """
    Load Whisper model với GPU
    Sử dụng faster-whisper với CTranslate2
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper chưa cài đặt")
        print("📦 Cài đặt: pip install faster-whisper")
        return None
    
    print("="*80)
    print("📦 LOADING WHISPER MODEL (GPU)")
    print("="*80)
    print(f"Model: {model_path}")
    
    # Try GPU first, fallback to CPU if cuDNN issues
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    
    try:
        start = time.time()
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        load_time = time.time() - start
        print(f"✅ Whisper loaded in {load_time:.2f}s")
        return model
    except Exception as e:
        print(f"⚠️  Lỗi load Whisper với GPU: {e}")
        print("🔄 Fallback to CPU...")
        try:
            model = WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8"
            )
            print("✅ Whisper loaded on CPU")
            return model
        except Exception as e2:
            print(f"❌ Lỗi load Whisper: {e2}")
            return None


def load_llm_unsloth_lora(
    base_model: str,
    lora_adapter_path: str,
    load_in_4bit: bool = True,
    max_seq_length: int = 2048
):
    """
    Load LLM với Unsloth và LoRA adapter
    
    Args:
        base_model: Base model path (e.g., "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit")
        lora_adapter_path: Path to LoRA adapter
        load_in_4bit: Load in 4-bit quantization
        max_seq_length: Maximum sequence length
    """
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        print("❌ Unsloth chưa cài đặt")
        print("📦 Cài đặt: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        return None, None
    
    print("="*80)
    print("📦 LOADING LLM WITH UNSLOTH + LORA (GPU)")
    print("="*80)
    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {lora_adapter_path}")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"Max sequence length: {max_seq_length}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  GPU không khả dụng, nhưng vẫn có thể chạy trên CPU (chậm)")
    
    try:
        start = time.time()
        
        # Load model với Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto
            load_in_4bit=load_in_4bit,
        )
        
        # Load LoRA adapter
        if Path(lora_adapter_path).exists():
            print(f"📂 Loading LoRA adapter from: {lora_adapter_path}")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,  # LoRA rank (should match training config)
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=3407,
            )
            
            # Load adapter weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            print("✅ LoRA adapter loaded")
        else:
            print(f"⚠️  LoRA adapter not found: {lora_adapter_path}")
            print("   Using base model only")
        
        load_time = time.time() - start
        print(f"✅ LLM loaded in {load_time:.2f}s")
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ Lỗi load LLM: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def transcribe_audio(whisper_model, audio_path: str, language: str = None):
    """Transcribe audio với Whisper"""
    print("\n" + "="*80)
    print("🎤 TRANSCRIBING AUDIO")
    print("="*80)
    print(f"Audio: {audio_path}")
    
    start = time.time()
    segments, info = whisper_model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True
    )
    
    # Collect text
    full_text = " ".join([seg.text for seg in segments])
    transcribe_time = time.time() - start
    
    print(f"Language: {info.language} ({info.language_probability:.2%})")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Processing time: {transcribe_time:.2f}s")
    print(f"Speed: {transcribe_time/info.duration:.2f}x realtime")
    print(f"\nTranscript:\n{full_text}")
    
    return full_text, info


def generate_with_llm(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate text với LLM"""
    print("\n" + "="*80)
    print("🤖 GENERATING WITH LLM")
    print("="*80)
    print(f"Prompt: {prompt[:100]}...")
    
    # Format prompt (Qwen format)
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    gen_time = time.time() - start
    
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Tokens: {len(outputs[0]) - len(inputs[0])}")
    print(f"\nResponse:\n{response}")
    
    return response


def pipeline_audio_to_text(
    audio_path: str,
    whisper_model_path: str = "models/final/whisper-vi-en-ct2",
    llm_base_model: str = "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
    lora_adapter_path: str = "models/finetuned/qwen-mixed",
    use_llm: bool = True
):
    """
    Pipeline hoàn chỉnh: Audio -> Transcript -> LLM Response
    
    Args:
        audio_path: Path to audio file
        whisper_model_path: Path to Whisper CT2 model
        llm_base_model: Base LLM model name
        lora_adapter_path: Path to LoRA adapter
        use_llm: Whether to use LLM for post-processing
    """
    print("="*80)
    print("🚀 AUDIO TO TEXT PIPELINE (GPU)")
    print("="*80)
    
    # Check GPU
    if not check_gpu():
        print("⚠️  GPU không khả dụng, sẽ chạy trên CPU (chậm hơn)")
    
    # Load Whisper
    whisper_model = load_whisper_gpu(whisper_model_path)
    if whisper_model is None:
        print("❌ Không thể load Whisper model")
        return None
    
    # Transcribe
    transcript, info = transcribe_audio(whisper_model, audio_path)
    
    # Optionally use LLM for post-processing
    if use_llm:
        # Load LLM
        llm_model, tokenizer = load_llm_unsloth_lora(
            base_model=llm_base_model,
            lora_adapter_path=lora_adapter_path
        )
        
        if llm_model is not None:
            # Create prompt from transcript
            prompt = f"""Bạn là một trợ lý AI chuyên xử lý và cải thiện văn bản từ transcript audio.

Transcript gốc:
{transcript}

Hãy:
1. Sửa lỗi chính tả và ngữ pháp
2. Cải thiện độ mượt mà và tự nhiên của câu văn
3. Giữ nguyên ý nghĩa gốc
4. Trả về văn bản đã được cải thiện

Văn bản đã cải thiện:"""
            
            # Generate improved text
            improved_text = generate_with_llm(llm_model, tokenizer, prompt)
            
            print("\n" + "="*80)
            print("📊 FINAL RESULT")
            print("="*80)
            print(f"Original transcript:\n{transcript}\n")
            print(f"Improved text:\n{improved_text}\n")
            
            return {
                "transcript": transcript,
                "improved_text": improved_text,
                "language": info.language,
                "duration": info.duration
            }
    
    return {
        "transcript": transcript,
        "language": info.language,
        "duration": info.duration
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Whisper (GPU) + LLM Unsloth LoRA (GPU)"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--whisper-model",
        default="models/final/whisper-vi-en-ct2",
        help="Path to Whisper CT2 model"
    )
    parser.add_argument(
        "--llm-base",
        default="unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
        help="Base LLM model name"
    )
    parser.add_argument(
        "--lora-adapter",
        default="models/finetuned/qwen-mixed",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only transcribe, don't use LLM"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for Whisper (vi, en, or None for auto)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    result = pipeline_audio_to_text(
        audio_path=args.audio,
        whisper_model_path=args.whisper_model,
        llm_base_model=args.llm_base,
        lora_adapter_path=args.lora_adapter,
        use_llm=not args.no_llm
    )
    
    if result:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

