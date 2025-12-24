#!/usr/bin/env python3
"""
Whisper với bfloat16 (bf16) - Fix cuDNN issues
"""
import sys
import os
from pathlib import Path

# Set environment variables TRƯỚC KHI import faster_whisper
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import sau khi set env
from faster_whisper import WhisperModel
import time

def transcribe_bf16(audio_path, model_path="models/final/whisper-vi-en-ct2"):
    """Transcribe với bfloat16"""
    print("="*80)
    print("🎤 WHISPER TRANSCRIPTION (GPU - bfloat16)")
    print("="*80)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print("="*80)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "bfloat16"
            print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Compute type: {compute_type}")
        else:
            device = "cpu"
            compute_type = "int8"
            print("\n⚠️  GPU không khả dụng, sử dụng CPU")
    except:
        device = "cpu"
        compute_type = "int8"
        print("\n⚠️  Sử dụng CPU")
    
    # Load model
    print("\n📦 Loading model...")
    start = time.time()
    
    try:
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        load_time = time.time() - start
        print(f"✅ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"⚠️  Lỗi với {device} + {compute_type}: {e}")
        if device == "cuda":
            print("🔄 Fallback to CPU...")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type
            )
            print("✅ Model loaded on CPU")
        else:
            raise
    
    # Transcribe
    print("\n🎤 Transcribing...")
    start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=None,
        beam_size=5,
        vad_filter=True
    )
    
    # Collect results
    print("\n" + "="*80)
    print("📝 TRANSCRIPT")
    print("="*80)
    
    full_text = []
    for segment in segments:
        timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
        print(f"{timestamp:30} {segment.text}")
        full_text.append(segment.text)
    
    transcribe_time = time.time() - start
    
    # Print info
    print("\n" + "="*80)
    print("📊 INFO")
    print("="*80)
    print(f"Language: {info.language} (confidence: {info.language_probability:.1%})")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Processing time: {transcribe_time:.2f}s")
    print(f"Speed: {transcribe_time/info.duration:.2f}x realtime")
    print(f"Device: {device} ({compute_type})")
    print("="*80)
    print("\nFull transcript:")
    print("-"*80)
    print(" ".join(full_text))
    print("="*80)
    
    return " ".join(full_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_whisper_bf16.py <audio_file> [model_path]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/final/whisper-vi-en-ct2"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    transcribe_bf16(audio_path, model_path)

