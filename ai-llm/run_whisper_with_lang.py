#!/usr/bin/env python3
"""
Whisper với chỉ định ngôn ngữ rõ ràng (vi hoặc en)
"""
import sys
import os
from pathlib import Path

# Set environment variables
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from faster_whisper import WhisperModel
import time

def transcribe_with_language(audio_path, model_path="models/final/whisper-vi-en-ct2", language=None):
    """
    Transcribe với chỉ định ngôn ngữ
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model
        language: 'vi', 'en', or None (auto-detect)
    """
    print("="*80)
    print("🎤 WHISPER TRANSCRIPTION")
    print("="*80)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print(f"Language: {language or 'auto-detect'}")
    print("="*80)
    
    # Device
    device = "cpu"
    compute_type = "int8"
    print("\n⚠️  Sử dụng CPU (an toàn)")
    
    # Load model
    print("\n📦 Loading model...")
    start = time.time()
    model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type
    )
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    # Transcribe
    print("\n🎤 Transcribing...")
    start = time.time()
    
    segments, info = model.transcribe(
        audio_path,
        language=language,  # Chỉ định ngôn ngữ
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
    if language:
        print(f"Language: {info.language} (FORCED - bạn đã chỉ định)")
    else:
        print(f"Language: {info.language} (auto-detected, confidence: {info.language_probability:.1%})")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Processing time: {transcribe_time:.2f}s")
    print(f"Speed: {transcribe_time/info.duration:.2f}x realtime")
    print("="*80)
    print("\nFull transcript:")
    print("-"*80)
    print(" ".join(full_text))
    print("="*80)
    
    return " ".join(full_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_whisper_with_lang.py <audio_file> [language] [model_path]")
        print()
        print("Language options:")
        print("  vi  - Tiếng Việt")
        print("  en  - Tiếng Anh")
        print("  None - Auto-detect (mặc định)")
        print()
        print("Examples:")
        print("  python run_whisper_with_lang.py audio.wav vi")
        print("  python run_whisper_with_lang.py audio.wav en")
        print("  python run_whisper_with_lang.py audio.wav  # auto-detect")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    model_path = sys.argv[3] if len(sys.argv) > 3 else "models/final/whisper-vi-en-ct2"
    
    # Validate language
    if language and language not in ['vi', 'en']:
        print(f"⚠️  Language '{language}' không hợp lệ. Sử dụng 'vi' hoặc 'en'")
        print("   Hoặc bỏ trống để auto-detect")
        language = None
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    transcribe_with_language(audio_path, model_path, language)

