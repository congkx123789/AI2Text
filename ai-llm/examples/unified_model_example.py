"""
Ví dụ sử dụng UnifiedModelManager - Kết hợp Whisper + Qwen

UnifiedModelManager quản lý cả Whisper và Qwen models,
cho phép sử dụng chúng cùng lúc một cách dễ dàng.
"""

from pathlib import Path
from src.models.unified import get_unified_manager, UnifiedModelManager

# Cách 1: Sử dụng global singleton (tự động load từ config)
print("=" * 80)
print("Cách 1: Sử dụng global singleton")
print("=" * 80)

manager = get_unified_manager()

# Process audio qua cả Whisper và Qwen
result = manager.process_audio(
    audio_path="data/raw/audio/example.wav",
    task="summarize"
)

print(f"Transcription: {result['transcription']}")
print(f"Summary: {result['response']}")
print(f"Language: {result['language']}")
print(f"ASR Model: {result['asr_model']}")
print(f"GEN Model: {result['gen_model']}")

# Cách 2: Tạo instance riêng với custom config
print("\n" + "=" * 80)
print("Cách 2: Tạo instance riêng")
print("=" * 80)

custom_manager = UnifiedModelManager(
    asr_model="./models/finetuned/whisper-mixed",
    gen_model="./models/finetuned/qwen-mixed",
    asr_device="cuda"
)

result = custom_manager.process_audio(
    audio_path="data/raw/audio/example.wav",
    task="answer",
    question="What is the main topic discussed?"
)

print(f"Transcription: {result['transcription']}")
print(f"Answer: {result['response']}")

# Cách 3: Chỉ sử dụng Whisper (không qua Qwen)
print("\n" + "=" * 80)
print("Cách 3: Chỉ transcribe (không qua Qwen)")
print("=" * 80)

transcription_result = manager.transcribe_only(
    audio_path="data/raw/audio/example.wav"
)

print(f"Text: {transcription_result['text']}")
print(f"Language: {transcription_result['language']}")
print(f"Segments: {len(transcription_result['segments'])} segments")

# Cách 4: Chỉ sử dụng Qwen (không cần audio)
print("\n" + "=" * 80)
print("Cách 4: Chỉ generate với Qwen")
print("=" * 80)

text = "This is a sample text that needs to be summarized."
summary = manager.generate_only(text, task="summarize")
print(f"Original: {text}")
print(f"Summary: {summary}")

# Cách 5: Sử dụng các task khác nhau
print("\n" + "=" * 80)
print("Cách 5: Các task khác nhau")
print("=" * 80)

audio_path = "data/raw/audio/example.wav"

# Analyze
result = manager.process_audio(audio_path, task="analyze")
print(f"Analysis: {result['response']}")

# Extract key info
result = manager.process_audio(audio_path, task="extract")
print(f"Key Info: {result['response']}")

# Translate
result = manager.process_audio(audio_path, task="translate")
print(f"Translation: {result['response']}")

