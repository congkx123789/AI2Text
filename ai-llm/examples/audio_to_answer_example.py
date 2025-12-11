"""
Ví dụ sử dụng endpoint /audio-to-answer - Kết hợp Whisper + Qwen

Endpoint này:
1. Nhận audio file
2. Whisper transcribe → text
3. Qwen xử lý text (summarize, answer, translate, analyze, extract)
"""

from src.api.client import create_client

# Khởi tạo client
client = create_client("http://localhost:8000")

# Ví dụ 1: Summarize audio
print("=" * 80)
print("Ví dụ 1: Summarize audio")
print("=" * 80)

result = client.audio_to_answer_upload(
    file_path="data/raw/audio/example.wav",
    task="summarize"
)

print(f"Transcription: {result['transcription']}")
print(f"\nSummary: {result['response']}")
print(f"Language: {result['language']}")

# Ví dụ 2: Answer question từ audio
print("\n" + "=" * 80)
print("Ví dụ 2: Answer question từ audio")
print("=" * 80)

result = client.audio_to_answer_upload(
    file_path="data/raw/audio/example.wav",
    task="answer",
    question="What is the main topic discussed?"
)

print(f"Transcription: {result['transcription']}")
print(f"\nAnswer: {result['response']}")

# Ví dụ 3: Analyze audio content
print("\n" + "=" * 80)
print("Ví dụ 3: Analyze audio content")
print("=" * 80)

result = client.audio_to_answer_upload(
    file_path="data/raw/audio/example.wav",
    task="analyze"
)

print(f"Transcription: {result['transcription']}")
print(f"\nAnalysis: {result['response']}")

# Ví dụ 4: Extract key information
print("\n" + "=" * 80)
print("Ví dụ 4: Extract key information")
print("=" * 80)

result = client.audio_to_answer_upload(
    file_path="data/raw/audio/example.wav",
    task="extract"
)

print(f"Transcription: {result['transcription']}")
print(f"\nKey Information: {result['response']}")

# Ví dụ 5: Sử dụng với file path trên server
print("\n" + "=" * 80)
print("Ví dụ 5: Sử dụng với file path trên server")
print("=" * 80)

result = client.audio_to_answer_file(
    audio_path="data/raw/audio/example.wav",
    task="summarize"
)

print(f"Transcription: {result['transcription']}")
print(f"\nSummary: {result['response']}")

