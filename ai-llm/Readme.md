# AI-LLM: Speech-to-Text + LLM Pipeline

Hệ thống tích hợp chuyển đổi giọng nói thành văn bản (ASR) và Large Language Model (LLM) để tạo ra một pipeline mạnh mẽ cho việc xử lý audio và tạo phản hồi thông minh.

## 📖 Tổng Quan

**AI-LLM** là một hệ thống end-to-end kết hợp:
- 🎤 **Whisper ASR** (fine-tuned) - Chuyển đổi audio thành text với độ chính xác cao
- 💬 **Qwen LLM** (fine-tuned) - Xử lý và tạo phản hồi từ text
- 🔍 **RAG Pipeline** - Retrieval Augmented Generation cho question answering
- 🌐 **REST API** - API đơn giản, dễ sử dụng

## 🏗️ Kiến Trúc Hệ Thống

### Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                             │
│  (Web App, Mobile App, CLI, Python Client)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    API SERVER (FastAPI)                     │
│  - /transcribe          - /audio-to-answer                 │
│  - /ask                 - /health                           │
└──────────────┬───────────────────────┬─────────────────────┘
               │                       │
               ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  ASR Pipeline    │    │  LLM Pipeline    │
    │  (Whisper)       │    │  (Qwen)          │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED MODEL MANAGER                          │
│  - Whisper Model (CTranslate2)                             │
│  - Qwen Model (Merged)                                     │
│  - Lazy Loading & Caching                                  │
└─────────────────────────────────────────────────────────────┘
             │                       │
             ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  Whisper ASR     │    │  Qwen LLM         │
│  - CTranslate2   │    │  - Merged Model   │
│  - Fine-tuned    │    │  - Fine-tuned     │
│  - EN + VI       │    │  - Multi-task     │
└──────────────────┘    └──────────────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────────────────────┐
    │      RAG Pipeline (Optional)      │
    │  - Hybrid Retriever (BM25+FAISS) │
    │  - Reranker                      │
    │  - Vector Store                  │
    └──────────────────────────────────┘
```

### Chi Tiết Kiến Trúc

#### 1. API Server Layer
- **Framework**: FastAPI
- **Endpoints**: RESTful API với Swagger UI
- **Features**: 
  - File upload support
  - CORS enabled
  - Health check
  - Error handling

#### 2. Model Layer
- **Whisper ASR**:
  - Format: CTranslate2 (nhanh hơn 5x)
  - Base: `openai/whisper-small`
  - Fine-tuned: Hỗ trợ EN + VI
  - Location: `models/final/whisper-vi-en-ct2/`
  
- **Qwen LLM**:
  - Format: Merged model (không cần PEFT)
  - Base: `Qwen2.5-0.5B-Instruct`
  - Fine-tuned: Multi-task (summarize, answer, translate, etc.)
  - Location: `models/finetuned/qwen-mixed-merged/`

#### 3. Unified Model Manager
- Quản lý cả Whisper và Qwen
- Lazy loading (load khi cần)
- Caching để tối ưu performance
- Auto-detect model format (CTranslate2, HuggingFace, LoRA)

#### 4. RAG Pipeline (Optional)
- Hybrid search: BM25 (lexical) + FAISS (semantic)
- Reranking với cross-encoder
- Vector store: FAISS index

### Data Flow

#### Flow 1: Audio → Text (Transcribe)
```
Audio File
  ↓
API Endpoint: /transcribe
  ↓
Unified Model Manager
  ↓
Whisper ASR (CTranslate2)
  ↓
Text + Segments + Language
  ↓
API Response
```

#### Flow 2: Audio → Answer (Whisper + Qwen)
```
Audio File
  ↓
API Endpoint: /audio-to-answer
  ↓
Unified Model Manager
  ↓
┌─────────────┬─────────────┐
│             │             │
Whisper ASR   →   Text      →   Qwen LLM
│             │             │
└─────────────┴─────────────┘
  ↓
Transcription + Response
  ↓
API Response
```

#### Flow 3: Question → Answer (RAG)
```
Question
  ↓
API Endpoint: /ask
  ↓
RAG Pipeline
  ↓
┌─────────────┬─────────────┬─────────────┐
│             │             │             │
Hybrid        →   Rerank    →   Qwen LLM
Retriever     │             │             │
(BM25+FAISS)  │             │             │
└─────────────┴─────────────┴─────────────┘
  ↓
Answer + Citations
  ↓
API Response
```

## 🚀 Cài Đặt

### Yêu Cầu
- Python 3.10+
- ffmpeg 4.4+
- GPU (khuyến nghị): RTX 5060 Ti hoặc tương đương

### Quick Start

```bash
# Clone và setup
git clone <repo-url>
cd ai-llm
python3 -m venv .venv
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt

# Setup config
cp .env.example .env
# Chỉnh sửa .env với model paths

# Start API
python3 -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

## 📡 Sử Dụng API

### Khởi Động API Server

```bash
# Cách 1: Direct
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# Cách 2: Script (nếu có)
./scripts/start_api.sh
```

API sẽ chạy tại:
- **Base URL**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Endpoints Chi Tiết

#### 1. Health Check
Kiểm tra trạng thái API và models

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "index_available": true
}
```

#### 2. Transcribe Audio (File Path)
Transcribe audio file từ server path

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "data/raw/audio/example.wav"
  }'
```

**Response:**
```json
{
  "text": "Transcribed text here...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "First segment"
    }
  ],
  "language": "en"
}
```

#### 3. Transcribe Audio (Upload)
Upload và transcribe audio file

```bash
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@audio.wav" \
  -F "model_size=small" \
  -F "language=vi"
```

**Parameters:**
- `file` (required): Audio file
- `model_size` (optional): `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
- `language` (optional): Language code (`vi`, `en`, `None` for auto-detect)

**Response:** Giống endpoint `/transcribe`

#### 4. Audio to Answer (File Path)
Kết hợp Whisper + Qwen: Transcribe và xử lý

```bash
curl -X POST "http://localhost:8000/audio-to-answer" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "data/raw/audio/example.wav",
    "task": "summarize",
    "question": null
  }'
```

**Parameters:**
- `audio_path` (required): Path to audio file
- `task` (optional): `summarize`, `answer`, `translate`, `analyze`, `extract` (default: `summarize`)
- `question` (optional): Question nếu `task="answer"`

**Response:**
```json
{
  "transcription": "Full transcribed text...",
  "response": "Summary or answer from Qwen...",
  "language": "en",
  "task": "summarize"
}
```

#### 5. Audio to Answer (Upload)
Upload audio và xử lý với Whisper + Qwen

```bash
curl -X POST "http://localhost:8000/audio-to-answer/upload" \
  -F "file=@audio.wav" \
  -F "task=summarize" \
  -F "question=What is the main topic?"
```

**Parameters:**
- `file` (required): Audio file
- `task` (optional): Task type (default: `summarize`)
- `question` (optional): Question nếu `task="answer"`
- `language` (optional): Language code
- `model_size` (optional): Whisper model size

**Response:** Giống endpoint `/audio-to-answer`

#### 6. Ask Question (RAG)
Hỏi câu hỏi dựa trên transcripts đã được index

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is discussed in the transcripts?",
    "top_k": 5
  }'
```

**Parameters:**
- `query` (required): Câu hỏi
- `top_k` (optional): Số kết quả top (default: 5)

**Response:**
```json
{
  "answer": "Answer from Qwen with citations...",
  "contexts": [
    {
      "id": "doc1",
      "text": "Relevant context text..."
    }
  ]
}
```

### Python Client

#### Cài đặt Client

```python
from src.api.client import create_client

# Tạo client
client = create_client("http://localhost:8000")
```

#### Examples

**1. Transcribe Audio:**
```python
# Upload và transcribe
result = client.transcribe_upload("audio.wav")
print(f"Text: {result['text']}")
print(f"Language: {result['language']}")

# Hoặc từ file path
result = client.transcribe_file("data/raw/audio/example.wav")
```

**2. Audio to Answer (Whisper + Qwen):**
```python
# Summarize
result = client.audio_to_answer_upload(
    file_path="audio.wav",
    task="summarize"
)
print(f"Transcription: {result['transcription']}")
print(f"Summary: {result['response']}")

# Answer question
result = client.audio_to_answer_upload(
    file_path="audio.wav",
    task="answer",
    question="What is the main topic?"
)
print(f"Answer: {result['response']}")

# Analyze
result = client.audio_to_answer_upload(
    file_path="audio.wav",
    task="analyze"
)
```

**3. Ask Question (RAG):**
```python
result = client.ask(
    query="What is discussed in the transcripts?",
    top_k=5
)
print(f"Answer: {result['answer']}")
print(f"Citations: {result['contexts']}")
```

### Unified Model Manager (Direct Usage)

Sử dụng trực tiếp không qua API:

```python
from src.models.unified import get_unified_manager

# Tạo manager
manager = get_unified_manager()

# Process audio qua Whisper + Qwen
result = manager.process_audio(
    audio_path="audio.wav",
    task="summarize"
)

print(result['transcription'])  # Text từ Whisper
print(result['response'])       # Response từ Qwen
print(result['language'])       # Detected language

# Chỉ transcribe (không qua Qwen)
transcription = manager.transcribe_only("audio.wav")
print(transcription['text'])

# Chỉ generate (không cần audio)
summary = manager.generate_only(
    text="Long text here...",
    task="summarize"
)
```

### Task Types cho Qwen

| Task | Mô tả | Example |
|------|-------|---------|
| `summarize` | Tóm tắt nội dung | "Summarize this text..." |
| `answer` | Trả lời câu hỏi | Cần thêm `question` parameter |
| `translate` | Dịch sang tiếng Anh | "Translate to English..." |
| `analyze` | Phân tích nội dung | "Analyze this text..." |
| `extract` | Trích xuất thông tin | "Extract key points..." |

## ⚙️ Configuration

Tạo file `.env`:

```env
# Fine-tuned Models
ASR_FINETUNED_MODEL=./models/final/whisper-vi-en-ct2
GEN_FINETUNED_MODEL=./models/finetuned/qwen-mixed-merged

# Device Configuration
ASR_DEVICE=auto          # auto, cuda, cpu
ASR_COMPUTE=float16      # float16, int8_float16, int8

# Generation Configuration
GEN_MAX_TOKENS=512

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_DIR=./vectorstore

# Data Directories
DATA_DIR=./data
MODELS_DIR=./models
```

## 📊 Performance

### Whisper ASR
- **CTranslate2 format**: ~5x nhanh hơn HuggingFace
- **GPU (RTX 5060 Ti)**: ~1.4 samples/second
- **CPU**: ~0.3 samples/second
- **Model size**: 463MB (float16)

### Qwen LLM
- **Merged model**: Không cần PEFT, inference nhanh hơn
- **GPU**: ~50-100 tokens/second
- **CPU**: ~10-20 tokens/second
- **Model size**: 513MB

## 🐛 Troubleshooting

### API không start
```bash
# Kiểm tra port
lsof -i :8000

# Kiểm tra dependencies
pip install -r requirements.txt
```

### Models không load
```bash
# Kiểm tra paths
ls -la models/final/whisper-vi-en-ct2/
ls -la models/finetuned/qwen-mixed-merged/

# Kiểm tra .env
cat .env
```

### Out of Memory
- Giảm batch size
- Dùng CPU: `ASR_DEVICE=cpu`
- Dùng quantization thấp: `ASR_COMPUTE=int8`

## 📚 Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Fine-tuning models
- [WHISPER_CONVERSION_GUIDE.md](WHISPER_CONVERSION_GUIDE.md) - Convert Whisper
- [UNIFIED_MODEL_GUIDE.md](UNIFIED_MODEL_GUIDE.md) - Unified Manager

## 📝 License

[Your License]

---

**Made with ❤️ for AI/ML enthusiasts**
