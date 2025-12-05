# AI-LLM: Speech-to-Text + RAG Pipeline API

## 📖 Giới thiệu

**AI-LLM** là một hệ thống tích hợp chuyển đổi giọng nói thành văn bản (ASR) và RAG (Retrieval Augmented Generation) để tạo ra một API mạnh mẽ cho việc:

- 🎤 **Chuyển đổi audio thành text** - Sử dụng Whisper model để transcribe các file audio
- 🔍 **Tìm kiếm thông minh** - Hybrid search kết hợp BM25 (lexical) và FAISS (vector) 
- 💬 **Trả lời câu hỏi** - RAG pipeline trả lời câu hỏi dựa trên transcripts với citations
- 🌐 **REST API** - API đơn giản, dễ sử dụng từ bất kỳ ứng dụng nào

### Kiến trúc hệ thống

```
Audio Files → ASR (Whisper) → Transcripts → Index (BM25 + FAISS) → RAG Pipeline → API
```

1. **ASR Pipeline**: Chuyển đổi audio files thành transcripts sử dụng faster-whisper
2. **Indexing**: Build hybrid index từ transcripts (BM25 cho keyword search + FAISS cho semantic search)
3. **RAG Pipeline**: Retrieve → Rerank → Generate với citations
4. **API Server**: FastAPI server cung cấp endpoints để sử dụng từ bất kỳ ứng dụng nào

## 🎯 Tính năng chính

- ✅ **Speech-to-Text**: Chuyển đổi audio (wav, mp3, m4a, flac) thành text với timestamps
- ✅ **Hybrid Search**: Kết hợp lexical (BM25) và semantic (FAISS) search
- ✅ **Question Answering**: Trả lời câu hỏi với citations từ transcripts
- ✅ **File Upload**: Upload và transcribe audio files trực tiếp qua API
- ✅ **REST API**: Standard REST API với Swagger UI documentation
- ✅ **Client Library**: Python client library sẵn có để dễ dàng tích hợp

## 📋 Yêu cầu hệ thống

- **Python**: 3.10 hoặc cao hơn
- **pip**: Package manager
- **ffmpeg**: Cho xử lý audio (cài đặt bên dưới)

### Cài đặt ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Tải từ [ffmpeg.org](https://ffmpeg.org/download.html) và thêm vào PATH

## 🚀 Cài đặt và Setup

### Bước 1: Clone và vào thư mục project

```bash
cd /home/alida/Documents/Cursor/AI2Text/ai-llm
```

### Bước 2: Tạo virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies

```bash
# Cài đặt project ở chế độ editable (cho phép import src.*)
pip install -e .

# Cài đặt tất cả packages cần thiết
pip install "faster-whisper[onnx]" huggingface_hub fastapi "uvicorn[standard]" python-dotenv sentence-transformers transformers faiss-cpu rank-bm25 python-multipart requests
```

### Bước 4: Tải models (một lần duy nhất)

Tải tất cả models cần thiết từ HuggingFace:

```bash
python3 scripts/download_models.py
```

Script này sẽ tự động tải:
- **Whisper Small** (`openai/whisper-small`) - Model ASR cho speech-to-text
- **all-MiniLM-L6-v2** - Model embedding cho vector search
- **ms-marco-MiniLM-L-6-v2** - Model reranker
- **Qwen2.5-0.5B-Instruct** - LLM generator

Models sẽ được lưu tại `models/base/` và có thể sử dụng offline sau lần đầu tiên.

### Bước 5: Chuẩn bị dữ liệu

Nếu bạn chưa có file `data/processed/transcripts.jsonl`:

1. **Tạo manifest từ audio files:**
```bash
python3 scripts/prepare_data.py --in data/raw --out data/interim/manifest.jsonl
```

2. **Chuyển đổi audio thành transcripts:**
```bash
# Sử dụng ASR bridge để transcribe
python3 -m src.tools.ai2text_bridge --manifest data/interim/manifest.jsonl --out data/processed/transcripts.jsonl
```

### Bước 6: Build index

Build hybrid index từ transcripts:

```bash
python3 scripts/build_index.py --in data/processed/transcripts.jsonl --out vectorstore
```

Index sẽ được lưu tại `vectorstore/` và có thể tái sử dụng.

### Bước 7: Khởi động API Server

```bash
# Cách 1: Sử dụng script (khuyến nghị)
bash scripts/run_api.sh

# Cách 2: Chạy trực tiếp với uvicorn
source .venv/bin/activate
PYTHONPATH=/home/alida/Documents/Cursor/AI2Text/ai-llm:$PYTHONPATH python3 -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại: **http://localhost:8000**

**Kiểm tra server đã chạy:**
```bash
curl http://localhost:8000/
# Hoặc mở browser: http://localhost:8000/docs
```

## 🎮 Sử dụng API để điều khiển Model

### 1. Swagger UI (Khuyến nghị cho testing)

Mở trình duyệt và truy cập: **http://localhost:8000/docs**

Swagger UI cung cấp:
- ✅ Interactive API documentation
- ✅ Test endpoints trực tiếp từ browser
- ✅ Xem request/response schemas
- ✅ Upload files dễ dàng

### 2. API Endpoints

#### Health Check

Kiểm tra trạng thái server và models:

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

#### Transcribe Audio (từ server path)

Chuyển đổi audio file thành text (file phải có trên server):

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "data/raw/audio/your-file.wav"}'
```

**Response:**
```json
{
  "text": "Full transcribed text here...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "First segment text..."
    }
  ],
  "language": "en"
}
```

#### Transcribe Audio (upload file) ⭐ Khuyến nghị

Upload và chuyển đổi audio file - **Không cần file trên server**:

```bash
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/your/audio.wav"
```

**Hỗ trợ formats:** wav, mp3, m4a, flac, và các format audio khác

**Response:** Giống như endpoint `/transcribe` ở trên

#### Ask Question (RAG Pipeline)

Đặt câu hỏi dựa trên transcripts đã được index:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic discussed?",
    "top_k": 5
  }'
```

**Parameters:**
- `query` (required): Câu hỏi của bạn
- `top_k` (optional, default: 5): Số lượng kết quả top để retrieve

**Response:**
```json
{
  "answer": "Generated answer with citations [1], [2]...",
  "contexts": [
    {
      "id": "doc-id-1",
      "text": "Relevant source text from transcripts..."
    },
    {
      "id": "doc-id-2",
      "text": "Another relevant chunk..."
    }
  ]
}
```

## 💻 Sử dụng từ các ứng dụng khác

### Python - Client Library (Khuyến nghị)

Sử dụng client library có sẵn để dễ dàng tích hợp:

```python
from src.api.client import AILLMClient

# Khởi tạo client
client = AILLMClient(base_url="http://localhost:8000")

# 1. Kiểm tra health
health = client.health_check()
print(f"Status: {health['status']}")
print(f"Models loaded: {health['models_loaded']}")

# 2. Transcribe từ server path
result = client.transcribe_file("data/raw/audio/file.wav")
print(f"Transcribed: {result['text']}")
print(f"Language: {result['language']}")

# 3. Upload và transcribe local file
result = client.transcribe_upload("/local/path/to/audio.wav")
print(f"Transcribed: {result['text']}")

# 4. Hỏi câu hỏi với RAG
answer = client.ask("What is discussed in the transcripts?", top_k=5)
print(f"Answer: {answer['answer']}")
print(f"Found {len(answer['contexts'])} citations")
for i, cite in enumerate(answer['contexts'], 1):
    print(f"  [{i}] {cite['text'][:100]}...")
```

**Xem ví dụ đầy đủ:** `examples/client_example.py`

### Python - Plain Requests

Nếu không muốn dùng client library:

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Transcribe từ server path
response = requests.post(
    f"{BASE_URL}/transcribe",
    json={"audio_path": "data/raw/audio/file.wav"}
)
result = response.json()
print(result['text'])

# Upload và transcribe
with open("audio.wav", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/transcribe/upload", files=files)
    result = response.json()
    print(result['text'])

# Ask question
response = requests.post(
    f"{BASE_URL}/ask",
    json={"query": "Your question here", "top_k": 5}
)
answer = response.json()
print(answer['answer'])
```

**Xem ví dụ đầy đủ:** `examples/python_requests_example.py`

### JavaScript/Node.js

```javascript
const BASE_URL = 'http://localhost:8000';

// 1. Health check
async function checkHealth() {
    const response = await fetch(`${BASE_URL}/health`);
    const data = await response.json();
    console.log('Health:', data);
    return data;
}

// 2. Transcribe từ server path
async function transcribeFile(audioPath) {
    const response = await fetch(`${BASE_URL}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_path: audioPath })
    });
    const data = await response.json();
    return data;
}

// 3. Upload và transcribe (Browser)
async function transcribeUpload(fileInput) {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const response = await fetch(`${BASE_URL}/transcribe/upload`, {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    return data;
}

// 4. Ask question
async function askQuestion(query, topK = 5) {
    const response = await fetch(`${BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, top_k: topK })
    });
    const data = await response.json();
    return data;
}

// Sử dụng
(async () => {
    const health = await checkHealth();
    console.log('API Status:', health.status);
    
    const transcript = await transcribeFile('data/raw/audio/file.wav');
    console.log('Transcribed:', transcript.text);
    
    const answer = await askQuestion('What is the main topic?');
    console.log('Answer:', answer.answer);
})();
```

**Xem ví dụ đầy đủ:** `examples/javascript_example.js`

### cURL Commands

```bash
# Health check
curl http://localhost:8000/health

# Transcribe từ server path
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "data/raw/audio/file.wav"}'

# Upload và transcribe
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/audio.wav"

# Ask question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question", "top_k": 5}'
```

**Xem script ví dụ:** `examples/curl_examples.sh`

## ⚙️ Cấu hình

Tạo file `.env` trong thư mục gốc để cấu hình:

```env
# Model paths (sau khi download models)
ASR_MODEL=./models/base/whisper-small
EMBEDDING_MODEL=./models/base/embedder
RERANKER_MODEL=./models/base/reranker
GEN_MODEL=./models/base/generator

# Directories
DATA_DIR=./data
VECTOR_DIR=./vectorstore
MODELS_DIR=./models

# ASR settings
ASR_DEVICE=auto      # auto, cpu, cuda
ASR_COMPUTE=float16  # float16, float32, int8

# Generation settings
GEN_MAX_TOKENS=512
```

## 📁 Cấu trúc Project

```
ai-llm/
├── data/
│   ├── raw/          # Audio files gốc
│   ├── interim/       # Manifest files
│   └── processed/     # Transcripts JSONL
├── models/
│   └── base/          # Downloaded models (Whisper, Embedder, Reranker, Generator)
├── vectorstore/       # Built index (BM25 + FAISS)
├── src/
│   ├── api/           # FastAPI server và client
│   │   ├── server.py  # API endpoints
│   │   └── client.py  # Python client library
│   ├── llm/           # LLM inference & training
│   ├── rag/           # RAG pipeline (retriever, reranker, indexer)
│   ├── tools/         # Utilities (ASR bridge, etc.)
│   └── config.py       # Configuration
├── scripts/
│   ├── download_models.py  # Download models từ HuggingFace
│   ├── build_index.py      # Build hybrid index
│   ├── prepare_data.py     # Tạo manifest từ audio
│   └── run_api.sh          # Script khởi động server
├── examples/
│   ├── client_example.py           # Python client library example
│   ├── python_requests_example.py  # Plain requests example
│   ├── javascript_example.js       # JavaScript example
│   └── curl_examples.sh            # cURL examples
└── README.md
```

## 🔧 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'src'`

**Giải pháp:**
```bash
# 1. Cài đặt project
pip install -e .

# 2. Set PYTHONPATH khi chạy
export PYTHONPATH=/home/alida/Documents/Cursor/AI2Text/ai-llm:$PYTHONPATH
# Hoặc
PYTHONPATH=/home/alida/Documents/Cursor/AI2Text/ai-llm:$PYTHONPATH python3 -m uvicorn ...
```

### Lỗi: `Vector store not found`

**Giải pháp:**
```bash
# Build lại index
python3 scripts/build_index.py --in data/processed/transcripts.jsonl --out vectorstore
```

### Lỗi: `faster_whisper` không tìm thấy

**Giải pháp:**
```bash
pip install "faster-whisper[onnx]"
```

### Server không khởi động

**Kiểm tra:**
1. Virtual environment đã được activate: `source .venv/bin/activate`
2. Port 8000 chưa bị sử dụng: `lsof -i :8000` hoặc `netstat -an | grep 8000`
3. Tất cả dependencies đã được cài đặt: `pip list | grep fastapi`

### Models chưa được tải

**Giải pháp:**
```bash
# Tải lại models
python3 scripts/download_models.py
```

### API trả về lỗi 500

**Kiểm tra:**
1. Models đã được tải: `ls models/base/`
2. Index đã được build: `ls vectorstore/`
3. Health check: `curl http://localhost:8000/health`

## 📝 Ghi chú quan trọng

- ✅ **Models offline**: Sau lần đầu tải, models được lưu local và có thể chạy offline
- ✅ **Index tái sử dụng**: Index được build một lần và có thể tái sử dụng
- ✅ **Hot reload**: Server tự động reload khi code thay đổi (development mode)
- ✅ **CORS enabled**: API cho phép requests từ bất kỳ origin nào (có thể cấu hình trong production)
- ✅ **File upload**: Hỗ trợ upload files trực tiếp qua API, không cần file trên server

## 🚀 Production Deployment

Để deploy trong production:

1. **Tắt hot-reload:**
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

2. **Sử dụng process manager** (PM2, systemd, etc.)

3. **Cấu hình CORS** trong `src/api/server.py` nếu cần

4. **Sử dụng reverse proxy** (nginx, Apache) với SSL

## 📚 Tài liệu thêm

- **API Documentation**: http://localhost:8000/docs (khi server chạy)
- **API Usage Guide**: Xem `API_USAGE.md`
- **Design Overview**: Xem `docs/design.md`

## 📄 License

[Thêm license của bạn ở đây]

## 👥 Contributors

[Thêm contributors ở đây]

---

**Happy coding! 🎉**

Nếu có vấn đề, hãy kiểm tra phần Troubleshooting hoặc mở issue trên repository.
