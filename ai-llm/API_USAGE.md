# API Usage Guide

## Quick Start

### 1. Start the API Server

```bash
cd /home/alida/Documents/Cursor/AI2Text/ai-llm
source .venv/bin/activate
PYTHONPATH=/home/alida/Documents/Cursor/AI2Text/ai-llm:$PYTHONPATH python3 -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test the API

Open browser: **http://localhost:8000/docs** for interactive Swagger UI

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/transcribe` | Transcribe audio (server path) |
| POST | `/transcribe/upload` | Upload and transcribe audio |
| POST | `/ask` | Ask question using RAG |

## Examples

### Python Client Library

```python
from src.api.client import AILLMClient

client = AILLMClient(base_url="http://localhost:8000")

# Health check
health = client.health_check()

# Transcribe from server path
result = client.transcribe_file("data/raw/audio/file.wav")

# Upload and transcribe
result = client.transcribe_upload("/local/path/to/audio.wav")

# Ask question
answer = client.ask("What is discussed?", top_k=5)
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Transcribe
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "data/raw/audio/file.wav"}'

# Upload file
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/audio.wav"

# Ask question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question", "top_k": 5}'
```

### JavaScript

```javascript
// Transcribe
const response = await fetch('http://localhost:8000/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_path: 'data/raw/audio/file.wav' })
});
const result = await response.json();

// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:8000/transcribe/upload', {
    method: 'POST',
    body: formData
});

// Ask question
const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'Your question', top_k: 5 })
});
```

## Response Formats

### Transcribe Response
```json
{
  "text": "Full transcribed text...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Segment text..."}
  ],
  "language": "en"
}
```

### Ask Response
```json
{
  "answer": "Generated answer with citations...",
  "contexts": [
    {"id": "doc-id-1", "text": "Source text..."}
  ]
}
```

### Health Response
```json
{
  "status": "healthy",
  "models_loaded": true,
  "index_available": true
}
```

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Server error (check error message)

Error response format:
```json
{
  "detail": "Error message here"
}
```

