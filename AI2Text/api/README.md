# ASR API Documentation

REST API for Vietnamese Speech-to-Text system with support for ASRModelWithTimestamps and BPE tokenizer.

## Quick Start

### 1. Start the API server

```bash
cd /home/alida/Documents/Cursor/AI2Text/AT2Text/AI2Text
python -m api.app_v2
```

Or using uvicorn directly:
```bash
uvicorn api.app_v2:app --host 0.0.0.0 --port 8000
```

### 2. Load a model

```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
    "model_name": "my_model",
    "device": "cuda"
  }'
```

### 3. Transcribe audio

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@path/to/audio.wav" \
  -F "model_name=my_model" \
  -F "return_timestamps=true"
```

## API Endpoints

### POST /load_model
Load a model checkpoint into memory.

**Request Body:**
```json
{
  "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
  "model_name": "my_model",
  "device": "cuda"
}
```

**Response:**
```json
{
  "status": "loaded",
  "model_name": "my_model",
  "checkpoint_path": "/path/to/checkpoint.pt",
  "device": "cuda"
}
```

### POST /transcribe
Transcribe a single audio file.

**Parameters:**
- `audio` (file): Audio file to transcribe
- `model_name` (query, default: "default"): Name of loaded model
- `return_timestamps` (query, default: false): Return word-level timestamps
- `language_id` (query, optional): Language ID (0=Vietnamese, 1=English)

**Response:**
```json
{
  "text": "transcribed text here",
  "timestamps": [
    {"word": "word1", "start": 0.0, "end": 0.5},
    {"word": "word2", "start": 0.5, "end": 1.0}
  ],
  "confidence": null,
  "model_name": "my_model",
  "processing_time": 0.123
}
```

### POST /transcribe_batch
Transcribe multiple audio files.

**Request Body:**
```json
{
  "audio_paths": [
    "/path/to/audio1.wav",
    "/path/to/audio2.wav"
  ],
  "return_timestamps": false,
  "language_id": null
}
```

**Response:**
```json
{
  "results": [
    {"text": "transcription 1"},
    {"text": "transcription 2"}
  ],
  "total_time": 0.456
}
```

### GET /models
List available model checkpoints.

**Response:**
```json
{
  "models": [
    {
      "name": "checkpoint_epoch_45",
      "path": "checkpoints/checkpoint_epoch_45.pt",
      "size_mb": 123.45,
      "full_path": "/full/path/to/checkpoint.pt"
    }
  ],
  "loaded": ["my_model"],
  "total": 1
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["my_model"],
  "device": "cuda"
}
```

### DELETE /models/{model_name}
Unload a model from memory.

**Response:**
```json
{
  "status": "unloaded",
  "model_name": "my_model"
}
```

## Python Client Example

```python
import requests

# Load model
response = requests.post(
    "http://localhost:8000/load_model",
    json={
        "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
        "model_name": "my_model",
        "device": "cuda"
    }
)
print(response.json())

# Transcribe audio
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"audio": f},
        params={
            "model_name": "my_model",
            "return_timestamps": True
        }
    )
    print(response.json())
```

## JavaScript/TypeScript Client Example

```typescript
// Load model
const loadModel = async () => {
  const response = await fetch('http://localhost:8000/load_model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      checkpoint_path: 'checkpoints/checkpoint_epoch_45.pt',
      model_name: 'my_model',
      device: 'cuda'
    })
  });
  return await response.json();
};

// Transcribe audio
const transcribe = async (audioFile: File) => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  
  const response = await fetch(
    'http://localhost:8000/transcribe?model_name=my_model&return_timestamps=true',
    {
      method: 'POST',
      body: formData
    }
  );
  return await response.json();
};
```

## Notes

- Models are loaded into memory and cached. Use `/load_model` before transcribing.
- Supports both BPE and character-level tokenizers based on model config.
- Timestamps are optional and require `return_timestamps=true`.
- Audio formats supported: WAV, MP3, FLAC, M4A, OGG, etc. (via torchaudio/librosa).

