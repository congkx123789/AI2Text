# ai-llm-ss — Minimal Speech-to-Text (ASR) from Scratch

This is a tiny end-to-end project: audio (.wav) + transcripts (.txt) → train a CTC model → serve a FastAPI `/transcribe` endpoint.

## Quickstart

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Training

```bash
# Train the model (1 epoch with checkpoint support)
python scripts/train_from_config.py config/train_merged.json

# Or train directly
python -m src.asr.train_ctc \
    --manifest data/processed/merged_dataset/train/manifest.csv \
    --audio_root data/processed/merged_dataset/train \
    --timestamps data/processed/merged_dataset/train/timestamps.json \
    --trim_segments \
    --vocab data/processed/vocab.json \
    --epochs 1 \
    --batch_size 32 \
    --device auto \
    --amp
```

### Resume Training

If training is interrupted, you can resume from a checkpoint:

```bash
python -m src.asr.train_ctc \
    --resume data/results/checkpoints/checkpoint_epoch_1.pt \
    --epochs 2 \
    [other arguments...]
```

### API Server

Start the API server:

```bash
# Using the serve script (recommended)
python scripts/serve_asr.py

# Or directly with uvicorn
uvicorn src.asr.api:app --host 127.0.0.1 --port 8001 --reload
```

The API will be available at:
- **API**: http://127.0.0.1:8001
- **Interactive Docs**: http://127.0.0.1:8001/docs
- **Health Check**: http://127.0.0.1:8001/health

## API Endpoints

### `POST /transcribe`
Transcribe an audio file to text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (audio file - WAV, MP3, FLAC, etc.)

**Response:**
```json
{
  "text": "transcribed text here",
  "duration": 5.23
}
```

**Example using curl:**
```bash
curl -X POST "http://127.0.0.1:8001/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

**Example using Python:**
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8001/transcribe",
        files={"file": f}
    )
result = response.json()
print(result["text"])
```

### `GET /health`
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "vocab_size": 40
}
```

### `GET /model/info`
Get detailed model information.

**Response:**
```json
{
  "model_path": "/path/to/asr_ctc.pt",
  "model_exists": true,
  "device": "cuda",
  "vocab_size": 40,
  "total_parameters": 1234567,
  "trainable_parameters": 1234567,
  "model_type": "CRNNCTC"
}
```

## Testing the API

Use the test script to verify the API is working:

```bash
# Test health and model info
python scripts/test_api.py

# Test transcription with an audio file
python scripts/test_api.py path/to/audio.wav
```

## Project Structure

```
ai-llm-ss/
├── config/
│   └── train_merged.json      # Training configuration
├── data/
│   ├── processed/             # Processed datasets
│   └── results/
│       ├── asr_ctc.pt        # Final trained model
│       └── checkpoints/       # Training checkpoints
├── scripts/
│   ├── train_from_config.py  # Train using config file
│   ├── serve_asr.py          # Start API server
│   └── test_api.py           # Test API endpoints
└── src/asr/
    ├── api.py                # FastAPI application
    ├── model.py              # CRNNCTC model
    ├── train_ctc.py          # Training script
    └── ...
```
