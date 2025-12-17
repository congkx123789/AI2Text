# AI2Text - Multi-Model Speech-to-Text System

A comprehensive multi-architecture Automatic Speech Recognition (ASR) system supporting multiple model types, training pipelines, and deployment options.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Model Comparison](#model-comparison)
- [Requirements](#requirements)
- [Documentation](#documentation)

## 🎯 Overview

**AI2Text** is a unified platform for speech-to-text conversion with support for multiple ASR architectures:

- **Transformer-based ASR** (`AI2Text/`) - Custom Transformer architecture with CTC/Attention
- **Whisper-based ASR** (`ai-llm/`) - Fine-tuned Whisper models with LLM integration
- **CTC-based ASR** (`ai-llm-ss/`) - CRNN-CTC architecture from scratch
- **Audio Processing** (`audio_processing/`) - Preprocessing and enhancement utilities
- **Web Frontend** (`frontend/`) - Interactive web interface

## 📁 Project Structure

```
AI2Text/
├── AI2Text/                    # Transformer-based ASR System
│   ├── api/                    # FastAPI REST API
│   │   ├── app.py             # Main API server
│   │   ├── asr_service.py      # ASR service wrapper
│   │   └── example_client.py  # API client examples
│   ├── models/                 # Model architectures
│   │   ├── asr_base.py        # Base ASR model
│   │   ├── asr_with_timestamps.py  # Timestamp-aware model
│   │   └── modern_components.py    # Modern components
│   ├── training/               # Training pipeline
│   │   ├── train.py           # Main training script
│   │   ├── dataset.py         # Dataset loaders
│   │   ├── callbacks.py       # Training callbacks
│   │   └── evaluate.py        # Evaluation utilities
│   ├── preprocessing/          # Data preprocessing
│   │   ├── audio_processing.py    # Audio feature extraction
│   │   ├── bpe_tokenizer.py       # BPE tokenization
│   │   ├── sentencepiece_tokenizer.py  # SentencePiece
│   │   └── text_cleaning.py        # Text normalization
│   ├── decoding/               # Decoding algorithms
│   │   ├── beam_search.py     # Beam search decoder
│   │   ├── lm_decoder.py      # Language model decoder
│   │   └── confidence.py      # Confidence scoring
│   ├── checkpoints/            # Model checkpoints
│   │   ├── best_model.pt      # Best trained model
│   │   └── checkpoint_epoch_*.pt  # Epoch checkpoints
│   ├── data/                   # Dataset storage
│   ├── configs/                 # Configuration files
│   └── tests/                  # Unit tests
│
├── ai-llm/                     # Whisper + LLM Pipeline
│   ├── src/
│   │   ├── api/                # FastAPI server
│   │   │   └── server.py      # Main API (port 8002/8003)
│   │   ├── tools/              # Core tools
│   │   │   └── ai2text_bridge.py  # Whisper transcription
│   │   ├── llm/                # LLM components
│   │   │   ├── infer.py        # Text generation
│   │   │   └── load.py         # Model loading
│   │   ├── rag/                # RAG pipeline
│   │   │   ├── pipeline.py    # RAG pipeline
│   │   │   ├── retriever.py   # Document retrieval
│   │   │   └── reranker.py    # Result reranking
│   │   └── eval/               # Evaluation scripts
│   ├── models/
│   │   ├── base/               # Base models
│   │   │   └── whisper-small/  # Original Whisper (HF format)
│   │   ├── final/               # Final trained models
│   │   │   └── whisper-vi-en-ct2/  # Fine-tuned Whisper (CT2)
│   │   └── finetuned/          # Fine-tuned models
│   │       ├── whisper-mixed/  # Whisper fine-tuned
│   │       └── qwen-mixed-merged/  # Qwen LLM merged
│   ├── scripts/                # Training & utility scripts
│   │   ├── train_whisper.py   # Whisper training
│   │   ├── train_llm.py       # LLM training
│   │   └── evaluate_whisper.py  # Evaluation
│   ├── data/                   # Training data
│   │   ├── raw/                # Raw audio files
│   │   ├── processed/          # Processed datasets
│   │   └── processed_finetune/ # Fine-tuning datasets
│   └── vectorstore/            # RAG vector store
│
├── ai-llm-ss/                  # CTC-based ASR System
│   ├── src/
│   │   └── asr/                # ASR components
│   │       ├── api.py          # FastAPI server (port 8001)
│   │       ├── model.py        # CRNN-CTC model
│   │       ├── features.py     # Feature extraction
│   │       └── decode.py       # CTC decoding
│   ├── scripts/
│   │   ├── serve_asr.py        # Start API server
│   │   └── train.py           # Training script
│   ├── data/
│   │   ├── results/            # Trained models
│   │   │   ├── asr_ctc.pt     # Main model
│   │   │   └── checkpoints/    # Training checkpoints
│   │   └── processed/          # Processed data
│   └── docs/
│       └── api_usage.md        # API documentation
│
├── audio_processing/            # Audio Preprocessing Utilities
│   ├── auto_preprocess.py      # Auto preprocessing
│   ├── noise_reduction.py      # Noise reduction
│   ├── audio_enhancer.py       # Audio enhancement
│   └── audio_filters.py        # Audio filtering
│
├── frontend/                    # Web Frontend
│   ├── index.html              # Main HTML
│   ├── app.js                  # Frontend logic
│   └── styles.css              # Styling
│
├── server.py                   # Legacy server (port 8000)
├── start_all_services.py       # Start all APIs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Key Components

### 1. AI2Text - Transformer ASR System

**Location**: `AI2Text/`

**Features**:
- Custom Transformer architecture
- Support for CTC and Attention mechanisms
- Timestamp prediction
- Beam search and LM decoding
- BPE/SentencePiece tokenization

**API**: Port 8000 (via `server.py` or `AI2Text/api/app.py`)

**Models**:
- `checkpoints/best_model.pt` - Best trained model
- `checkpoints/checkpoint_epoch_*.pt` - Training checkpoints

### 2. ai-llm - Whisper + LLM Pipeline

**Location**: `ai-llm/`

**Features**:
- Fine-tuned Whisper models (CTranslate2 format)
- Qwen LLM integration
- RAG pipeline for question answering
- Audio-to-answer workflow

**APIs**:
- Port 8002: Fine-tuned Whisper (`whisper-vi-en-ct2`)
- Port 8003: Original Whisper (`whisper-small`)

**Models**:
- `models/base/whisper-small/` - Original Whisper (HuggingFace)
- `models/final/whisper-vi-en-ct2/` - Fine-tuned Whisper (CTranslate2)
- `models/finetuned/qwen-mixed-merged/` - Fine-tuned Qwen LLM

### 3. ai-llm-ss - CTC-based ASR

**Location**: `ai-llm-ss/`

**Features**:
- CRNN-CTC architecture
- Trained from scratch
- Optimized for Ryzen 9 9900X + RTX 5060 Ti

**API**: Port 8001

**Models**:
- `data/results/asr_ctc.pt` - Main model
- `data/results/checkpoints/checkpoint_epoch_*.pt` - Training checkpoints

### 4. Audio Processing

**Location**: `audio_processing/`

**Features**:
- Automatic preprocessing
- Noise reduction
- Audio enhancement
- Format conversion (16kHz mono WAV)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start API Servers

#### Option A: Start All Services
```bash
python start_all_services.py
```

#### Option B: Start Individual Services

**AI2Text API (Transformer)**:
```bash
cd AI2Text
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Whisper Fine-tuned API**:
```bash
cd ai-llm
source .venv/bin/activate
ASR_MODEL=/home/alida/Documents/Cursor/AI2Text/ai-llm/models/final/whisper-vi-en-ct2 \
ASR_DEVICE=auto ASR_COMPUTE=float16 \
uvicorn src.api.server:app --host 0.0.0.0 --port 8002
```

**Whisper Original API**:
```bash
cd ai-llm
source .venv/bin/activate
ASR_MODEL=/home/alida/Documents/Cursor/AI2Text/ai-llm/models/base/whisper-small \
ASR_DEVICE=auto ASR_COMPUTE=int8 \
uvicorn src.api.server:app --host 0.0.0.0 --port 8003
```

**CTC ASR API**:
```bash
cd ai-llm-ss
python scripts/serve_asr.py --host 0.0.0.0 --port 8001
```

### 3. Start Frontend

```bash
cd frontend
python3 -m http.server 5500
```

Access at: `http://localhost:5500`

## 🚢 Deployment

Để triển khai hệ thống trên một máy tính mới, xem hướng dẫn chi tiết trong [**DEPLOYMENT.md**](DEPLOYMENT.md).

Hướng dẫn bao gồm:
- ✅ Yêu cầu hệ thống
- ✅ Cài đặt dependencies
- ✅ Cấu hình môi trường
- ✅ Download/copy models
- ✅ Chạy các services
- ✅ Troubleshooting

**Quick deployment checklist:**
1. Clone/copy project
2. Cài đặt Python 3.9+ và virtual environment
3. Cài đặt dependencies: `pip install -r requirements.txt`
4. Cài đặt FFmpeg và system dependencies
5. Copy models từ máy cũ (hoặc download)
6. Chạy: `python start_all_services.py`

Xem chi tiết tại: [DEPLOYMENT.md](DEPLOYMENT.md)

## 🌐 API Endpoints

### AI2Text API (Port 8000)

- `POST /transcribe` - Transcribe audio file
- `GET /models` - List available models
- `GET /health` - Health check

### Whisper APIs (Ports 8002, 8003)

- `POST /transcribe/upload` - Upload and transcribe audio
- `POST /transcribe` - Transcribe from server path
- `GET /health` - Health check

### CTC ASR API (Port 8001)

- `POST /transcribe` - Transcribe audio file
- `GET /model/info` - Model information
- `GET /health` - Health check

## 📊 Model Comparison

| Model | Architecture | Format | Port | Best For |
|-------|-------------|--------|------|----------|
| **AI2Text** | Transformer | PyTorch | 8000 | Custom training, Vietnamese |
| **Whisper Fine-tuned** | Whisper (CT2) | CTranslate2 | 8002 | Fast inference, EN+VI |
| **Whisper Original** | Whisper (HF) | HuggingFace | 8003 | General purpose |
| **CTC ASR** | CRNN-CTC | PyTorch | 8001 | Lightweight, CPU-friendly |

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:

- **PyTorch** - Deep learning framework
- **FastAPI** - API framework
- **Transformers** - HuggingFace models
- **faster-whisper** - Fast Whisper inference
- **librosa** - Audio processing
- **torchaudio** - Audio I/O

## 📚 Documentation

- **AI2Text**: See `AI2Text/README.md`
- **ai-llm**: See `ai-llm/Readme.md`
- **ai-llm-ss**: See `ai-llm-ss/README.md` and `ai-llm-ss/docs/api_usage.md`
- **Models & Technology**: See [**MODELS_TECHNOLOGY.md**](MODELS_TECHNOLOGY.md) - Chi tiết về công nghệ và triển khai models
- **Deployment**: See [**DEPLOYMENT.md**](DEPLOYMENT.md) - Hướng dẫn triển khai trên máy mới

## 🔍 Usage Examples

### Python Client

```python
import requests

# Transcribe with AI2Text API
response = requests.post(
    "http://localhost:8000/transcribe",
    files={"audio": open("audio.wav", "rb")},
    params={"model_name": "best_model"}
)
print(response.json())

# Transcribe with Whisper API
response = requests.post(
    "http://localhost:8002/transcribe/upload",
    files={"file": open("audio.wav", "rb")},
    data={"language": "vi", "model_size": "small"}
)
print(response.json())
```

### cURL

```bash
# AI2Text API
curl -X POST "http://localhost:8000/transcribe" \
     -F "audio=@audio.wav" \
     -F "model_name=best_model"

# Whisper API
curl -X POST "http://localhost:8002/transcribe/upload" \
     -F "file=@audio.wav" \
     -F "language=vi"
```

## 🛠️ Development

### Project Structure Guidelines

- **`AI2Text/`**: Main Transformer-based ASR system
- **`ai-llm/`**: Whisper + LLM integration
- **`ai-llm-ss/`**: CTC-based ASR from scratch
- **`audio_processing/`**: Shared audio utilities
- **`frontend/`**: Web interface

### Adding New Models

1. Place model files in appropriate `checkpoints/` or `models/` directory
2. Update API configuration to detect new models
3. Test with `/models` endpoint to verify detection

## 📝 License

This project is released under the **MIT License**.

Copyright (c) 2025 congkx123789

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 👥 Contributors

- **congkx123789**

---

**Last Updated**: 2024-12-09

