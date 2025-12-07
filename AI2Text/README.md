# AI2Text - Vietnamese & English ASR System

A high-performance Automatic Speech Recognition (ASR) system for Vietnamese and English, built with PyTorch. Features transformer-based architecture with timestamp prediction, BPE tokenization, and BF16 mixed precision training.

**Version**: 2.0.0  
**Last Updated**: 2024

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Requirements](#-requirements)
3. [Installation](#-installation)
4. [Quick Start & Detailed Guide](#-quick-start--hướng-dẫn-chạy-chi-tiết)
5. [API Documentation](#-api-documentation)
6. [Training Guide](#️-training-guide)
7. [Project Structure](#-project-structure)
8. [Model Specifications](#-model-specifications)
9. [Technical Specifications](#-technical-specifications)
10. [Configuration](#️-configuration)
11. [Troubleshooting](#-troubleshooting)
12. [Performance](#-performance)

---

## 🚀 Features

- **Bilingual Support**: Vietnamese and English speech recognition
- **Timestamp Prediction**: Word-level timestamp prediction for streaming applications
- **BPE Tokenization**: Subword tokenization for better OOV handling
- **BF16 Mixed Precision**: Optimized training with BF16 for RTX 5060 Ti 16GB
- **REST API**: Easy-to-use REST API for inference
- **Auto-Rollback**: Automatic recovery from training failures
- **Curriculum Learning**: Progressive training strategy
- **Modern Architecture**: RMSNorm, RoPE, SiLU activations (LLaMA-style)

---

## 📋 Requirements

### Hardware

**Minimum Requirements**:
- **GPU**: NVIDIA GPU with 8GB VRAM (CUDA 11.8+)
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 16GB
- **Storage**: 50GB free space (SSD recommended)

**Recommended Requirements**:
- **GPU**: RTX 5060 Ti 16GB (or equivalent)
- **CPU**: Ryzen 9 9900X (16+ cores, 4.5GHz+)
- **RAM**: 64GB
- **Storage**: 200GB+ free space, SSD 3000MB/s+

**Optimal Configuration** (for this project):
- **GPU**: RTX 5060 Ti 16GB VRAM
- **CPU**: Ryzen 9 9900X (16 cores)
- **RAM**: 64GB DDR5
- **Storage**: NVMe SSD 3000MB/s+

### Software

- **OS**: Linux (Ubuntu 20.04+), Windows 10/11 (WSL2), macOS (CPU only)
- **Python**: 3.8, 3.9, 3.10, or 3.11 (recommended: 3.10)
- **CUDA**: 11.8 or 12.1+ (for GPU support)
- **cuDNN**: 8.0+

### Dependencies

```
torch>=2.0.0
torchaudio>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
librosa>=0.10.0
soundfile>=0.12.0
pandas>=1.5.0
numpy>=1.23.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 🔧 Installation

### Step 1: Clone Repository

```bash
cd /home/alida/Documents/Cursor/AI2Text/AI2Text
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install fastapi uvicorn python-multipart
pip install librosa soundfile
pip install pyyaml tqdm pandas numpy
pip install transformers
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🎯 Quick Start & Hướng Dẫn Chạy Chi Tiết

### Mục Lục Hướng Dẫn

1. [Setup Môi Trường](#1-setup-môi-trường)
2. [Chuẩn Bị Dữ Liệu](#2-chuẩn-bị-dữ-liệu)
3. [Build BPE Vocabulary](#3-build-bpe-vocabulary)
4. [Training Model](#4-training-model)
5. [Chạy Inference](#5-chạy-inference)
6. [Chạy API Server](#6-chạy-api-server)
7. [Test & Validation](#7-test--validation)

---

### 1. Setup Môi Trường (Environment Setup)

#### 1.1 Kiểm Tra Hệ Thống

```bash
# Kiểm tra Python version (cần 3.8+)
python --version

# Kiểm tra CUDA (nếu có GPU)
nvidia-smi

# Kiểm tra disk space (cần ít nhất 50GB)
df -h
```

#### 1.2 Tạo Virtual Environment

```bash
# Di chuyển đến thư mục project
cd /home/alida/Documents/Cursor/AI2Text/AI2Text

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

#### 1.3 Cài Đặt Dependencies

```bash
# Cài đặt PyTorch với CUDA 11.8
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các dependencies khác
pip install fastapi uvicorn python-multipart
pip install librosa soundfile
pip install pyyaml tqdm pandas numpy
pip install transformers
```

#### 1.4 Verify Installation

```bash
# Kiểm tra PyTorch và CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Kiểm tra các thư viện khác
python -c "import fastapi, librosa, yaml; print('All dependencies installed successfully!')"
```

---

### 2. Chuẩn Bị Dữ Liệu (Data Preparation)

#### 2.1 Cấu Trúc Thư Mục Dữ Liệu

```bash
# Tạo cấu trúc thư mục
mkdir -p data/processed/full_merged_dataset/{train,val,test}/audio

# Cấu trúc mong muốn:
# data/processed/full_merged_dataset/
# ├── train/
# │   ├── audio/
# │   │   ├── 001.wav
# │   │   ├── 002.wav
# │   │   └── ...
# │   └── manifest.csv
# ├── val/
# │   ├── audio/
# │   └── manifest.csv
# └── test/
#     ├── audio/
#     └── manifest.csv
```

#### 2.2 Tạo Manifest CSV

**Format của manifest.csv**:
```csv
audio_path,text,duration_seconds,language_id
train/audio/001.wav,"xin chào việt nam",2.5,0
train/audio/002.wav,"hello world",1.8,1
train/audio/003.wav,"tôi là người việt nam",3.2,0
```

**Tạo manifest từ thư mục audio**:
```python
import os
import pandas as pd
import librosa
from pathlib import Path

def create_manifest(audio_dir, output_file):
    """Tạo manifest CSV từ thư mục audio."""
    audio_dir = Path(audio_dir)
    data = []
    
    for audio_file in audio_dir.glob("*.wav"):
        # Load audio để lấy duration
        duration = librosa.get_duration(filename=str(audio_file))
        
        # Đọc transcript từ file .txt cùng tên (nếu có)
        transcript_file = audio_file.with_suffix('.txt')
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""  # Cần điền sau
        
        # Xác định language_id (0=Vietnamese, 1=English)
        language_id = 0  # Mặc định Vietnamese
        
        data.append({
            'audio_path': str(audio_file.relative_to(audio_dir.parent)),
            'text': text,
            'duration_seconds': duration,
            'language_id': language_id
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created manifest with {len(df)} samples: {output_file}")

# Sử dụng
create_manifest("data/processed/full_merged_dataset/train/audio", 
                "data/processed/full_merged_dataset/train/manifest.csv")
```

---

### 3. Build BPE Vocabulary

```bash
# Tạo corpus từ manifests
python -c "
import pandas as pd
train_df = pd.read_csv('data/processed/full_merged_dataset/train/manifest.csv')
val_df = pd.read_csv('data/processed/full_merged_dataset/val/manifest.csv')
all_texts = pd.concat([train_df['text'], val_df['text']])
with open('corpus.txt', 'w', encoding='utf-8') as f:
    for text in all_texts:
        f.write(text + '\n')
print(f'Created corpus.txt with {len(all_texts)} lines')
"

# Build BPE vocabulary
python build_bpe_vocab.py \
    --corpus corpus.txt \
    --output models/bilingual_bpe_2k.json \
    --vocab_size 2000
```

---

### 4. Training Model

#### 4.1 Training Lần Đầu

```bash
# Training với config mặc định
python training/train.py --config configs/small_25m_bpe2k.yaml

# Training với các options
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --use_timestamps  # Bật timestamp training

# Training chỉ với Vietnamese
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --language vi
```

#### 4.2 Resume Training

```bash
# Resume từ checkpoint
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --resume checkpoints/checkpoint_epoch_45.pt
```

#### 4.3 Monitor Training

```bash
# Xem log real-time
tail -f logs/training_small_25m_bpe2k.log

# Xem metrics
tail -f logs/training_small_25m_bpe2k.log | grep -E "Loss|WER|CER"
```

---

### 5. Chạy Inference

#### 5.1 Inference Qua Python

```python
from api.asr_service import ASRService

# Load model
service = ASRService(
    checkpoint_path="checkpoints/checkpoint_epoch_45.pt",
    device="cuda"
)

# Transcribe
result = service.transcribe(
    audio_path="test_audio.wav",
    return_timestamps=True
)

print(f"Text: {result['text']}")
print(f"Timestamps: {result['timestamps']}")
```

#### 5.2 Inference với Python Client

```bash
python api/example_client.py \
    --url http://localhost:8000 \
    --checkpoint checkpoints/checkpoint_epoch_45.pt \
    --audio test_audio.wav \
    --model-name my_model \
    --timestamps
```

---

### 6. Chạy API Server

#### 6.1 Phương pháp 1: Script

```bash
chmod +x start_api.sh
./start_api.sh
```

#### 6.2 Phương pháp 2: Python

```bash
python -m api.app_v2
```

#### 6.3 Phương pháp 3: Uvicorn

```bash
uvicorn api.app_v2:app --host 0.0.0.0 --port 8000 --reload
```

#### 6.4 Test API

```bash
# Health check
curl http://localhost:8000/health

# Load model
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_path": "checkpoints/checkpoint_epoch_45.pt", "model_name": "my_model", "device": "cuda"}'

# Transcribe
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@test_audio.wav" \
  -F "model_name=my_model" \
  -F "return_timestamps=true"
```

---

### 7. Test & Validation

```bash
# Run validation
python run_validation.py \
    --checkpoint checkpoints/checkpoint_epoch_45.pt \
    --test_manifest data/processed/full_merged_dataset/test/manifest.csv

# Calculate WER/CER
python check_wer.py \
    --reference reference.txt \
    --hypothesis hypothesis.txt

# Run tests
python -m pytest tests/
```

---

## 📚 API Documentation

### Endpoints

#### `POST /load_model`
Load a model checkpoint into memory.

**Request**:
```json
{
  "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
  "model_name": "my_model",
  "device": "cuda"
}
```

**Response**:
```json
{
  "status": "loaded",
  "model_name": "my_model",
  "checkpoint_path": "/absolute/path/to/checkpoint.pt",
  "device": "cuda"
}
```

#### `POST /transcribe`
Transcribe a single audio file.

**Parameters**:
- `audio` (file, required): Audio file to transcribe
- `model_name` (query, default: "default"): Name of loaded model
- `return_timestamps` (query, default: false): Return word-level timestamps
- `language_id` (query, optional): Language ID (0=Vietnamese, 1=English)

**Response**:
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

#### `POST /transcribe_batch`
Transcribe multiple audio files.

#### `GET /models`
List available model checkpoints.

#### `GET /health`
Health check endpoint.

#### `DELETE /models/{model_name}`
Unload a model from memory.

---

## 🏋️ Training Guide

### Training Configuration

Edit `configs/small_25m_bpe2k.yaml`:

```yaml
# Model architecture
d_model: 320
num_encoder_layers: 16
num_heads: 4
d_ff: 1280

# Training hyperparameters
batch_size: 32
num_epochs: 120
learning_rate: 0.0005
weight_decay: 0.01

# Mixed precision
use_amp: true  # BF16 mixed precision

# Data
dataset_root: "data/processed/full_merged_dataset"
```

### Training Options

```bash
# Basic training
python training/train.py --config configs/small_25m_bpe2k.yaml

# Resume from checkpoint
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --resume checkpoints/checkpoint_epoch_45.pt

# Language-specific training
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --language vi  # or "en"

# Disable timestamps
python training/train.py \
    --config configs/small_25m_bpe2k.yaml \
    --no_timestamps
```

---

## 📁 Project Structure

```
AI2Text/
├── api/                    # REST API Layer
│   ├── app_v2.py          # Main FastAPI server
│   ├── asr_service.py     # ASR service class
│   └── example_client.py  # Python client example
│
├── models/                # Model Architecture
│   ├── asr_base.py       # Base ASR model
│   ├── asr_with_timestamps.py  # Model with timestamps
│   ├── modern_components.py    # RMSNorm, RoPE
│   └── bilingual_bpe_2k.json   # BPE vocabulary
│
├── preprocessing/         # Data Preprocessing
│   ├── audio_processing.py
│   ├── bpe_tokenizer.py
│   └── text_cleaning.py
│
├── training/              # Training Layer
│   ├── train.py          # Main training script
│   ├── dataset.py        # Dataset & DataLoader
│   ├── callbacks.py      # Training callbacks
│   └── smart_callbacks.py  # Auto-rollback, curriculum
│
├── configs/              # Configuration Files
│   ├── small_25m_bpe2k.yaml
│   └── default.yaml
│
├── checkpoints/          # Model Checkpoints
├── data/                 # Data Directory
├── logs/                 # Training Logs
└── tests/                # Test Suite
```

---

## 🤖 Model Specifications

### Architecture

**Small 25M Model**:
- **Encoder**: 16 layers, d_model=320, num_heads=4
- **Decoder**: Linear projection (CTC)
- **Timestamp Head**: Optional word-level timestamps
- **Parameters**: ~25M

### Input/Output

**Input**:
- Audio: 16kHz, mono, WAV
- Features: Mel spectrogram (80 bins)
- Shape: `(batch, time_frames, 80)`

**Output**:
- Text: Transcribed text (string)
- Timestamps: Word-level timestamps (optional)
- Shape: `(batch, time_frames, vocab_size)`

### Tokenizer

- **BPE**: 2000 or 18000 tokens
- **Character**: Character-level (optional)
- **Languages**: Vietnamese + English

---

## 📋 Technical Specifications

### API Specifications

#### Request/Response Formats

**Load Model Request**:
```json
{
  "checkpoint_path": "string (required)",
  "model_name": "string (optional, default: 'default')",
  "device": "string (optional, 'cuda' | 'cpu', default: 'cuda')"
}
```

**Transcribe Response**:
```json
{
  "text": "string",
  "timestamps": [{"word": "string", "start": float, "end": float}] | null,
  "confidence": float | null,
  "model_name": "string",
  "processing_time": float
}
```

### Performance Specifications

**Training** (RTX 5060 Ti 16GB):
- Batch Size 32: ~2-3 seconds/batch
- Throughput: ~10-15 batches/second
- Epoch Time: ~30-60 minutes (depends on dataset)

**Inference**:
- Short Audio (< 5s): 50-150ms
- Medium Audio (5-30s): 150-500ms
- Long Audio (30-60s): 500ms-2s
- Throughput: ~10-20x real-time

### Data Format Specifications

**Manifest CSV**:
```csv
audio_path,text,duration_seconds,language_id
train/audio/001.wav,"text here",2.5,0
```

**Checkpoint Format**:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'config': dict
}
```

---

## ⚙️ Configuration

### Key Configuration Options

```yaml
# Model
d_model: 320
num_encoder_layers: 16
num_heads: 4
d_ff: 1280

# Training
batch_size: 32
num_epochs: 120
learning_rate: 0.0005
use_amp: true  # BF16 mixed precision

# Tokenization
tokenizer_type: "bpe"
bpe_vocab_path: "models/bilingual_bpe_2k.json"

# Timestamps
use_timestamps: true
timestamp_loss_weight: 0.01
```

---

## 🔍 Troubleshooting

### CUDA Out of Memory

```bash
# Giải pháp 1: Giảm batch size
batch_size: 16  # Trong config

# Giải pháp 2: Gradient accumulation
gradient_accumulation_steps: 4

# Giải pháp 3: Tắt cache_in_ram
cache_in_ram: false
```

### Model Not Found

```bash
# Kiểm tra checkpoint
ls -lh checkpoints/

# Load với absolute path
python training/train.py \
    --resume /absolute/path/to/checkpoint.pt
```

### Data Not Found

```bash
# Kiểm tra cấu trúc thư mục
tree data/processed/full_merged_dataset/

# Kiểm tra manifest
head data/processed/full_merged_dataset/train/manifest.csv
```

---

## 📊 Performance

### Hardware Optimization

Optimized for:
- **GPU**: RTX 5060 Ti 16GB VRAM
- **CPU**: Ryzen 9 9900X (16+ cores)
- **RAM**: 64GB
- **Storage**: SSD 3000MB/s+

### Recommended Settings

```yaml
batch_size: 32
num_workers: 16
use_amp: true
prefetch_factor: 4
use_bucketing: true
```

---

## 📝 Notes

- Models are loaded into memory and cached
- Supports both BPE and character-level tokenizers
- Timestamps are optional and require `return_timestamps=true`
- Audio formats: WAV, MP3, FLAC, M4A, OGG (auto-converted to 16kHz mono)
- BF16 mixed precision is used (not FP16) for better stability

---

## 🔗 Additional Resources

- **Training Logs**: `logs/training_small_25m_bpe2k.log`
- **Checkpoints**: `checkpoints/`
- **Model Configs**: `configs/`

---

## 📄 License

[Add your license here]

---

**Happy Transcribing! 🎤→📝**

