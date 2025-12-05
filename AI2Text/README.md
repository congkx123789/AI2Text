# AI2Text - Vietnamese & English ASR System

A high-performance Automatic Speech Recognition (ASR) system for Vietnamese and English, built with PyTorch. Features transformer-based architecture with timestamp prediction, BPE tokenization, and BF16 mixed precision training.

## 📑 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Training](#️-training)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)

## 🚀 Features

- **Bilingual Support**: Vietnamese and English speech recognition
- **Timestamp Prediction**: Word-level timestamp prediction for streaming applications
- **BPE Tokenization**: Subword tokenization for better OOV handling
- **BF16 Mixed Precision**: Optimized training with BF16 for RTX 5060 Ti 16GB
- **REST API**: Easy-to-use REST API for inference
- **Auto-Rollback**: Automatic recovery from training failures
- **Curriculum Learning**: Progressive training strategy

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 5060 Ti 16GB recommended)
- **CPU**: Multi-core CPU (Ryzen 9 9900X recommended)
- **RAM**: 64GB recommended for large datasets
- **Storage**: SSD with 3000MB/s+ read speed recommended

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+
- FastAPI
- Other dependencies (see installation)

## 🔧 Installation

### 1. Clone the repository

```bash
cd /home/alida/Documents/Cursor/AI2Text/AT2Text/AI2Text
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn python-multipart
pip install librosa soundfile
pip install pyyaml tqdm pandas numpy
pip install transformers  # For BPE tokenizer
```

### 4. Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Quick Start

### Running the API Server

#### Method 1: Using the startup script

```bash
./start_api.sh
```

#### Method 2: Using Python directly

```bash
python -m api.app_v2
```

#### Method 3: Using uvicorn

```bash
uvicorn api.app_v2:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Using the API

#### 1. Load a model

```bash
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
    "model_name": "my_model",
    "device": "cuda"
  }'
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

#### 2. Transcribe audio

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@path/to/audio.wav" \
  -F "model_name=my_model" \
  -F "return_timestamps=true"
```

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

## 📚 API Documentation

### Endpoints

#### `POST /load_model`
Load a model checkpoint into memory.

**Request Body:**
```json
{
  "checkpoint_path": "checkpoints/checkpoint_epoch_45.pt",
  "model_name": "my_model",
  "device": "cuda"
}
```

#### `POST /transcribe`
Transcribe a single audio file.

**Parameters:**
- `audio` (file, required): Audio file to transcribe
- `model_name` (query, default: "default"): Name of loaded model
- `return_timestamps` (query, default: false): Return word-level timestamps
- `language_id` (query, optional): Language ID (0=Vietnamese, 1=English)

**Supported formats:** WAV, MP3, FLAC, M4A, OGG, etc.

#### `POST /transcribe_batch`
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

#### `GET /models`
List available model checkpoints.

#### `GET /health`
Health check endpoint.

#### `DELETE /models/{model_name}`
Unload a model from memory.

### Python Client Example

```python
from api.example_client import ASRAPIClient

# Initialize client
client = ASRAPIClient("http://localhost:8000")

# Load model
client.load_model(
    checkpoint_path="checkpoints/checkpoint_epoch_45.pt",
    model_name="my_model",
    device="cuda"
)

# Transcribe single file
result = client.transcribe(
    audio_path="audio.wav",
    model_name="my_model",
    return_timestamps=True
)
print(f"Text: {result['text']}")
print(f"Timestamps: {result['timestamps']}")

# Transcribe batch
results = client.transcribe_batch(
    audio_paths=["audio1.wav", "audio2.wav"],
    model_name="my_model"
)
```

### JavaScript/TypeScript Example

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

## 🏋️ Training

### Prepare Data

1. Place your audio files in `data/processed/full_merged_dataset/train/audio/`
2. Create a manifest CSV file with columns: `audio_path`, `text`, `duration_seconds`
3. Update config file to point to your dataset

### Start Training

```bash
python training/train.py --config configs/small_25m_bpe2k.yaml
```

### Resume Training

```bash
python training/train.py \
  --config configs/small_25m_bpe2k.yaml \
  --resume checkpoints/checkpoint_epoch_45.pt
```

### Training Configuration

Edit `configs/small_25m_bpe2k.yaml` to customize:

```yaml
# Model architecture
d_model: 320
num_encoder_layers: 16
num_heads: 4

# Training hyperparameters
batch_size: 32
num_epochs: 120
learning_rate: 0.0005

# Mixed precision
use_amp: true  # BF16 mixed precision

# Data
dataset_root: "data/processed/full_merged_dataset"
```

### Monitor Training

Training logs are saved to `logs/training_small_25m_bpe2k.log`. Checkpoints are saved to `checkpoints/` every 5 epochs.

## 📁 Project Structure

```
AI2Text/
├── api/                    # REST API
│   ├── app_v2.py          # Main API server
│   ├── asr_service.py     # ASR service class
│   ├── example_client.py  # Python client example
│   └── README.md          # API documentation
├── checkpoints/           # Model checkpoints
├── configs/               # Configuration files
│   ├── small_25m_bpe2k.yaml
│   ├── medium_60m_bpe2k.yaml
│   └── default.yaml
├── data/                  # Data directory
│   ├── processed/         # Processed datasets
│   └── raw/               # Raw datasets
├── models/                # Model definitions
│   ├── asr_base.py
│   ├── asr_with_timestamps.py
│   └── bilingual_bpe_2k.json
├── preprocessing/         # Data preprocessing
│   ├── audio_processing.py
│   ├── bpe_tokenizer.py
│   └── text_cleaning.py
├── training/              # Training scripts
│   ├── train.py
│   ├── dataset.py
│   └── callbacks.py
├── utils/                 # Utility functions
├── start_api.sh           # API startup script
└── README.md              # This file
```

## ⚙️ Configuration

### Model Configurations

- **small_25m_bpe2k.yaml**: 25M parameter model, BPE 2k vocab
- **medium_60m_bpe2k.yaml**: 60M parameter model, BPE 2k vocab
- **default.yaml**: Default configuration

### Key Configuration Options

```yaml
# Model capacity
d_model: 320              # Hidden dimension
num_encoder_layers: 16    # Number of encoder layers
num_heads: 4              # Attention heads

# Training
batch_size: 32            # Batch size
learning_rate: 0.0005      # Learning rate
use_amp: true             # BF16 mixed precision

# Tokenization
tokenizer_type: "bpe"     # "bpe" or "char"
bpe_vocab_path: "models/bilingual_bpe_2k.json"

# Timestamps
use_timestamps: true
timestamp_loss_weight: 0.01
```

## 🔍 Troubleshooting

### API Issues

**Problem**: Model not found
```bash
# Check available models
curl http://localhost:8000/models

# Load model with correct path
curl -X POST "http://localhost:8000/load_model" \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_path": "checkpoints/checkpoint_epoch_45.pt", "model_name": "my_model"}'
```

**Problem**: CUDA out of memory
- Reduce batch size in config
- Use CPU: `"device": "cpu"` when loading model
- Unload unused models: `DELETE /models/{model_name}`

### Training Issues

**Problem**: OOM (Out of Memory)
- Reduce `batch_size` in config
- Reduce `num_workers`
- Set `cache_in_ram: false`
- Use gradient accumulation

**Problem**: Loss explosion
- Auto-rollback is enabled by default
- Check `auto_rollback` settings in config
- Reduce learning rate

**Problem**: Model not learning
- Check data quality
- Verify tokenizer matches training data
- Check WER/CER metrics in logs

## 📊 Performance

### Hardware Optimization

The system is optimized for:
- **GPU**: RTX 5060 Ti 16GB VRAM
- **CPU**: Ryzen 9 9900X (16+ cores)
- **RAM**: 64GB
- **Storage**: SSD 3000MB/s+

### Recommended Settings

```yaml
batch_size: 32              # Adjust based on VRAM
num_workers: 16             # Match CPU cores
use_amp: true               # BF16 for speed
prefetch_factor: 4         # For fast SSD
```

## 📝 Notes

- Models are loaded into memory and cached. Use `/load_model` before transcribing.
- Supports both BPE and character-level tokenizers based on model config.
- Timestamps are optional and require `return_timestamps=true`.
- Audio formats supported: WAV, MP3, FLAC, M4A, OGG, etc. (via torchaudio/librosa).
- BF16 mixed precision is used for training (not FP16) for better numerical stability.

## 🔗 Additional Resources

- **API Documentation**: See `api/README.md`
- **Training Logs**: `logs/training_small_25m_bpe2k.log`
- **Checkpoints**: `checkpoints/`

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

---

**Happy Transcribing! 🎤→📝**

