# AI2Text: Bilingual ASR System

A state-of-the-art **Automatic Speech Recognition (ASR)** system for **Vietnamese and English** using Transformer-based Sequence-to-Sequence architecture. Optimized for high-performance training on modern hardware.

## 🎯 Overview

This project implements a bilingual ASR model that can transcribe speech in both Vietnamese and English. The system uses a modern Transformer encoder-decoder architecture with advanced optimizations for efficient training and inference.

### Key Features

- 🌐 **Bilingual Support**: Vietnamese + English with language-aware embeddings
- 🚀 **High Performance**: Optimized for RTX 5060TI 16GB VRAM, Ryzen 9 9990X, 64GB RAM
- 🧠 **Modern Architecture**: LLaMA-style components (RMSNorm, SiLU, RoPE)
- ⚡ **Efficient Training**: Gradient checkpointing, mixed precision (AMP), Flash Attention
- 📊 **Real-time Monitoring**: Live metrics display (Loss, LR, WER, CER)
- 🎓 **Advanced Training**: Curriculum learning, auto-rollback, learning rate scheduling
- 🔧 **Production Ready**: REST API, evaluation tools, comprehensive logging

## 📐 Model Architecture

### Overall Structure

The model follows a **Sequence-to-Sequence (Seq2Seq)** Transformer architecture:

```
Audio Input → Encoder → Encoded Features → Decoder → Text Output
```

### Model Specifications

- **Total Parameters**: ~30.3M
- **Model Dimension (d_model)**: 256
- **Encoder Layers**: 14
- **Decoder Layers**: 6
- **Attention Heads**: 8
- **Feed-Forward Dimension (d_ff)**: 2048
- **Vocabulary Size**: 3,500 (SentencePiece BPE)
- **Input Features**: 80-dimensional Mel-spectrograms
- **Subsampling Factor**: 2x (via convolutional subsampling)

### Encoder Architecture (`ASREncoder`)

The encoder processes audio features and produces encoded representations:

1. **Convolutional Subsampling**
   - 2D Convolution with stride=2 (2x subsampling)
   - Reduces sequence length while preserving information
   - Output: `(batch, time/2, freq/2, channels)`

2. **Linear Projection**
   - Projects subsampled features to model dimension
   - Layer normalization for stability

3. **Language Embedding**
   - Learnable embeddings for language identification
   - Supports bilingual training (Vietnamese=0, English=1)
   - Added to all time steps: `x = x + lang_emb.unsqueeze(1)`

4. **Rotary Positional Embedding (RoPE)**
   - Modern positional encoding used in LLaMA
   - Applied per attention head
   - Better generalization to longer sequences

5. **Encoder Layers** (14 layers)
   Each layer consists of:
   - **Self-Attention** (Multi-Head Attention)
     - Uses Flash Attention (SDPA) for efficiency
     - Bidirectional (non-causal) for encoder
     - RoPE applied to queries and keys
   - **Feed-Forward Network**
     - SiLU (Swish) activation instead of ReLU
     - Pre-norm architecture (RMSNorm before each sub-layer)
   - **Residual Connections**
   - **Gradient Checkpointing** (middle layers only)

6. **Final Normalization**
   - RMSNorm for final output
   - Dropout for regularization

**Output**: `(batch, time/2, d_model)` encoded features

### Decoder Architecture (`ASRDecoder`)

The decoder generates text tokens autoregressively:

1. **Token Embedding**
   - Embedding layer: `vocab_size → d_model`
   - Scaled by `√d_model` for numerical stability

2. **Rotary Positional Embedding (RoPE)**
   - Same RoPE as encoder
   - Applied to decoder self-attention

3. **Decoder Layers** (6 layers)
   Each layer consists of:
   - **Self-Attention** (Causal)
     - Causal mask prevents attending to future tokens
     - Enables autoregressive generation
     - RoPE applied
   - **Cross-Attention** (Encoder-Decoder Attention)
     - Queries from decoder, Keys/Values from encoder
     - Allows decoder to attend to all encoder positions
     - No RoPE (attends to encoder positions)
   - **Feed-Forward Network**
     - Same structure as encoder (SiLU, pre-norm)
   - **Residual Connections**
   - **Gradient Checkpointing** (middle layers only)

4. **Output Projection**
   - Linear layer: `d_model → vocab_size`
   - Produces logits for vocabulary

**Output**: `(batch, tgt_len, vocab_size)` logits

### Modern Components

#### RMSNorm (Root Mean Square Layer Normalization)
- More efficient than LayerNorm
- Used in LLaMA and modern LLMs
- Applied before attention and FFN

#### SiLU (Swish) Activation
- `SiLU(x) = x * sigmoid(x)`
- Prevents "dead neurons" (unlike ReLU)
- Smoother gradient flow

#### Rotary Positional Embedding (RoPE)
- Relative positional encoding
- Applied directly to attention queries/keys
- Better generalization to longer sequences

#### Flash Attention (SDPA)
- PyTorch's `scaled_dot_product_attention`
- Memory-efficient attention computation
- Automatic kernel selection (FlashAttention-2 or MemoryEfficient)

### Training Process

1. **Teacher Forcing** (Training)
   - Target tokens shifted right: `[SOS, token1, token2, ..., EOS]`
   - Decoder receives: `[SOS, token1, token2, ...]` (without EOS)
   - Predicts: `[token1, token2, ..., EOS]`
   - Enables parallel training

2. **Autoregressive Generation** (Inference)
   - Starts with SOS token
   - Generates one token at a time
   - Stops at EOS or max length
   - Supports temperature sampling, repetition penalty

### Loss Function

- **Cross-Entropy Loss** with:
  - Ignore padding tokens (index 0)
  - Mean reduction
  - Applied to flattened logits and targets

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- CUDA-capable GPU (recommended: RTX 5060TI 16GB or better)
- 64GB RAM (recommended)
- SSD with 3000MB/s+ read speed

### Setup

```bash
# Clone repository
git clone <repository-url>
cd AI2Text

# Install dependencies
pip install -r requirements.txt

# Download or prepare tokenizer
# Place tokenizer model at: models/tokenizer_vi_en_3500.model
```

## 📊 Dataset

The model is trained on a merged bilingual dataset:
- **Training**: ~194,167 samples (77% Vietnamese, 23% English)
- **Validation**: ~30,123 samples
- **Test**: Available for evaluation

Dataset structure:
```
data/processed/full_merged_dataset/
├── train/
│   ├── audio/
│   └── manifest.csv
├── val/
│   ├── audio/
│   └── manifest.csv
└── test/
    ├── audio/
    └── manifest.csv
```

## 🏋️ Training

### Basic Training

```bash
python training/train.py --config configs/default.yaml
```

### Resume Training

```bash
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt
```

### Training Configuration

Key hyperparameters (from `configs/default.yaml`):

- **Batch Size**: 64 (effective: 256 with gradient accumulation)
- **Learning Rate**: 0.0003
- **Epochs**: 50
- **Warmup**: 3% of training steps
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: Enabled (AMP)
- **Gradient Clipping**: 0.5

### Real-time Monitoring

The training script displays real-time metrics every batch:
- **Loss**: Current and average training loss
- **LR**: Current learning rate
- **WER**: Word Error Rate (during validation)
- **CER**: Character Error Rate (during validation)
- **GPU Memory**: VRAM usage
- **Speed**: Samples per second
- **ETA**: Estimated time remaining

Example output:
```
Batch    100/3033 (  3.3%) | Loss: 8.331588 | Avg: 8.332509 | LR: 1.65e-06 | BestVal: N/A | BestWER: N/A | BestCER: N/A | GPU: 9.5GB | Speed: 83.7 samp/s | ETA: 1:05:05
```

### Training Features

#### Curriculum Learning
- Starts with shorter sentences
- Gradually increases difficulty
- Configurable via `curriculum_learning` in config

#### Auto-Rollback
- Automatically reverts to previous checkpoint if loss spikes
- Prevents training collapse
- Configurable threshold and patience

#### Learning Rate Scheduling
- Warmup phase (3% of training)
- Cosine annealing decay
- Automatic LR restoration on resume

## 🔍 Evaluation

### Calculate WER/CER

```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml
```

### Metrics

- **WER (Word Error Rate)**: Percentage of word errors
- **CER (Character Error Rate)**: Percentage of character errors
- **Validation Loss**: Cross-entropy loss on validation set

## 🌐 API Usage

### Start API Server

```bash
bash start_api.sh
# or
python api/app.py
```

### API Endpoints

#### Transcribe Audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@audio_file.wav" \
  -F "language=vi"  # or "en"
```

#### Health Check

```bash
curl http://localhost:8000/health
```

### Python Client

```python
from api.asr_service import ASRService

service = ASRService("checkpoints/best_model.pt")
transcript = service.transcribe("audio.wav", language="vi")
print(transcript)
```

## 📁 Project Structure

```
AI2Text/
├── api/                    # REST API service
│   ├── app.py             # FastAPI application
│   ├── asr_service.py     # ASR inference service
│   └── example_client.py  # Example client code
├── configs/                # Configuration files
│   └── default.yaml       # Main training config
├── data/                   # Dataset storage
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── models/                 # Model definitions
│   ├── asr_base.py       # Main ASR model (Encoder-Decoder)
│   ├── modern_components.py  # RMSNorm, RoPE, etc.
│   └── lora.py           # LoRA implementation
├── preprocessing/         # Data preprocessing
│   ├── audio_processing.py    # Audio feature extraction
│   ├── text_cleaning.py       # Text normalization
│   └── sentencepiece_tokenizer.py  # Tokenizer
├── training/              # Training code
│   ├── train.py          # Main training script
│   ├── dataset.py        # Data loading
│   ├── callbacks.py      # Training callbacks
│   └── smart_callbacks.py # Advanced callbacks
├── decoding/              # Decoding strategies
│   ├── beam_search.py    # Beam search decoder
│   └── lm_decoder.py     # Language model decoder
├── utils/                 # Utilities
│   ├── logger.py         # Logging utilities
│   ├── metrics.py        # WER/CER calculation
│   └── manifest_loader.py # Dataset loading
└── scripts/               # Helper scripts
    ├── train_tokenizer.py # Train SentencePiece tokenizer
    └── monitor_training.py # Training monitoring
```

## ⚙️ Optimization Features

### Hardware Optimizations

#### For RTX 5060TI 16GB VRAM:
- **Gradient Checkpointing**: Saves ~40% VRAM
- **Mixed Precision (AMP)**: 2x speedup, 50% memory reduction
- **Flash Attention**: Efficient attention computation
- **Efficient Batching**: Dynamic batching with length sorting

#### For Ryzen 9 9990X:
- **Multi-core Data Loading**: 12 workers
- **Persistent Workers**: Reduces worker startup overhead
- **Prefetch Factor**: 4 batches per worker
- **Optimized CPU Operations**: Efficient tensor operations

#### For 64GB RAM:
- **Large Batch Sizes**: Effective batch size 256
- **Better Caching**: Dataset caching options
- **Memory-efficient Data Loading**: Optimized data pipeline

#### For SSD 3000MB/s:
- **Fast I/O**: Parallel data loading
- **Efficient Prefetching**: Reduces I/O wait time

### Training Optimizations

- **Gradient Accumulation**: Simulates larger batch sizes
- **Learning Rate Warmup**: Stable training start
- **Cosine Annealing**: Smooth learning rate decay
- **Gradient Clipping**: Prevents gradient explosion
- **Auto-Rollback**: Prevents training collapse
- **Curriculum Learning**: Progressive difficulty increase

## 📈 Performance

### Model Performance

- **Parameters**: 30,325,164 total (all trainable)
- **Training Speed**: ~80-85 samples/second
- **GPU Memory**: ~9-10GB VRAM usage
- **Batch Processing**: 64 samples/batch (effective 256)

### Training Metrics

Monitor training progress:
- Real-time loss, LR, WER, CER
- Best validation metrics tracking
- GPU utilization and memory usage
- Estimated time to completion

## 🔧 Configuration

Edit `configs/default.yaml` to customize:

- Model architecture (layers, dimensions)
- Training hyperparameters (LR, batch size, epochs)
- Data settings (dataset path, tokenizer)
- Optimization settings (AMP, checkpointing)
- Advanced features (curriculum learning, auto-rollback)

## 📝 Logging

Training logs are saved to:
- **Console**: Real-time metrics every batch
- **File**: `logs/training.log` (detailed logs)
- **Checkpoints**: `checkpoints/` (model snapshots)

## 🧪 Testing

Run tests:

```bash
python tests/run_tests_simple.py
# or
pytest tests/
```

## 📚 References

- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

## 📄 License

[Add your license here]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

- SentencePiece for tokenization
- PyTorch team for excellent framework
- Hugging Face for inspiration

---

**Built with ❤️ for bilingual ASR**

