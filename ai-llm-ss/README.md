# AI-LLM-SS: Automatic Speech Recognition với CTC

Hệ thống ASR (Automatic Speech Recognition) từ đầu sử dụng kiến trúc CRNN-CTC, được tối ưu cho training trên Ryzen 9 9900X và RTX 5060 Ti 16GB.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc Model](#kiến-trúc-model)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Dataset](#dataset)
- [Training](#training)
- [Sử dụng Model](#sử-dụng-model)
- [API Server](#api-server)
- [Cấu hình tối ưu](#cấu-hình-tối-ưu)
- [Cấu trúc Project](#cấu-trúc-project)
- [Scripts](#scripts)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)
- [Examples](#examples)

## 🎯 Tổng quan

Project này implement một hệ thống ASR hoàn chỉnh từ đầu với:
- **Model**: CRNN-CTC (Convolutional Recurrent Neural Network với Connectionist Temporal Classification)
- **Features**: Log Mel-Spectrogram (80 mel bins)
- **Loss**: CTC Loss
- **Decoder**: Greedy Decoding
- **Dataset**: Hỗ trợ LibriSpeech (English) và VietSpeech (Vietnamese)

## 🏗️ Kiến trúc Model

### CRNNCTC Model

```
Input: Audio waveform (16kHz mono)
  ↓
Log Mel-Spectrogram (80 mel bins, hop=160, n_fft=400)
  ↓
CNN Encoder:
  - Conv1d(80 → 128, kernel=5)
  - ReLU
  - Conv1d(128 → 128, kernel=5)
  - ReLU
  ↓
Bidirectional LSTM:
  - 3 layers
  - Hidden size: 256
  - Output: 512 (256×2 bidirectional)
  ↓
Linear Head: 512 → vocab_size
  ↓
Output: CTC logits (T, B, V)
```

**Thông số model:**
- **Total Parameters**: 4,132,715 (~4.1M)
- **Trainable Parameters**: 4,132,715
- **Input**: 16kHz mono audio
- **Mel features**: 80 bins
- **CNN channels**: 128
- **RNN hidden**: 256 (bidirectional → 512)
- **RNN layers**: 3
- **Vocab size**: 107 tokens (hiện tại: `<blank>`, `<unk>`, space, a-z, 0-9, Vietnamese characters)

**Chi tiết kiến trúc:**

1. **CNN Encoder** (Feature Extraction):
   - Conv1d layer 1: 80 input channels → 128 output channels, kernel_size=5, padding=2
   - ReLU activation
   - Conv1d layer 2: 128 → 128 channels, kernel_size=5, padding=2
   - ReLU activation
   - Tổng parameters CNN: ~82K

2. **Bidirectional LSTM** (Sequence Modeling):
   - 3 layers bidirectional LSTM
   - Input size: 128 (từ CNN)
   - Hidden size: 256 per direction
   - Output size: 512 (256 forward + 256 backward)
   - Tổng parameters LSTM: ~4M

3. **Linear Head** (Classification):
   - Input: 512 (từ LSTM)
   - Output: vocab_size (107)
   - Tổng parameters: ~55K

## 💻 Yêu cầu hệ thống

### Hardware (Đã tối ưu cho)
- **CPU**: Ryzen 9 9900X (12 cores, 24 threads)
- **GPU**: RTX 5060 Ti 16GB VRAM
- **RAM**: Khuyến nghị 32GB+

### Software
- Python >= 3.9
- CUDA (cho GPU training)
- PyTorch với CUDA support

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd ai-llm-ss
```

### 2. Tạo virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Kiểm tra cài đặt
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Dataset

### Dataset hiện có

Project hỗ trợ 2 datasets chính:

1. **LibriSpeech Alignments** (English)
   - Location: `data/raw/librispeech_alignments/`
   - Format: WAV files + manifest.csv + timestamps.json
   - Splits: train/val/test

2. **VietSpeech** (Vietnamese)
   - Location: `data/raw/VietSpeech/`
   - Format: WAV files + manifest.csv + timestamps.json
   - Splits: train/val/test

### Dataset đã xử lý

- **Full merged dataset**: `data/processed/full_merged_dataset/`
  - Training samples: **194,451**
  - Format: manifest.csv với các cột:
    - `audio_path`: Đường dẫn đến file audio
    - `transcript`: Text transcription
    - Các metadata khác (tùy dataset)
  - Vocabulary: `data/processed/vocab.json` (107 tokens)
    - Special tokens: `<blank>` (index 0), `<unk>` (index 1)
    - Characters: a-z, 0-9, space, punctuation
    - Vietnamese characters: à, á, â, ã, è, é, ê, ì, í, ò, ó, ô, õ, ù, ú, ý, ă, đ, ơ, ư, và các dấu thanh

### Audio Preprocessing

Tất cả audio được xử lý tự động:
1. **Resample** về 16kHz (nếu cần)
2. **Convert to mono** (nếu stereo)
3. **Extract Log Mel-Spectrogram**:
   - n_fft: 400
   - hop_length: 160 (10ms frames)
   - n_mels: 80
   - Log transform: log(mel + 1e-6)
4. **Optional trimming**: Nếu có timestamps.json, có thể trim audio theo segments

### Dataset Classes

- **`ManifestDataset`**: Đọc từ manifest.csv (khuyến nghị)
  - Hỗ trợ timestamps.json cho trimming
  - Flexible audio root path
  - Robust audio loading (torchaudio + soundfile fallback)

- **`ASRDataset`**: Đọc từ audio_dir và text_dir
  - Đơn giản hơn, phù hợp cho dataset nhỏ
  - Yêu cầu naming convention: `audio.wav` ↔ `audio.txt`

### Chuẩn bị dataset

```bash
python scripts/prepare_data.py
```

## 🚀 Training

### Training với config file (Khuyến nghị)

```bash
python scripts/train_from_config.py config/train_merged.json
```

### Cách chạy nhanh (mặc định 5M CRNN-CTC)

```bash
# 1) Kích hoạt môi trường ảo (nếu có)
source .venv/bin/activate

# 2) Huấn luyện với cấu hình mặc định (char-level, ~5M tham số)
python scripts/train_from_config.py config/train_merged.json

# Ghi chú:
# - Manifest train: data/processed/full_merged_dataset/train/manifest.csv
# - Vocab: data/processed/vocab.json (char-level)
# - Checkpoint sẽ lưu mỗi epoch tại data/results/checkpoints
# - Mô hình cuối lưu tại data/results/asr_ctc.pt
```

### 📺 Trạng thái hiển thị trên Terminal

Khi training đang chạy, bạn sẽ thấy output trực tiếp trên terminal như sau:

```
Using bfloat16 (bf16) precision with AMP
Memory optimization: Emptying CUDA cache after each batch
Gradient accumulation: 5 steps (effective batch size: 160)
Resuming from checkpoint: data/results/checkpoints/checkpoint_epoch_8.pt
Resuming from epoch 9

epoch 9 [||||||||||||||||||||] 20/6080 | loss 2.345
epoch 9 [||||||||||||||||||||] 40/6080 | loss 2.312
epoch 9 [||||||||||||||||||||] 60/6080 | loss 2.298
...
epoch 9 [||||||||||||||||||||] 6080/6080 | loss 2.156
epoch 9 | loss 2.156
Saved checkpoint to data/results/checkpoints/checkpoint_epoch_9.pt

epoch 10 [||||||||||||||||||||] 20/6080 | loss 2.134
...
```

**Giải thích output:**
- `epoch X`: Số epoch hiện tại
- `[||||||||...]`: Progress bar cho batch hiện tại trong epoch
- `20/6080`: Batch hiện tại / Tổng số batches trong epoch
- `loss X.XXX`: Loss của batch hiện tại (sau khi scale lại)
- `epoch X | loss X.XXX`: Average loss của toàn bộ epoch
- `Saved checkpoint to ...`: Đường dẫn checkpoint vừa lưu

**Lưu ý:** 
- Progress được log mỗi `log_interval` batches (mặc định: 20)
- Checkpoint được lưu sau mỗi epoch
- Nếu resume từ checkpoint, sẽ hiển thị "Resuming from epoch X"

### Training trực tiếp

```bash
python scripts/train_asr.py
```

### Cấu hình training hiện tại

File `config/train_merged.json` đã được tối ưu cho Ryzen 9 9900X + RTX 5060 Ti:

```json
{
  "manifest": "data/processed/full_merged_dataset/train/manifest.csv",
  "audio_root": "data/processed/full_merged_dataset/train",
  "vocab": "data/processed/vocab.json",
  "device": "auto",
  "amp": true,
  "batch_size": 32,
  "gradient_accumulation_steps": 5,
  "max_grad_norm": 1.0,
  "empty_cache": true,
  "num_workers": 8,
  "epochs": 20,
  "log_interval": 20,
  "lr": 0.001
}
```

### Chi tiết Training Process

#### Loss Function
- **CTC Loss** với `blank=0` và `zero_infinity=True`
- CTC loss tự động xử lý alignment giữa audio frames và text tokens
- Không cần forced alignment, model tự học mapping

#### Optimizer
- **AdamW** optimizer với learning rate = 0.001
- Gradient clipping với `max_grad_norm=1.0` để ổn định training
- Gradient accumulation: 5 steps (effective batch size = 32 × 5 = 160)

#### Mixed Precision Training
- **Automatic Mixed Precision (AMP)** với bfloat16
- Giảm memory usage ~50% và tăng tốc độ training ~1.5-2x
- Tự động detect GPU support bf16, fallback về fp32 nếu không hỗ trợ

#### Training Loop
1. Load batch từ DataLoader (8 workers, pin_memory enabled)
2. Forward pass với autocast (bf16)
3. Tính CTC loss và scale bởi gradient_accumulation_steps
4. Backward pass (accumulate gradients)
5. Mỗi 5 batches: gradient clipping → optimizer step → zero gradients
6. Empty CUDA cache sau mỗi batch (nếu enabled)
7. Log progress mỗi 20 batches
8. Save checkpoint sau mỗi epoch

### 📊 Xem tiến độ training (Training Progress Monitoring)

Có nhiều cách để theo dõi tiến độ training real-time:

#### 1. Monitor đẹp nhất (Khuyến nghị) ⭐
```bash
python scripts/training_progress_bar.py
```
**Tính năng:**
- Progress bars màu sắc cho epoch, GPU memory, GPU utilization
- Hiển thị loss từ checkpoint
- Ước tính thời gian còn lại dựa trên checkpoint thực tế
- Auto-refresh mỗi 2 giây
- Hiển thị PID process, elapsed time, config

#### 2. Monitor chi tiết
```bash
python scripts/training_progress.py
```
**Tính năng:**
- Progress bars cho epoch và GPU
- Ước tính thời gian còn lại
- Auto-refresh mỗi 3 giây

#### 3. Monitor cơ bản
```bash
python scripts/monitor_training.py
```
**Tính năng:**
- Thông tin cơ bản về process, GPU, checkpoint
- Auto-refresh mỗi 5 giây

#### 4. Bash script đơn giản
```bash
./scripts/watch_training.sh
```
**Tính năng:**
- Script bash đơn giản, không cần Python
- Hiển thị epoch hiện tại, GPU info
- Auto-refresh mỗi 5 giây

#### Thông tin hiển thị:
- ✅ **Training Process**: PID, thời gian đã chạy
- 📊 **GPU Status**: Memory usage, utilization với progress bar
- 📁 **Training Progress**: Epoch hiện tại/tổng số, progress bar, loss
- ⏱️ **Time Estimates**: Thời gian mỗi epoch, thời gian còn lại, tổng thời gian
- ⚙️ **Configuration**: Batch size, gradient accumulation, workers, AMP

**Lưu ý:** Các script này sẽ tự động detect training process đang chạy và hiển thị thông tin real-time. Nhấn `Ctrl+C` để dừng monitoring.

### Checkpoints

- Checkpoints được lưu tại: `data/results/checkpoints/checkpoint_epoch_X.pt`
- Model cuối cùng: `data/results/asr_ctc.pt`

### Resume training

```bash
# Sửa config/train_merged.json, thêm:
"resume": "data/results/checkpoints/checkpoint_epoch_X.pt"

# Hoặc dùng script
./scripts/resume_training.sh
```

## 🔧 Sử dụng Model

### Load và transcribe

#### Basic Usage

```python
import torch
from src.asr.model import CRNNCTC
from src.asr.features import wav_to_logmelspec, ensure_mono16k
from src.asr.decode import greedy_decode
import torchaudio
import json

# Load vocabulary
vocab_path = "data/processed/vocab.json"
vocab = json.load(open(vocab_path, encoding="utf-8"))
itos = {i: c for i, c in enumerate(vocab)}
print(f"Vocabulary size: {len(vocab)}")

# Load model
model_path = "data/results/asr_ctc.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()
print(f"Model loaded on {device}")

# Load audio
audio_path = "audio.wav"
wav, sr = torchaudio.load(audio_path)
print(f"Original: {sr}Hz, {wav.shape}")

# Preprocess audio
wav, sr = ensure_mono16k(wav, sr)  # Ensure 16kHz mono
print(f"After preprocessing: {sr}Hz, {wav.shape}")

# Extract features
feats = wav_to_logmelspec(wav, sr)  # (T, 80)
feats = feats.unsqueeze(0).to(device)  # (1, T, 80)
print(f"Features shape: {feats.shape}")

# Transcribe
with torch.no_grad():
    logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
    # logits: (T, B, V), lens: (B,)
    text = greedy_decode(logits.cpu(), itos)[0]

print(f"Transcription: {text}")
```

#### Batch Processing

```python
import torch
from torch.utils.data import DataLoader
from src.asr.dataset import ManifestDataset, collate_batch
from src.asr.model import CRNNCTC
from src.asr.decode import greedy_decode
import json

# Load model
vocab = json.load(open("data/processed/vocab.json"))
itos = {i: c for i, c in enumerate(vocab)}
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
model.load_state_dict(torch.load("data/results/asr_ctc.pt"))
model.eval()

# Create dataset
dataset = ManifestDataset(
    manifest_path="data/processed/full_merged_dataset/test/manifest.csv",
    vocab_path="data/processed/vocab.json",
    audio_root="data/processed/full_merged_dataset/test"
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch)

# Process batches
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

results = []
for X, Xlen, Y, Ylen in dataloader:
    X, Xlen = X.to(device), Xlen.to(device)
    
    with torch.no_grad():
        logits, out_lens = model(X, Xlen)
        texts = greedy_decode(logits.cpu(), itos)
    
    # Decode ground truth
    ground_truths = []
    for y, ylen in zip(Y, Ylen):
        gt = "".join([itos.get(int(idx), "") for idx in y[:ylen] if int(idx) != 0])
        ground_truths.append(gt)
    
    for pred, gt in zip(texts, ground_truths):
        results.append({"prediction": pred, "ground_truth": gt})
        print(f"Pred: {pred}")
        print(f"GT:   {gt}\n")
```

### CTC Decoding

Model sử dụng **Greedy Decoding**:
1. Lấy argmax của logits tại mỗi time step
2. Loại bỏ blank tokens (index 0)
3. Loại bỏ duplicate tokens liên tiếp
4. Map indices về characters

**Lưu ý**: Greedy decoding đơn giản và nhanh, nhưng không tối ưu. Có thể cải thiện bằng:
- Beam search decoding
- Language model integration
- External lexicon

## 🌐 API Server

### Khởi động API server

```bash
python scripts/serve_asr.py
```

Hoặc với tùy chọn:

```bash
python scripts/serve_asr.py --host 0.0.0.0 --port 8001
```

### API Endpoints

- **GET /** - Thông tin API
- **GET /health** - Health check và model status
- **GET /model/info** - Thông tin model
- **POST /transcribe** - Transcribe audio file

### Sử dụng API

```bash
# Health check
curl http://localhost:8001/health

# Transcribe audio
curl -X POST "http://localhost:8001/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

### API Documentation

Truy cập Swagger UI tại: `http://localhost:8001/docs`

## ⚙️ Cấu hình tối ưu

### Tối ưu cho Ryzen 9 9900X + RTX 5060 Ti 16GB

Các thông số đã được tối ưu:

| Thông số | Giá trị | Lý do |
|----------|---------|-------|
| `batch_size` | 32 | Cân bằng giữa memory và throughput |
| `gradient_accumulation_steps` | 5 | Effective batch = 160, tối ưu convergence |
| `num_workers` | 8 | Tận dụng 24 threads của Ryzen 9 |
| `amp` | true | Mixed precision (bfloat16) để tăng tốc |
| `empty_cache` | true | Giải phóng memory sau mỗi batch |
| `pin_memory` | true | Tăng tốc GPU transfer |

### Environment variables

```bash
# Giảm memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Ước tính thời gian training

Với dataset 194,451 samples:
- **Thời gian/epoch**: ~29-42 phút
- **Tổng thời gian (20 epochs)**: ~9-14 giờ

Tính toán chi tiết:
```bash
python scripts/estimate_training_time.py config/train_merged.json
```

## 📁 Cấu trúc Project

### Cấu trúc tổng quan

```
ai-llm-ss/
├── config/                         # Configuration files
├── data/                           # Data directory
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed datasets
│   └── results/                    # Training results
├── docs/                           # Documentation
├── experiments/                    # Experiment results
├── notebooks/                       # Jupyter notebooks
├── scripts/                        # Utility scripts
├── src/                            # Source code
│   └── asr/                        # ASR module
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
└── README.md                       # This file
```

### Chi tiết từng file và thư mục

#### 📂 Root Directory

| File/Directory | Mô tả |
|---------------|-------|
| `README.md` | Tài liệu chính của project (file này) |
| `requirements.txt` | Danh sách Python dependencies cần thiết |
| `pyproject.toml` | Project metadata và package configuration |

#### 📂 config/

| File | Mô tả | Nội dung |
|------|-------|----------|
| `train_merged.json` | **Config file chính cho training** | Chứa tất cả hyperparameters: batch_size, learning_rate, epochs, paths, etc. Được sử dụng bởi `train_from_config.py` |

**Ví dụ nội dung:**
```json
{
  "manifest": "data/processed/full_merged_dataset/train/manifest.csv",
  "audio_root": "data/processed/full_merged_dataset/train",
  "vocab": "data/processed/vocab.json",
  "batch_size": 32,
  "epochs": 20,
  "lr": 0.001,
  ...
}
```

#### 📂 src/

##### `src/__init__.py`
- File khởi tạo package root
- Hiện tại trống, có thể thêm package-level imports

##### `src/asr/` - ASR Module

| File | Dòng code | Chức năng chính | Chi tiết |
|------|-----------|-----------------|----------|
| `__init__.py` | - | Package initialization | Khởi tạo ASR module |
| `model.py` | 26 | **CRNNCTC Model** | Định nghĩa kiến trúc model:<br>- CNN encoder (2 Conv1d layers)<br>- Bidirectional LSTM (3 layers)<br>- Linear classification head<br>- Forward pass logic |
| `features.py` | 20 | **Audio Feature Extraction** | Hàm xử lý audio:<br>- `ensure_mono16k()`: Resample và convert to mono<br>- `wav_to_logmelspec()`: Extract log mel-spectrogram (80 bins) |
| `dataset.py` | 96 | **Dataset Classes** | 2 dataset classes:<br>- `ASRDataset`: Đọc từ audio_dir + text_dir<br>- `ManifestDataset`: Đọc từ manifest.csv (khuyến nghị)<br>- `collate_batch()`: Batch collation với padding |
| `train_ctc.py` | 175 | **Training Loop** | Training script chính:<br>- Parse arguments<br>- Setup DataLoader<br>- Initialize model, loss, optimizer<br>- Training loop với gradient accumulation<br>- Checkpoint saving<br>- Mixed precision training (AMP) |
| `decode.py` | 15 | **CTC Decoding** | `greedy_decode()`:<br>- Greedy decoding từ CTC logits<br>- Loại bỏ blank tokens<br>- Loại bỏ duplicate tokens |
| `tokenizer.py` | 28 | **Text Tokenization** | Vocabulary management:<br>- `build_char_vocab()`: Build vocabulary từ transcripts<br>- `encode()`: Encode text to indices<br>- `save_vocab()`: Save vocabulary to JSON |
| `api.py` | 193 | **FastAPI Server** | REST API cho ASR:<br>- `/health`: Health check<br>- `/model/info`: Model information<br>- `/transcribe`: Transcribe audio file<br>- Auto preprocessing support<br>- CORS enabled |

#### 📂 scripts/

| File | Loại | Dòng code | Chức năng |
|------|------|-----------|-----------|
| `train_from_config.py` | Python | 32 | **Training với config file**<br>- Đọc JSON config<br>- Parse và convert sang command-line args<br>- Gọi `train_ctc.py` với args |
| `train_asr.py` | Python | 23 | **Training script đơn giản**<br>- Default settings<br>- Wrapper cho `train_ctc.py`<br>- Tự động detect CUDA |
| `serve_asr.py` | Python | 44 | **API Server Launcher**<br>- Start FastAPI server với uvicorn<br>- Configurable host/port<br>- Support reload mode |
| `prepare_data.py` | Python | 21 | **Dataset Preparation**<br>- Build vocabulary từ transcripts<br>- Save vocab.json<br>- Process raw data |
| `monitor_training.py` | Python | 193 | **Real-time Training Monitor**<br>- Monitor process status<br>- GPU memory/usage<br>- Checkpoint progress<br>- Time estimates<br>- Auto-refresh mỗi 5s |
| `watch_training.sh` | Bash | ~40 | **Simple Training Monitor**<br>- Bash version của monitor<br>- Hiển thị process, GPU, checkpoints |
| `estimate_training_time.py` | Python | ~60 | **Training Time Estimator**<br>- Tính toán thời gian training<br>- Dựa trên dataset size và config<br>- Hiển thị estimates chi tiết |
| `test_api.py` | Python | 108 | **API Testing Script**<br>- Test `/health` endpoint<br>- Test `/model/info` endpoint<br>- Test `/transcribe` endpoint<br>- Comprehensive error handling |
| `resume_training.sh` | Bash | - | **Resume Training Helper**<br>- Script để resume từ checkpoint |

#### 📂 data/

##### `data/raw/` - Raw Datasets

| Directory | Mô tả |
|-----------|-------|
| `librispeech_alignments/` | LibriSpeech English dataset với alignments:<br>- `train/`, `val/`, `test/` splits<br>- `manifest.csv`: Metadata<br>- `timestamps.json`: Word/phoneme alignments<br>- `README.md`: Dataset documentation |
| `VietSpeech/` | Vietnamese VietSpeech dataset:<br>- `train/`, `val/`, `test/` splits<br>- `manifest.csv`: Metadata<br>- `timestamps.json`: Word alignments<br>- `README.md`: Dataset documentation |
| `DATASETS_README.md` | Tài liệu về datasets |

##### `data/processed/` - Processed Datasets

| File/Directory | Mô tả |
|----------------|-------|
| `vocab.json` | **Vocabulary file** (107 tokens):<br>- JSON array của characters<br>- Index 0: `<blank>`<br>- Index 1: `<unk>`<br>- Còn lại: a-z, 0-9, Vietnamese chars |
| `merged_dataset/` | Merged dataset (cũ):<br>- `train/`, `val/`, `test/`<br>- `manifest.csv`<br>- `timestamps.json` |
| `full_merged_dataset/` | **Full merged dataset (chính)**:<br>- `train/`: 194,451 samples<br>  - `audio/`: WAV files<br>  - `manifest.csv`: Training manifest<br>  - `manifest.csv.backup`: Backup<br>- `val/`: Validation set<br>- `test/`: Test set |

##### `data/results/` - Training Results

| File/Directory | Mô tả |
|----------------|-------|
| `asr_ctc.pt` | **Final trained model**<br>- Model state dict<br>- Load với `torch.load()`<br>- ~48MB |
| `checkpoints/` | Training checkpoints:<br>- `checkpoint_epoch_1.pt`<br>- `checkpoint_epoch_2.pt`<br>- `checkpoint_epoch_3.pt`<br>- ...<br>Mỗi checkpoint chứa:<br>- `epoch`: Epoch number<br>- `model_state_dict`<br>- `optimizer_state_dict`<br>- `loss`: Average loss<br>- `scaler_state_dict` (nếu AMP) |

#### 📂 tests/

| File | Mô tả |
|------|-------|
| `test_decode.py` | Unit test cho `greedy_decode()`:<br>- Test output shapes<br>- Test với dummy logits |

#### 📂 docs/

- Thư mục cho documentation (hiện tại trống)
- Có thể thêm design docs, evaluation protocols, etc.

#### 📂 experiments/

- `reports/`: Experiment reports và results
- Lưu kết quả experiments, metrics, comparisons

#### 📂 notebooks/

- Jupyter notebooks cho exploration và analysis
- Data visualization, model analysis, etc.

### File Dependencies Flow

```
User Input
    ↓
scripts/train_from_config.py
    ↓ (reads)
config/train_merged.json
    ↓ (calls)
src/asr/train_ctc.py
    ↓ (uses)
src/asr/dataset.py → data/processed/full_merged_dataset/
src/asr/model.py
src/asr/features.py
    ↓ (saves)
data/results/checkpoints/checkpoint_epoch_X.pt
    ↓ (final)
data/results/asr_ctc.pt
```

### API Flow

```
User Request
    ↓
scripts/serve_asr.py
    ↓ (starts)
src/asr/api.py (FastAPI)
    ↓ (uses)
src/asr/model.py
src/asr/features.py
src/asr/decode.py
    ↓ (loads)
data/results/asr_ctc.pt
data/processed/vocab.json
    ↓ (returns)
Transcription Result
```

### Data Flow trong Training

```
Audio Files (WAV)
    ↓
src/asr/dataset.py (ManifestDataset)
    ↓ (loads)
src/asr/features.py (wav_to_logmelspec)
    ↓ (extracts)
Log Mel-Spectrogram (T, 80)
    ↓
src/asr/model.py (CRNNCTC)
    ↓ (forward)
CTC Logits (T, B, V)
    ↓
torch.nn.CTCLoss
    ↓
Loss Value
    ↓
Backward Pass
    ↓
Optimizer Step
    ↓
Checkpoint Save
```

## 🛠️ Scripts

### Training Scripts

- **`train_from_config.py`**: Training với config file (khuyến nghị)
- **`train_asr.py`**: Training với default settings

### Monitoring Scripts

- **`monitor_training.py`**: Python monitor với thông tin chi tiết
- **`watch_training.sh`**: Bash script đơn giản
- **`estimate_training_time.py`**: Ước tính thời gian training

### Utility Scripts

- **`prepare_data.py`**: Chuẩn bị và merge datasets
- **`serve_asr.py`**: Khởi động API server
- **`test_api.py`**: Test API endpoints
- **`resume_training.sh`**: Resume training từ checkpoint

## 📈 Performance

### Model Performance

- **Architecture**: CRNN-CTC
- **Total Parameters**: 4,132,715 (~4.1M)
- **Trainable Parameters**: 4,132,715
- **Model Size**: ~48MB (checkpoint file)
- **Input**: 16kHz mono audio
- **Features**: 80 mel-spectrogram bins
- **Training**: Mixed precision (bfloat16) với AMP
- **Inference Speed**: 
  - GPU: ~10-50x real-time (tùy audio length)
  - CPU: ~1-5x real-time

### Training Performance

Với cấu hình tối ưu (Ryzen 9 9900X + RTX 5060 Ti 16GB):
- **Dataset**: 194,451 training samples
- **Batches per epoch**: 6,077 (batch_size=32)
- **Effective batch size**: 160 (với gradient accumulation)
- **GPU Utilization**: 30-50%
- **GPU Memory Usage**: ~13-14GB / 16GB (83-87%)
- **CPU Utilization**: ~30-40% (8 workers)
- **Time per epoch**: ~29-42 phút
- **Total training time (20 epochs)**: ~9-14 giờ
- **Throughput**: ~150-200 samples/second

### Memory Breakdown

- **Model parameters**: ~16MB (fp32) / ~8MB (fp16/bf16)
- **Optimizer states**: ~32MB (AdamW với 2x momentum buffers)
- **Gradients**: ~16MB
- **Activations**: ~2-4GB (tùy batch size và sequence length)
- **DataLoader buffers**: ~500MB-1GB (8 workers)
- **PyTorch overhead**: ~1-2GB

### Inference Performance

```python
# Benchmark inference speed
import time
import torch

# Single sample
audio_length_seconds = 5.0
feats_length = int(audio_length_seconds * 16000 / 160)  # ~500 frames

model.eval()
with torch.no_grad():
    # Warmup
    dummy_input = torch.randn(1, feats_length, 80).to(device)
    _ = model(dummy_input, torch.tensor([feats_length], device=device))
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input, torch.tensor([feats_length], device=device))
    elapsed = time.time() - start
    
    avg_time = elapsed / 100
    real_time_factor = audio_length_seconds / avg_time
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Real-time factor: {real_time_factor:.2f}x")
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Triệu chứng**: `torch.OutOfMemoryError: CUDA out of memory`

**Giải pháp**:
1. Giảm `batch_size` trong config:
```json
{
  "batch_size": 24,  // Giảm từ 32
  "gradient_accumulation_steps": 7  // Tăng để giữ effective batch ~160
}
```

2. Giảm `num_workers`:
```json
{
  "num_workers": 4  // Giảm từ 8
}
```

3. Enable `empty_cache`:
```json
{
  "empty_cache": true
}
```

4. Set environment variable:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

5. Kiểm tra GPU memory:
```bash
nvidia-smi
# Nếu có process khác đang dùng GPU, kill chúng
```

### Training chậm

**Triệu chứng**: Training mất quá nhiều thời gian, GPU utilization thấp

**Giải pháp**:
1. Kiểm tra `num_workers`:
   - Nên = số CPU cores / 3
   - Với Ryzen 9 9900X (24 threads): 8 workers là tối ưu
   - Quá nhiều workers có thể gây overhead

2. Kiểm tra `pin_memory`:
   - Tự động enable nếu batch_size <= 128
   - Đảm bảo GPU có đủ memory cho pin_memory

3. Kiểm tra GPU utilization:
```bash
watch -n 1 nvidia-smi
# GPU utilization nên > 30%
```

4. Kiểm tra I/O bottleneck:
```bash
# Nếu disk I/O chậm, dataset nên ở SSD
# Hoặc giảm num_workers để giảm I/O load
```

5. Kiểm tra AMP:
```bash
# Đảm bảo AMP enabled và GPU hỗ trợ bf16
python -c "import torch; print(torch.cuda.is_bf16_supported())"
```

### Model không load

**Triệu chứng**: `FileNotFoundError` hoặc `KeyError` khi load model

**Giải pháp**:
1. Kiểm tra model path:
```python
import os
model_path = "data/results/asr_ctc.pt"
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
```

2. Kiểm tra vocab path:
```python
import json
vocab_path = "data/processed/vocab.json"
vocab = json.load(open(vocab_path))
print(f"Vocab size: {len(vocab)}")
```

3. Kiểm tra model architecture match:
```python
# Vocab size phải khớp với model
vocab = json.load(open("data/processed/vocab.json"))
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
# Nếu vocab size khác, model sẽ không load được
```

### Loss không giảm / Training không hội tụ

**Triệu chứng**: Loss cao và không giảm sau nhiều epochs

**Giải pháp**:
1. Kiểm tra learning rate:
   - LR = 0.001 là hợp lý
   - Có thể thử learning rate schedule (cosine, step decay)

2. Kiểm tra gradient:
```python
# Thêm vào training loop để debug
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item()}")
```

3. Kiểm tra data quality:
   - Đảm bảo audio và transcript khớp
   - Kiểm tra có samples bị corrupt không

4. Kiểm tra CTC loss:
   - CTC loss có thể rất cao ở đầu training (normal)
   - Nếu loss = inf, kiểm tra `zero_infinity=True` trong CTCLoss

### Audio loading errors

**Triệu chứng**: `RuntimeError` khi load audio

**Giải pháp**:
1. Install soundfile (fallback):
```bash
pip install soundfile
```

2. Kiểm tra audio format:
   - Hỗ trợ: WAV, MP3, FLAC, OGG
   - Nên dùng WAV 16kHz mono để tối ưu

3. Kiểm tra corrupted files:
```python
import torchaudio
try:
    wav, sr = torchaudio.load("problematic_audio.wav")
except Exception as e:
    print(f"Error: {e}")
```

### Checkpoint không resume được

**Triệu chứng**: Training không resume từ checkpoint

**Giải pháp**:
1. Kiểm tra checkpoint format:
```python
checkpoint = torch.load("checkpoint_epoch_X.pt")
print(checkpoint.keys())
# Should have: 'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'
```

2. Đảm bảo config match:
   - Model architecture phải giống nhau
   - Vocab size phải giống nhau

3. Load checkpoint manually:
```python
checkpoint = torch.load("checkpoint_epoch_X.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📝 License

[Thêm license của bạn]

## 👥 Contributors

[Thêm contributors]

## 🔬 Chi tiết kỹ thuật

### CTC Loss

Connectionist Temporal Classification (CTC) là loss function đặc biệt cho sequence-to-sequence tasks:
- **Không cần alignment**: CTC tự động học alignment giữa input frames và output tokens
- **Blank token**: Sử dụng `<blank>` (index 0) để handle:
  - Silence/no speech
  - Repetitions (cần blank giữa các ký tự giống nhau)
- **Forward-backward algorithm**: CTC sử dụng dynamic programming để tính loss hiệu quả
- **Zero infinity**: `zero_infinity=True` để tránh NaN khi alignment không khả thi

### Mel-Spectrogram Features

- **Sample rate**: 16kHz (industry standard cho speech)
- **Frame size**: 400 samples (25ms) với n_fft=400
- **Hop size**: 160 samples (10ms) → 100 frames/second
- **Mel bins**: 80 (tối ưu cho speech recognition)
- **Log transform**: log(mel + 1e-6) để compress dynamic range

### Vocabulary và Tokenization

- **Character-level**: Mỗi character là một token
- **Special tokens**:
  - `<blank>` (index 0): CTC blank token
  - `<unk>` (index 1): Unknown character
- **Case handling**: Tất cả text được lowercase trước khi tokenize
- **Multilingual**: Hỗ trợ English và Vietnamese characters

### Data Augmentation

Hiện tại chưa có data augmentation, nhưng có thể thêm:
- Speed perturbation (±10-20%)
- Volume normalization
- Noise injection
- Time masking (SpecAugment)

### Model Regularization

- **Gradient clipping**: max_grad_norm=1.0
- **Dropout**: Có thể thêm vào LSTM layers
- **Weight decay**: Có thể thêm vào AdamW optimizer

## 💡 Examples

### Example 1: Transcribe single file

```python
#!/usr/bin/env python3
"""Simple transcription example."""
import torch
import torchaudio
from src.asr.model import CRNNCTC
from src.asr.features import wav_to_logmelspec, ensure_mono16k
from src.asr.decode import greedy_decode
import json
import sys

def transcribe(audio_path, model_path="data/results/asr_ctc.pt"):
    # Load vocab
    vocab = json.load(open("data/processed/vocab.json", encoding="utf-8"))
    itos = {i: c for i, c in enumerate(vocab)}
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    wav, sr = ensure_mono16k(wav, sr)
    feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)
    
    # Transcribe
    with torch.no_grad():
        logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
        text = greedy_decode(logits.cpu(), itos)[0]
    
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file.wav>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = transcribe(audio_path)
    print(f"Transcription: {result}")
```

### Example 2: Batch evaluation

```python
#!/usr/bin/env python3
"""Evaluate model on test set."""
import torch
from torch.utils.data import DataLoader
from src.asr.dataset import ManifestDataset, collate_batch
from src.asr.model import CRNNCTC
from src.asr.decode import greedy_decode
import json
from jiwer import wer, cer

def evaluate(model_path, test_manifest, audio_root):
    # Load vocab
    vocab = json.load(open("data/processed/vocab.json", encoding="utf-8"))
    itos = {i: c for i, c in enumerate(vocab)}
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    # Create dataset
    dataset = ManifestDataset(
        manifest_path=test_manifest,
        vocab_path="data/processed/vocab.json",
        audio_root=audio_root
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_batch)
    
    # Evaluate
    predictions = []
    references = []
    
    for X, Xlen, Y, Ylen in dataloader:
        X, Xlen = X.to(device), Xlen.to(device)
        
        with torch.no_grad():
            logits, out_lens = model(X, Xlen)
            texts = greedy_decode(logits.cpu(), itos)
        
        # Decode ground truth
        for y, ylen in zip(Y, Ylen):
            gt = "".join([itos.get(int(idx), "") for idx in y[:ylen] if int(idx) != 0])
            references.append(gt)
        
        predictions.extend(texts)
    
    # Calculate metrics
    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)
    
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")
    print(f"Character Error Rate (CER): {char_error_rate:.4f}")
    
    return word_error_rate, char_error_rate

if __name__ == "__main__":
    evaluate(
        "data/results/asr_ctc.pt",
        "data/processed/full_merged_dataset/test/manifest.csv",
        "data/processed/full_merged_dataset/test"
    )
```

### Example 3: Custom training loop

```python
#!/usr/bin/env python3
"""Custom training loop với validation."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.asr.dataset import ManifestDataset, collate_batch
from src.asr.model import CRNNCTC
from src.asr.train_ctc import main as train_main
# ... (chi tiết trong train_ctc.py)
```

## 🙏 Acknowledgments

- **LibriSpeech dataset**: Open-source English speech dataset
- **VietSpeech dataset**: Vietnamese speech dataset
- **PyTorch team**: Deep learning framework
- **CTC algorithm**: Connectionist Temporal Classification paper

## 📚 References

- [CTC Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf): Connectionist Temporal Classification
- [LibriSpeech](https://www.openslr.org/12/): Speech recognition dataset
- [PyTorch CTC Loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

---

**Lưu ý**: README này được tạo tự động dựa trên codebase hiện tại. Cập nhật khi có thay đổi trong project.

**Version**: 1.0.0  
**Last Updated**: 2024-12-07

