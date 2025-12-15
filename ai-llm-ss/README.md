# AI-LLM-SS: Automatic Speech Recognition với CTC

Hệ thống ASR (Automatic Speech Recognition) từ đầu sử dụng kiến trúc CRNN-CTC, được tối ưu cho training trên Ryzen 9 9900X và RTX 5060 Ti 16GB.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc Model](#kiến-trúc-model)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation & Metrics](#evaluation--metrics)
- [Visualization](#visualization)
- [Sử dụng Model](#sử-dụng-model)
- [API Server](#api-server)
- [Cấu hình tối ưu](#cấu-hình-tối-ưu)
- [Cấu trúc Project](#cấu-trúc-project)
- [Scripts](#scripts)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)

## 🎯 Tổng quan

Project này implement một hệ thống ASR hoàn chỉnh từ đầu với:
- **Model**: CRNN-CTC (Convolutional Recurrent Neural Network với Connectionist Temporal Classification)
- **Features**: Log Mel-Spectrogram (80 mel bins)
- **Loss**: CTC Loss
- **Decoder**: Greedy Decoding
- **Dataset**: Hỗ trợ LibriSpeech (English) và VietSpeech (Vietnamese)
- **Evaluation**: WER, CER, SER, RTF metrics với visualization
- **Logging**: TensorBoard integration

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

**Dependencies chính:**
- `torch`, `torchaudio` - PyTorch và audio processing
- `numpy` - Numerical computing
- `librosa` - Audio analysis
- `jiwer` - WER/CER calculation
- `fastapi`, `uvicorn` - API server
- `matplotlib` - Visualization
- `tensorboard` - Training visualization
- `tqdm` - Progress bars

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
  "lr": 0.001,
  "log_dir": "runs/asr_ctc"
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
8. **TensorBoard logging** mỗi log_interval batches và mỗi epoch
9. Save checkpoint sau mỗi epoch

### TensorBoard Logging

Training tự động log vào TensorBoard:

```bash
# Xem TensorBoard trong khi training
tensorboard --logdir=runs
```

**Metrics được log:**
- `Loss/Train_step` - Loss mỗi log_interval batches
- `Loss/Train_epoch` - Average loss mỗi epoch

**Logs được lưu tại:** `runs/asr_ctc/run_YYYYMMDD-HHMMSS/`

### Training Progress Monitoring

Có nhiều cách để theo dõi tiến độ training real-time:

#### 1. TensorBoard (Khuyến nghị) ⭐
```bash
tensorboard --logdir=runs
```
Xem biểu đồ loss real-time trên web interface.

#### 2. Monitor đẹp nhất
```bash
python scripts/training_progress_bar.py
```
**Tính năng:**
- Progress bars màu sắc cho epoch, GPU memory, GPU utilization
- Hiển thị loss từ checkpoint
- Ước tính thời gian còn lại
- Auto-refresh mỗi 2 giây

#### 3. Monitor chi tiết
```bash
python scripts/monitor_training.py
```

#### 4. Bash script đơn giản
```bash
./scripts/watch_training.sh
```

### Checkpoints

- Checkpoints được lưu tại: `data/results/checkpoints/checkpoint_epoch_X.pt`
- Mỗi checkpoint chứa:
  - `epoch`: Số epoch
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: Optimizer state
  - `loss`: Average loss của epoch
  - `scaler_state_dict`: AMP scaler state (nếu có)

### Resume training

```bash
# Sửa config/train_merged.json, thêm:
"resume": "data/results/checkpoints/checkpoint_epoch_X.pt"

# Hoặc dùng script
./scripts/resume_training.sh
```

## 📈 Evaluation & Metrics

### Test trên Full Dataset

Chạy evaluation trên toàn bộ test set:

```bash
python scripts/test_full_dataset.py \
  --checkpoint data/results/checkpoints/checkpoint_epoch_12.pt \
  --vocab data/processed/vocab.json \
  --test_manifest data/processed/test/manifest.csv \
  --audio_root data/processed/test \
  --batch_size 16 \
  --output experiments/reports/all_predictions_epoch12.json
```

### Metrics được tính toán

Script tự động tính toán và lưu các metrics sau:

#### Accuracy Metrics
- **WER (Word Error Rate)**: Tỷ lệ lỗi từ - chỉ số quan trọng nhất
- **CER (Character Error Rate)**: Tỷ lệ lỗi ký tự - quan trọng cho Tiếng Việt
- **SER (Sentence Error Rate)**: Tỷ lệ câu có lỗi
- **Exact Match Accuracy**: Tỷ lệ câu hoàn toàn đúng
- **Word-level Accuracy**: Tỷ lệ câu đúng từng từ

#### Performance Metrics
- **RTF (Real-Time Factor)**: Tốc độ xử lý so với độ dài audio
  - RTF < 1: Nhanh hơn thời gian thực ✅
  - RTF = 1: Bằng thời gian thực
  - RTF > 1: Chậm hơn thời gian thực
- **Total Audio Duration**: Tổng thời lượng audio (giây)
- **Total Inference Time**: Tổng thời gian inference (giây)

### Kết quả Test hiện tại (Epoch 12)

```
Model Epoch: 12
Total samples tested: 32,818

Metrics:
  Word Error Rate (WER):     47.37%
  Character Error Rate (CER): 21.04%
  Exact Match Accuracy:       0.41% (134/32818)
  Word-level Accuracy:        0.39% (128/32818)
  Sentence Error Rate (SER): 99.61% (32690/32818)
  Real-Time Factor (RTF):     0.0007 (1440.9x real-time)
    Total audio seconds:      227,287.43
    Total inference seconds:  157.74
```

**Files được tạo:**
- `experiments/reports/all_predictions_epoch12.json` - Tất cả predictions
- `experiments/reports/metrics.json` - Metrics summary

## 📊 Visualization

### Tạo Biểu Đồ Training Report

```bash
# Biểu đồ tổng hợp (linear scale)
python scripts/visualize_training_report.py

# Biểu đồ log scale (3 subplots riêng)
python scripts/plot_log_scale_metrics.py

# Biểu đồ kết hợp log scale (Loss, WER, CER cùng biểu đồ)
python scripts/plot_combined_log_metrics.py
```

**Output files:**
- `experiments/reports/training_report.png` - Báo cáo tổng hợp
- `experiments/reports/log_scale_metrics.png` - Loss, WER, CER với log scale
- `experiments/reports/combined_log_metrics.png` - Kết hợp cả 3 metrics

### Tạo Báo Cáo Đánh Giá

```bash
python scripts/generate_evaluation_report.py
```

**Output file:**
- `experiments/reports/evaluation_report.md` - Báo cáo markdown chi tiết

Báo cáo bao gồm:
- Tóm tắt metrics
- Phân tích điểm mạnh/yếu
- Khuyến nghị cải thiện
- So sánh với tiêu chuẩn ngành

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

# Load model
model_path = "data/results/checkpoints/checkpoint_epoch_12.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))

checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device).eval()

# Load audio
audio_path = "audio.wav"
wav, sr = torchaudio.load(audio_path)

# Preprocess audio
wav, sr = ensure_mono16k(wav, sr)  # Ensure 16kHz mono

# Extract features
feats = wav_to_logmelspec(wav, sr)  # (T, 80)
feats = feats.unsqueeze(0).to(device)  # (1, T, 80)

# Transcribe
with torch.no_grad():
    logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
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

checkpoint = torch.load("data/results/checkpoints/checkpoint_epoch_12.pt", map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Create dataset
dataset = ManifestDataset(
    manifest_path="data/processed/test/manifest.csv",
    vocab_path="data/processed/vocab.json",
    audio_root="data/processed/test"
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
│   └── train_merged.json          # Training config
├── data/                           # Data directory
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed datasets
│   │   ├── vocab.json             # Vocabulary file
│   │   └── test/                   # Test dataset
│   └── results/                    # Training results
│       └── checkpoints/            # Model checkpoints
├── docs/                           # Documentation
├── experiments/                    # Experiment results
│   └── reports/                    # Evaluation reports & plots
├── notebooks/                       # Jupyter notebooks
├── scripts/                        # Utility scripts
│   ├── train_asr.py               # Training wrapper
│   ├── train_from_config.py       # Training from config
│   ├── test_full_dataset.py       # Full dataset evaluation
│   ├── visualize_training_report.py # Training visualization
│   ├── plot_log_scale_metrics.py   # Log scale plots
│   ├── plot_combined_log_metrics.py # Combined metrics plot
│   ├── generate_evaluation_report.py # Evaluation report
│   ├── serve_asr.py               # API server
│   └── ...                        # Other utilities
├── src/                            # Source code
│   └── asr/                        # ASR module
│       ├── model.py               # CRNNCTC model
│       ├── features.py            # Audio feature extraction
│       ├── dataset.py              # Dataset classes
│       ├── decode.py               # CTC decoding
│       ├── train_ctc.py            # Training loop
│       ├── tokenizer.py            # Tokenizer utilities
│       └── api.py                  # FastAPI endpoints
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
└── README.md                       # This file
```

### Chi tiết Modules

#### `src/asr/model.py`
- **CRNNCTC**: Model chính với CNN encoder + Bidirectional LSTM + Linear head

#### `src/asr/features.py`
- **wav_to_logmelspec**: Extract log mel-spectrogram từ audio
- **ensure_mono16k**: Resample và convert về mono 16kHz

#### `src/asr/dataset.py`
- **ManifestDataset**: Dataset từ manifest.csv (khuyến nghị)
- **ASRDataset**: Dataset từ audio_dir và text_dir
- **collate_batch**: Batch collation function

#### `src/asr/decode.py`
- **greedy_decode**: Greedy CTC decoding

#### `src/asr/train_ctc.py`
- **main**: Training loop với CTC loss
- Hỗ trợ: AMP, gradient accumulation, checkpointing, TensorBoard logging

#### `src/asr/api.py`
- FastAPI endpoints cho ASR service

## 📜 Scripts

### Training Scripts

| Script | Mô tả |
|--------|-------|
| `train_asr.py` | Wrapper script cho training với defaults |
| `train_from_config.py` | Training từ config file (khuyến nghị) |
| `resume_training.sh` | Resume training từ checkpoint |

### Evaluation Scripts

| Script | Mô tả |
|--------|-------|
| `test_full_dataset.py` | Test trên full dataset với WER/CER/SER/RTF |
| `test_model.py` | Test model cơ bản |
| `test_model_detailed.py` | Test với phân tích chi tiết |

### Visualization Scripts

| Script | Mô tả |
|--------|-------|
| `visualize_training_report.py` | Tạo biểu đồ training report tổng hợp |
| `plot_log_scale_metrics.py` | Vẽ Loss, WER, CER với log scale (3 subplots) |
| `plot_combined_log_metrics.py` | Vẽ kết hợp Loss, WER, CER với log scale |
| `generate_evaluation_report.py` | Tạo báo cáo đánh giá markdown |

### Monitoring Scripts

| Script | Mô tả |
|--------|-------|
| `training_progress_bar.py` | Monitor training với progress bars đẹp |
| `monitor_training.py` | Monitor training cơ bản |
| `watch_training.sh` | Bash script monitor đơn giản |

### Utility Scripts

| Script | Mô tả |
|--------|-------|
| `prepare_data.py` | Chuẩn bị và xử lý dataset |
| `serve_asr.py` | Khởi động API server |
| `test_api.py` | Test API endpoints |
| `estimate_training_time.py` | Ước tính thời gian training |

## 🔬 Chi tiết kỹ thuật

### CTC Loss

CTC (Connectionist Temporal Classification) loss cho phép:
- Không cần forced alignment giữa audio và text
- Model tự học mapping từ audio frames → text tokens
- Xử lý được các từ có độ dài khác nhau

### Feature Extraction

- **Sample Rate**: 16kHz (mono)
- **Window**: 400 samples (25ms)
- **Hop**: 160 samples (10ms)
- **Mel bins**: 80
- **Normalization**: Log transform với epsilon=1e-6

### Training Optimizations

1. **Mixed Precision (AMP)**: Giảm memory và tăng tốc độ
2. **Gradient Accumulation**: Simulate batch size lớn hơn
3. **Gradient Clipping**: Ổn định training
4. **Pin Memory**: Tăng tốc GPU transfer
5. **Multiple Workers**: Parallel data loading
6. **Empty Cache**: Giải phóng GPU memory

### Model Performance

**Current Results (Epoch 12):**
- WER: 47.37%
- CER: 21.04%
- RTF: 0.0007 (1440x real-time) ✅
- Model size: ~47MB

**Điểm mạnh:**
- Tốc độ xử lý rất nhanh (1440x real-time)
- Model nhỏ gọn (~4.1M parameters)
- Phù hợp cho edge deployment

**Cần cải thiện:**
- WER còn cao, cần thêm dữ liệu training
- CER cần cải thiện cho Tiếng Việt có dấu

## 📝 License

[Thêm license của bạn]

## 🙏 Acknowledgments

- LibriSpeech dataset
- VietSpeech dataset
- PyTorch team
- CTC algorithm inventors
