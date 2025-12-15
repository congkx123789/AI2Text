# Tài Liệu Chi Tiết: Công Nghệ và Triển Khai Model ASR

## 📋 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Kiến Trúc Model](#kiến-trúc-model)
3. [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
4. [Chi Tiết Triển Khai](#chi-tiết-triển-khai)
5. [Pipeline Xử Lý Dữ Liệu](#pipeline-xử-lý-dữ-liệu)
6. [Quy Trình Training](#quy-trình-training)
7. [Inference và Decoding](#inference-và-decoding)
8. [Tối Ưu Hóa Hiệu Năng](#tối-ưu-hóa-hiệu-năng)
9. [API và Deployment](#api-và-deployment)
10. [Đánh Giá và Metrics](#đánh-giá-và-metrics)

---

## 🎯 Tổng Quan

Model ASR (Automatic Speech Recognition) này được xây dựng từ đầu sử dụng kiến trúc **CRNN-CTC** (Convolutional Recurrent Neural Network với Connectionist Temporal Classification). Đây là một hệ thống end-to-end hoàn chỉnh từ xử lý audio đến nhận diện văn bản, được tối ưu hóa cho training trên hardware hiện đại (Ryzen 9 9900X + RTX 5060 Ti 16GB).

### Đặc Điểm Chính

- **Kiến trúc**: CRNN-CTC (CNN Encoder + Bidirectional LSTM + CTC Loss)
- **Input**: Audio waveform 16kHz mono
- **Features**: Log Mel-Spectrogram (80 mel bins)
- **Output**: Text transcription (character-level)
- **Loss Function**: CTC Loss
- **Decoder**: Greedy Decoding
- **Model Size**: ~4.1M parameters (~47MB)
- **Performance**: RTF = 0.0007 (1440x real-time)

---

## 🏗️ Kiến Trúc Model

### 1. CRNNCTC Architecture

Model được triển khai trong `src/asr/model.py` với cấu trúc 3 tầng chính:

```
Input Audio (16kHz mono)
    ↓
Log Mel-Spectrogram Extraction
    ↓
CNN Encoder (Feature Extraction)
    ↓
Bidirectional LSTM (Sequence Modeling)
    ↓
Linear Head (Classification)
    ↓
CTC Logits Output
```

### 2. Chi Tiết Các Layer

#### **CNN Encoder** (Feature Extraction)

```python
CNN Layers:
  - Conv1d(input=80, output=128, kernel=5, padding=2)
    → ReLU activation
  - Conv1d(input=128, output=128, kernel=5, padding=2)
    → ReLU activation
```

**Chức năng:**
- Nhận input là log mel-spectrogram với 80 mel bins
- Sử dụng 2 lớp Conv1d để trích xuất features từ frequency domain
- Kernel size = 5 với padding = 2 để giữ nguyên kích thước temporal
- Output: 128 feature channels

**Thông số:**
- Parameters: ~82,000
- Input shape: (B, T, 80) → (B, 128, T) → (B, T, 128)

#### **Bidirectional LSTM** (Sequence Modeling)

```python
LSTM Configuration:
  - Input size: 128 (từ CNN)
  - Hidden size: 256 per direction
  - Number of layers: 3
  - Bidirectional: True
  - Output size: 512 (256 forward + 256 backward)
```

**Chức năng:**
- Xử lý sequence temporal từ CNN features
- Bidirectional để capture context cả quá khứ và tương lai
- 3 layers để học representations phức tạp hơn
- Output: 512-dimensional features (256×2)

**Thông số:**
- Parameters: ~4,000,000
- Input shape: (B, T, 128) → (B, T, 512)

#### **Linear Head** (Classification)

```python
Linear Layer:
  - Input: 512 (từ LSTM)
  - Output: vocab_size (107 tokens)
```

**Chức năng:**
- Map từ sequence features sang vocabulary tokens
- Output logits cho mỗi token tại mỗi time step
- Vocab size: 107 (bao gồm `<blank>`, `<unk>`, và các ký tự)

**Thông số:**
- Parameters: ~55,000
- Output shape: (B, T, vocab_size)

### 3. Forward Pass

```python
def forward(self, x, x_lens):
    # x: (B, T, F) - Batch, Time, Features (80 mel bins)
    x = x.transpose(1, 2)    # (B, F, T) - Transpose cho Conv1d
    x = self.cnn(x)          # (B, C, T) - CNN feature extraction
    x = x.transpose(1, 2)    # (B, T, C) - Transpose lại cho LSTM
    x, _ = self.rnn(x)       # (B, T, 2H) - Bidirectional LSTM
    logits = self.head(x)    # (B, T, V) - Classification logits
    return logits.transpose(0, 1), x_lens  # (T, B, V) - CTC format
```

**Lưu ý:** Output được transpose về format (T, B, V) để phù hợp với CTC Loss của PyTorch.

---

## 🔧 Công Nghệ Sử Dụng

### 1. Deep Learning Framework

#### **PyTorch** (Core Framework)

**Version:** Latest stable (từ requirements.txt)

**Sử dụng:**
- **`torch.nn`**: Neural network layers (Conv1d, LSTM, Linear, CTCLoss)
- **`torch.optim`**: Optimizers (AdamW)
- **`torch.utils.data`**: Dataset và DataLoader
- **`torch.amp`**: Automatic Mixed Precision (AMP) training
- **`torch.backends.cudnn`**: CUDA optimizations

**Lý do chọn PyTorch:**
- Dynamic computation graph phù hợp cho research
- API trực quan và dễ debug
- Hỗ trợ tốt cho CTC Loss
- Mixed precision training tích hợp sẵn
- Ecosystem phong phú (torchaudio, torchvision)

#### **TorchAudio** (Audio Processing)

**Sử dụng:**
- **`torchaudio.load()`**: Load audio files (WAV, MP3, FLAC, etc.)
- **`torchaudio.functional.resample()`**: Resample audio về 16kHz
- **`torchaudio.transforms.MelSpectrogram`**: Extract mel-spectrogram

**Lý do:**
- Tích hợp tốt với PyTorch (tensor-based)
- Hỗ trợ nhiều format audio
- GPU-accelerated processing
- Consistent với PyTorch ecosystem

### 2. Audio Feature Extraction

#### **Log Mel-Spectrogram**

**Triển khai trong `src/asr/features.py`:**

```python
def wav_to_logmelspec(waveform, sr=16000, n_fft=400, hop=160, n_mels=80):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, 
        n_fft=n_fft,      # 400 samples = 25ms window
        hop_length=hop,   # 160 samples = 10ms hop
        n_mels=n_mels     # 80 mel bins
    )(waveform)
    logmel = torch.log(mel + 1e-6)  # Log transform với epsilon
    return logmel.transpose(0, 1)    # (frames, n_mels)
```

**Thông số:**
- **Sample Rate**: 16kHz (chuẩn cho ASR)
- **Window Size**: 400 samples (25ms) - cân bằng giữa temporal và frequency resolution
- **Hop Length**: 160 samples (10ms) - frame rate 100 FPS
- **Mel Bins**: 80 - đủ để capture frequency information
- **Normalization**: Log transform với epsilon=1e-6 để tránh log(0)

**Lý do chọn Mel-Spectrogram:**
- Mimic cách human ear nhận thức âm thanh (mel scale)
- Compact representation (80 bins vs 200+ FFT bins)
- Phù hợp cho speech recognition
- Standard trong ASR systems

### 3. Loss Function

#### **CTC Loss (Connectionist Temporal Classification)**

**Triển khai:** `torch.nn.CTCLoss`

```python
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
```

**Thông số:**
- **blank**: Index 0 (token đặc biệt cho CTC)
- **zero_infinity**: True - xử lý các trường hợp loss = infinity

**Cách hoạt động:**
1. CTC cho phép alignment tự động giữa audio frames và text tokens
2. Không cần forced alignment (khác với HMM-based systems)
3. Xử lý được các từ có độ dài khác nhau
4. Tự động merge duplicate tokens và loại bỏ blank tokens

**Lợi ích:**
- End-to-end training (không cần alignment labels)
- Xử lý được variable-length sequences
- Robust với timing variations
- Industry standard cho ASR

### 4. Optimizer

#### **AdamW Optimizer**

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001)
```

**Thông số:**
- **Learning Rate**: 0.001 (1e-3)
- **Weight Decay**: Mặc định (0.01)
- **Beta1, Beta2**: Mặc định (0.9, 0.999)

**Lý do chọn AdamW:**
- Cải thiện của Adam với weight decay đúng cách
- Tốt cho training deep networks
- Adaptive learning rate
- Stable convergence

### 5. Training Optimizations

#### **Automatic Mixed Precision (AMP)**

```python
from torch.amp import GradScaler, autocast

scaler = GradScaler('cuda', enabled=args.amp and device.type == "cuda")
autocast_ctx = lambda: autocast('cuda', dtype=torch.bfloat16)
```

**Sử dụng:**
- **bfloat16** precision thay vì float32
- Giảm memory usage ~50%
- Tăng tốc độ training ~1.5-2x
- Tự động detect GPU support

**Implementation:**
```python
with autocast_ctx():
    logits, out_lens = model(X, Xlen)
    log_probs = logits.log_softmax(dim=-1)
    loss = ctc(log_probs, Y, out_lens, Ylen)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### **Gradient Accumulation**

```python
gradient_accumulation_steps = 5
effective_batch_size = batch_size * gradient_accumulation_steps  # 32 * 5 = 160
```

**Cách hoạt động:**
1. Accumulate gradients qua N batches
2. Chỉ update weights sau N steps
3. Simulate batch size lớn hơn mà không cần memory lớn hơn

**Lợi ích:**
- Training với effective batch size lớn hơn
- Giảm memory requirements
- Stable gradients
- Better convergence

#### **Gradient Clipping**

```python
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

**Lý do:**
- Ngăn gradient explosion
- Ổn định training
- Đặc biệt quan trọng với RNN/LSTM

### 6. Data Loading Optimizations

#### **DataLoader Configuration**

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=12,              # Parallel data loading
    pin_memory=True,            # Faster GPU transfer
    persistent_workers=True,     # Keep workers alive
    prefetch_factor=2          # Prefetch batches
)
```

**Tối ưu hóa:**
- **num_workers=12**: Tận dụng 24 threads của Ryzen 9 9900X
- **pin_memory=True**: Copy data từ CPU → GPU nhanh hơn
- **persistent_workers=True**: Không recreate workers mỗi epoch
- **prefetch_factor=2**: Prefetch 2 batches để giảm GPU starvation

#### **Batch Collation**

```python
def collate_batch(batch):
    xs, ys = zip(*batch)
    x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return X, x_lens, Y, y_lens
```

**Chức năng:**
- Pad sequences về cùng độ dài trong batch
- Lưu original lengths cho CTC loss
- Batch-first format cho efficiency

### 7. Memory Management

#### **CUDA Cache Management**

```python
if args.empty_cache and device.type == "cuda":
    torch.cuda.empty_cache()  # Sau mỗi batch
```

**Lý do:**
- Giải phóng fragmented memory
- Quan trọng với limited VRAM (16GB)
- Giảm OOM errors

#### **Environment Variables**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Lợi ích:**
- Giảm memory fragmentation
- Tăng memory efficiency

---

## 💻 Chi Tiết Triển Khai

### 1. Dataset Implementation

#### **ManifestDataset** (`src/asr/dataset.py`)

**Cấu trúc:**
```python
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, vocab_path, audio_root=None, 
                 timestamps_path=None, trim_to_segments=False):
        # Load manifest.csv
        # Load vocabulary
        # Load timestamps (optional)
    
    def __getitem__(self, idx):
        # Load audio file
        # Preprocess audio (resample, mono)
        # Optional: trim using timestamps
        # Extract features (log mel-spectrogram)
        # Encode transcript to token indices
        return features, token_indices
```

**Tính năng:**
- Đọc từ CSV manifest (flexible và scalable)
- Hỗ trợ timestamps.json cho audio trimming
- Robust audio loading (torchaudio + soundfile fallback)
- Automatic preprocessing (resample, mono conversion)

**Audio Preprocessing:**
```python
def ensure_mono16k(waveform, sr):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)  # Stereo → Mono
    return waveform, 16000
```

### 2. Training Loop (`src/asr/train_ctc.py`)

#### **Cấu trúc Training Loop**

```python
for epoch in range(start_epoch, epochs + 1):
    model.train()
    for batch_idx, (X, Xlen, Y, Ylen) in enumerate(dataloader):
        # 1. Move to device
        X, Xlen, Y, Ylen = X.to(device), Xlen.to(device), ...
        
        # 2. Forward pass với AMP
        with autocast_ctx():
            logits, out_lens = model(X, Xlen)
            log_probs = logits.log_softmax(dim=-1)
            loss = ctc(log_probs, Y, out_lens, Ylen) / gradient_accumulation_steps
        
        # 3. Backward pass
        scaler.scale(loss).backward()
        
        # 4. Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 5. Logging
        if batch_idx % log_interval == 0:
            log_metrics(...)
    
    # 6. Save checkpoint
    save_checkpoint(epoch, model, optimizer, scaler, loss)
```

#### **Checkpointing**

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # AMP scaler
    'loss': avg_loss
}
torch.save(checkpoint, checkpoint_path)
```

**Tính năng:**
- Save sau mỗi epoch
- Resume training từ checkpoint
- Lưu cả optimizer state và scaler state

### 3. Decoding (`src/asr/decode.py`)

#### **Greedy Decoding**

```python
def greedy_decode(logits, itos):
    # logits: (T, B, V) - Time, Batch, Vocabulary
    pred = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)
    texts = []
    for seq in pred:
        prev = None
        out = []
        for idx in seq.tolist():
            if idx != 0 and idx != prev:  # Skip blank (0) và duplicates
                out.append(itos[idx])
            prev = idx
        texts.append("".join(out))
    return texts
```

**Cách hoạt động:**
1. Lấy argmax tại mỗi time step
2. Loại bỏ blank tokens (index 0)
3. Loại bỏ duplicate tokens liên tiếp
4. Map indices về characters

**Lưu ý:**
- Đơn giản và nhanh
- Không tối ưu như beam search
- Có thể cải thiện bằng language model

### 4. Tokenization (`src/asr/tokenizer.py`)

#### **Character-level Vocabulary**

```python
def build_char_vocab(transcripts, extra_tokens=("<blank>", "<unk>")):
    chars = set()
    for t in transcripts:
        chars.update(list(t.lower().strip()))
    vocab = list(extra_tokens) + sorted(chars)
    return {"vocab": vocab, "stoi": dict, "itos": dict}
```

**Cấu trúc Vocabulary:**
- Index 0: `<blank>` (CTC blank token)
- Index 1: `<unk>` (unknown character)
- Index 2+: Các ký tự (a-z, 0-9, space, punctuation, Vietnamese characters)

**Hiện tại:** 107 tokens (bao gồm Vietnamese diacritics)

---

## 📊 Pipeline Xử Lý Dữ Liệu

### 1. Data Preparation

```
Raw Audio Files (various formats)
    ↓
Audio Loading (torchaudio/soundfile)
    ↓
Resample to 16kHz
    ↓
Convert to Mono
    ↓
Feature Extraction (Log Mel-Spectrogram)
    ↓
Text Transcription
    ↓
Character-level Tokenization
    ↓
Dataset Ready
```

### 2. Training Data Flow

```
ManifestDataset
    ↓
DataLoader (batch_size=32, num_workers=12)
    ↓
Batch Collation (padding sequences)
    ↓
GPU Transfer (pin_memory=True)
    ↓
Model Forward Pass
    ↓
CTC Loss Calculation
    ↓
Backward Pass (gradient accumulation)
    ↓
Optimizer Step
```

### 3. Inference Data Flow

```
Audio File
    ↓
Preprocessing (resample, mono)
    ↓
Log Mel-Spectrogram
    ↓
Model Inference
    ↓
CTC Logits
    ↓
Greedy Decoding
    ↓
Text Output
```

---

## 🚀 Quy Trình Training

### 1. Configuration

**File:** `config/train_merged.json`

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
  "num_workers": 12,
  "epochs": 20,
  "log_interval": 20,
  "lr": 0.001
}
```

### 2. Training Command

```bash
python scripts/train_from_config.py config/train_merged.json
```

### 3. Monitoring

#### **TensorBoard**

```bash
tensorboard --logdir=runs
```

**Metrics logged:**
- `Loss/Train_step`: Loss mỗi log_interval batches
- `Loss/Train_epoch`: Average loss mỗi epoch

#### **Progress Monitoring**

```bash
python scripts/training_progress_bar.py
```

**Features:**
- Real-time progress bars
- GPU memory usage
- GPU utilization
- Estimated time remaining

### 4. Checkpoint Management

- **Location**: `data/results/checkpoints/checkpoint_epoch_X.pt`
- **Frequency**: Sau mỗi epoch
- **Resume**: Thêm `"resume": "path/to/checkpoint.pt"` vào config

---

## 🔍 Inference và Decoding

### 1. Single Audio Inference

```python
# Load model
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# Load audio
wav, sr = torchaudio.load(audio_path)
wav, sr = ensure_mono16k(wav, sr)

# Extract features
feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
    text = greedy_decode(logits.cpu(), itos)[0]
```

### 2. Batch Inference

```python
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch)

for X, Xlen, Y, Ylen in dataloader:
    X, Xlen = X.to(device), Xlen.to(device)
    with torch.no_grad():
        logits, out_lens = model(X, Xlen)
        texts = greedy_decode(logits.cpu(), itos)
```

### 3. Performance Metrics

**RTF (Real-Time Factor):**
```python
rtf = total_inference_time / total_audio_duration
# RTF < 1: Nhanh hơn real-time
# RTF = 1: Bằng real-time
# RTF > 1: Chậm hơn real-time
```

**Current Performance:**
- RTF: 0.0007 (1440x real-time) ✅
- Total audio: 227,287 seconds (~63 hours)
- Inference time: 157.74 seconds (~2.6 minutes)

---

## ⚡ Tối Ưu Hóa Hiệu Năng

### 1. Hardware Optimizations

#### **CUDA Optimizations**

```python
if device.type == "cuda":
    cudnn.benchmark = True  # Optimize cuDNN operations
```

**Lợi ích:**
- Auto-tune cuDNN cho hardware cụ thể
- Tăng tốc độ convolution và LSTM operations

#### **Memory Optimizations**

- **AMP (bfloat16)**: Giảm memory ~50%
- **Gradient Accumulation**: Simulate large batch với memory nhỏ
- **Empty Cache**: Giải phóng fragmented memory
- **Pin Memory**: Faster CPU→GPU transfer

### 2. Data Loading Optimizations

- **Multiple Workers**: Parallel data loading (12 workers)
- **Persistent Workers**: Không recreate workers mỗi epoch
- **Prefetch Factor**: Prefetch batches để giảm GPU starvation
- **Pin Memory**: Faster GPU transfer

### 3. Model Optimizations

- **Efficient Architecture**: CRNN thay vì Transformer (nhẹ hơn)
- **Bidirectional LSTM**: Capture context tốt với parameters hợp lý
- **Compact Model**: 4.1M parameters (dễ deploy)

### 4. Training Optimizations

- **Mixed Precision**: 1.5-2x speedup
- **Gradient Accumulation**: Stable training với large effective batch
- **Gradient Clipping**: Stable convergence
- **Checkpointing**: Resume training dễ dàng

---

## 🌐 API và Deployment

### 1. FastAPI Server (`src/asr/api.py`)

#### **Framework: FastAPI**

**Lý do chọn:**
- Modern Python web framework
- Automatic API documentation (Swagger)
- Type hints support
- High performance (async support)
- Easy integration với PyTorch

#### **Endpoints**

**GET `/`**: API information
```json
{
  "message": "ASR CTC API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

**GET `/health`**: Health check
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "vocab_size": 107
}
```

**GET `/model/info`**: Model information
```json
{
  "model_path": "...",
  "device": "cuda",
  "vocab_size": 107,
  "total_parameters": 4132715,
  "trainable_parameters": 4132715,
  "model_type": "CRNNCTC"
}
```

**POST `/transcribe`**: Transcribe audio
```python
# Input: multipart/form-data với audio file
# Output: {"text": "...", "duration": 1.23}
```

#### **CORS Configuration**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. API Usage

#### **Start Server**

```bash
python scripts/serve_asr.py --host 0.0.0.0 --port 8001
```

#### **Test API**

```bash
# Health check
curl http://localhost:8001/health

# Transcribe
curl -X POST "http://localhost:8001/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

#### **Swagger UI**

Truy cập: `http://localhost:8001/docs`

### 3. Model Loading

```python
# Load vocabulary
VOCAB = json.load(open(VOCAB_PATH))
ITOS = {i: c for i, c in enumerate(VOCAB)}

# Load model
MODEL = CRNNCTC(n_mels=80, vocab_size=len(VOCAB))
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
MODEL.to(device).eval()
```

**Lưu ý:**
- Model được load một lần khi server start
- Inference với `torch.no_grad()` để tối ưu
- Auto preprocessing audio (resample, mono)

---

## 📈 Đánh Giá và Metrics

### 1. Evaluation Metrics

#### **Accuracy Metrics**

**WER (Word Error Rate):**
```python
wer = (substitutions + insertions + deletions) / total_words
```

**CER (Character Error Rate):**
```python
cer = (char_substitutions + char_insertions + char_deletions) / total_chars
```

**SER (Sentence Error Rate):**
```python
ser = sentences_with_errors / total_sentences
```

**Exact Match Accuracy:**
```python
exact_match = perfect_predictions / total_samples
```

#### **Performance Metrics**

**RTF (Real-Time Factor):**
```python
rtf = total_inference_time / total_audio_duration
```

**Current Results (Epoch 12):**
- WER: 47.37%
- CER: 21.04%
- SER: 99.61%
- Exact Match: 0.41%
- RTF: 0.0007 (1440x real-time) ✅

### 2. Evaluation Tools

#### **Full Dataset Evaluation**

```bash
python scripts/test_full_dataset.py \
  --checkpoint data/results/checkpoints/checkpoint_epoch_12.pt \
  --vocab data/processed/vocab.json \
  --test_manifest data/processed/test/manifest.csv \
  --audio_root data/processed/test \
  --batch_size 16 \
  --output experiments/reports/all_predictions_epoch12.json
```

**Output:**
- `all_predictions_epoch12.json`: Tất cả predictions
- `metrics.json`: Metrics summary

#### **Visualization**

```bash
# Training report
python scripts/visualize_training_report.py

# Log scale metrics
python scripts/plot_log_scale_metrics.py

# Combined metrics
python scripts/plot_combined_log_metrics.py

# Evaluation report
python scripts/generate_evaluation_report.py
```

### 3. Metrics Calculation

**Sử dụng thư viện `jiwer`:**

```python
from jiwer import wer, cer

wer_score = wer(ground_truths, predictions)
cer_score = cer(ground_truths, predictions)
```

**Custom metrics:**
- Sentence Error Rate (SER)
- Exact Match Accuracy
- Word-level Accuracy
- RTF calculation

---

## 📦 Dependencies và Requirements

### Core Dependencies

```txt
torch                    # PyTorch deep learning framework
torchaudio              # Audio processing
numpy                   # Numerical computing
librosa                 # Audio analysis (optional)
jiwer                   # WER/CER calculation
fastapi                 # API framework
uvicorn                 # ASGI server
python-multipart        # File upload support
pydantic>=2             # Data validation
requests                # HTTP client
```

### Optional Dependencies

```txt
tensorboard             # Training visualization
soundfile               # Audio loading fallback
matplotlib              # Plotting
tqdm                    # Progress bars
```

### System Requirements

- **Python**: >= 3.9
- **CUDA**: >= 11.0 (cho GPU training)
- **RAM**: 32GB+ (khuyến nghị)
- **VRAM**: 16GB+ (cho training với batch_size=32)

---

## 🎓 Kiến Thức Kỹ Thuật

### 1. CTC Algorithm

**Connectionist Temporal Classification (CTC)** là một loss function cho phép:
- Training end-to-end mà không cần alignment labels
- Xử lý variable-length sequences
- Tự động merge duplicate tokens
- Loại bỏ blank tokens trong decoding

**CTC Alignment:**
- Input: Audio frames (T frames)
- Output: Text tokens (U tokens, U < T)
- CTC tìm tất cả possible alignments và marginalize

### 2. Mel-Spectrogram

**Mel Scale:**
- Mimic human auditory perception
- Logarithmic scale ở high frequencies
- Linear scale ở low frequencies

**Mel-Spectrogram:**
- Convert frequency domain sang mel scale
- Compact representation (80 bins vs 200+ FFT bins)
- Standard cho speech recognition

### 3. Bidirectional LSTM

**LSTM (Long Short-Term Memory):**
- Xử lý long-term dependencies
- Forget gate để loại bỏ irrelevant information
- Input/Output gates để control information flow

**Bidirectional:**
- Process sequence cả forward và backward
- Capture context từ cả quá khứ và tương lai
- Output: Concatenate forward và backward hidden states

### 4. Mixed Precision Training

**bfloat16:**
- 16-bit floating point với same exponent range như float32
- Giảm memory usage ~50%
- Tăng tốc độ training ~1.5-2x
- Stable training (không như fp16)

**Gradient Scaling:**
- Scale gradients để tránh underflow
- Unscale trước khi optimizer step
- Update scale factor dynamically

---

## 🔮 Hướng Phát Triển

### 1. Model Improvements

- **Beam Search Decoding**: Thay greedy decoding
- **Language Model Integration**: N-gram hoặc neural LM
- **Attention Mechanism**: Thêm attention layers
- **Transformer Architecture**: Thử Transformer thay LSTM

### 2. Training Improvements

- **Data Augmentation**: Noise, speed variation, pitch shift
- **Learning Rate Scheduling**: Cosine annealing, warmup
- **Regularization**: Dropout, weight decay tuning
- **More Training Data**: Tăng dataset size

### 3. Deployment Improvements

- **Model Quantization**: INT8 quantization
- **ONNX Export**: Deploy trên nhiều platforms
- **TensorRT**: GPU acceleration
- **Mobile Deployment**: CoreML, TensorFlow Lite

### 4. Evaluation Improvements

- **Per-domain Metrics**: WER theo domain (English/Vietnamese)
- **Error Analysis**: Phân tích loại lỗi
- **Confidence Scores**: Thêm confidence cho predictions
- **Ablation Studies**: Phân tích contribution của từng component

---

## 📚 Tài Liệu Tham Khảo

### Papers

1. **CTC Paper**: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" (Graves et al., 2006)
2. **Mel-Spectrogram**: "Mel Frequency Cepstral Coefficients for Music Modeling" (Tzanetakis et al., 2001)

### Libraries

- **PyTorch Documentation**: https://pytorch.org/docs/
- **TorchAudio Documentation**: https://pytorch.org/audio/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **CTC Loss**: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

### Datasets

- **LibriSpeech**: https://www.openslr.org/12/
- **VietSpeech**: Vietnamese speech dataset

---

## 📝 Kết Luận

Model ASR này được xây dựng từ đầu với các công nghệ hiện đại:

✅ **Deep Learning**: PyTorch với CRNN-CTC architecture
✅ **Audio Processing**: TorchAudio với Mel-Spectrogram features
✅ **Training Optimizations**: AMP, gradient accumulation, clipping
✅ **Data Loading**: Parallel loading với multiple workers
✅ **API Deployment**: FastAPI với async support
✅ **Evaluation**: Comprehensive metrics (WER, CER, SER, RTF)

**Điểm mạnh:**
- Model nhỏ gọn (4.1M parameters)
- Tốc độ xử lý rất nhanh (1440x real-time)
- End-to-end pipeline hoàn chỉnh
- Dễ deploy và scale

**Cần cải thiện:**
- WER còn cao (47.37%)
- Cần thêm training data
- Có thể thử architecture phức tạp hơn

---

**Tác giả**: [Tên của bạn]
**Ngày tạo**: 2025-01-XX
**Version**: 1.0.0

