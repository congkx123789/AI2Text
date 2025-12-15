# Công Nghệ và Triển Khai Models - AI2Text System

Tài liệu chi tiết về các công nghệ được sử dụng và cách triển khai các models trong hệ thống AI2Text.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Model 1: AI2Text Transformer ASR](#model-1-ai2text-transformer-asr)
- [Model 2: Whisper Fine-tuned (CTranslate2)](#model-2-whisper-fine-tuned-ctranslate2)
- [Model 3: Whisper Original (HuggingFace)](#model-3-whisper-original-huggingface)
- [Model 4: CRNN-CTC ASR](#model-4-crnn-ctc-asr)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Decoding Algorithms](#decoding-algorithms)
- [Training Technologies](#training-technologies)
- [Deployment Architecture](#deployment-architecture)

---

## 🎯 Tổng Quan

Hệ thống AI2Text sử dụng **4 kiến trúc model khác nhau** để đáp ứng các nhu cầu khác nhau:

| Model | Kiến Trúc | Công Nghệ Chính | Use Case |
|-------|-----------|----------------|----------|
| **AI2Text** | Transformer Seq2Seq | PyTorch, Attention, CTC | Custom training, Vietnamese |
| **Whisper Fine-tuned** | Whisper (CT2) | CTranslate2, faster-whisper | Fast inference, EN+VI |
| **Whisper Original** | Whisper (HF) | HuggingFace Transformers | General purpose |
| **CTC ASR** | CRNN-CTC | PyTorch, CTC Loss | Lightweight, CPU-friendly |

---

## 🏗️ Model 1: AI2Text Transformer ASR

### Công Nghệ Sử Dụng

#### 1. **Kiến Trúc Transformer Seq2Seq**

```
Input Audio (16kHz mono)
    ↓
Log Mel-Spectrogram (80 mel bins)
    ↓
ConvSubsampling (2x hoặc 4x)
    ↓
Transformer Encoder (14 layers)
    ├─ Multi-Head Attention (8 heads)
    ├─ RMSNorm (thay vì LayerNorm)
    ├─ Rotary Positional Embedding (RoPE)
    ├─ SiLU Activation
    └─ Feed-Forward Network (2048 dim)
    ↓
Transformer Decoder (6 layers)
    ├─ Self-Attention
    ├─ Cross-Attention (encoder-decoder)
    ├─ RMSNorm
    └─ Feed-Forward Network
    ↓
Output: Token Logits (3500 vocab)
```

#### 2. **Các Công Nghệ Hiện Đại**

**a) RMSNorm (Root Mean Square Layer Normalization)**
- Thay thế LayerNorm truyền thống
- Tốc độ nhanh hơn ~10-15%
- Numerical stability tốt hơn
- Được sử dụng trong LLaMA, GPT-3

**b) Rotary Positional Embedding (RoPE)**
- Encoding vị trí tốt hơn so với positional encoding cố định
- Hỗ trợ sequence length dài hơn
- Tự nhiên hơn cho attention mechanism

**c) SiLU Activation (Swish)**
- Thay thế ReLU
- Smooth gradient, tốt cho training
- Performance tốt hơn ReLU trong nhiều tasks

**d) Scaled Dot-Product Attention (SDPA)**
- Sử dụng `F.scaled_dot_product_attention` của PyTorch
- Flash Attention trên GPU (RTX 5060 Ti)
- Tốc độ nhanh hơn 2-4x so với implementation thủ công
- Numerical stability tốt hơn

**e) Gradient Checkpointing**
- Giảm memory usage khi training
- Trade-off: memory vs computation
- Quan trọng cho RTX 5060 Ti 16GB

#### 3. **Tokenization: SentencePiece BPE**

- **Vocab size**: 3500 tokens
- **Format**: SentencePiece model file
- **Hỗ trợ**: Vietnamese + English
- **Lợi ích**:
  - Subword tokenization
  - Xử lý OOV (Out-of-Vocabulary) tốt
  - Compact representation

#### 4. **Training Technologies**

**a) Mixed Precision Training (AMP)**
```python
# Sử dụng bfloat16 (tốt hơn float16)
amp_dtype = torch.bfloat16  # RTX 5060 Ti hỗ trợ
scaler = GradScaler()
```

**b) Gradient Accumulation**
- Effective batch size = batch_size × gradient_accumulation_steps
- Ví dụ: 64 × 4 = 256 effective batch size
- Cho phép training với batch size lớn trên GPU nhỏ

**c) Learning Rate Scheduling**
- Warmup: 3% của total epochs
- Cosine annealing sau warmup
- Adaptive learning rate

**d) Curriculum Learning**
- Bắt đầu với câu ngắn
- Tăng dần độ khó
- Cải thiện convergence

**e) Auto-Rollback**
- Tự động phát hiện model collapse
- Rollback về checkpoint tốt nhất
- Tránh lãng phí thời gian training

#### 5. **Model Specifications**

```
Total Parameters: ~30M
Encoder Layers: 14
Decoder Layers: 6
Attention Heads: 8
Model Dimension: 256
FFN Dimension: 2048
Vocab Size: 3500
Input: 80 mel spectrogram bins
Output: Token sequence (autoregressive)
```

### Triển Khai

#### 1. **Model Loading**

```python
from models.asr_base import ASRModel
from api.asr_service import ASRService

# Load model từ checkpoint
service = ASRService(
    checkpoint_path="checkpoints/best_model.pt",
    device="cuda"
)
```

#### 2. **Inference Pipeline**

```python
# 1. Audio preprocessing
audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
features = audio_processor.extract_mel_spectrogram(audio_data)

# 2. Model inference
logits = model(features)

# 3. Decoding (beam search hoặc greedy)
decoder = BeamSearchDecoder(vocab_size=3500, beam_width=5)
result = decoder.decode(logits)

# 4. Tokenization
text = tokenizer.decode(result['text'])
```

#### 3. **API Endpoint**

```python
# FastAPI endpoint
@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile,
    model_name: str = "best_model",
    use_beam_search: bool = True,
    beam_width: int = 5
):
    # Auto-load model if not cached
    if model_name not in models_cache:
        model = load_model(f"checkpoints/{model_name}.pt")
        models_cache[model_name] = model
    
    # Transcribe
    result = service.transcribe(audio_path, return_timestamps=False)
    return result
```

#### 4. **Optimization cho RTX 5060 Ti**

- **torch.compile()**: JIT compilation cho inference nhanh hơn
- **Mixed precision**: bfloat16 cho training và inference
- **Gradient checkpointing**: Giảm memory usage
- **Efficient batching**: Dynamic batching với bucketing

---

## 🎤 Model 2: Whisper Fine-tuned (CTranslate2)

### Công Nghệ Sử Dụng

#### 1. **CTranslate2 Framework**

- **Framework**: CTranslate2 (C++ backend)
- **Format**: Optimized binary format (model.bin)
- **Tốc độ**: Nhanh hơn 5-10x so với HuggingFace
- **Memory**: Tiết kiệm memory hơn 2-3x

#### 2. **faster-whisper Library**

```python
from faster_whisper import WhisperModel

# Load CTranslate2 model
model = WhisperModel(
    "models/final/whisper-vi-en-ct2",
    device="cuda",
    compute_type="float16"  # hoặc int8 cho CPU
)
```

#### 3. **Model Architecture**

- **Base**: OpenAI Whisper Small
- **Fine-tuned**: Vietnamese + English
- **Format**: CTranslate2 (converted từ HuggingFace)
- **Size**: ~244M parameters (Whisper Small)

#### 4. **Fine-tuning Process**

**a) Data Preparation**
- Vietnamese + English mixed dataset
- Audio preprocessing: 16kHz mono
- Text normalization

**b) Training**
- LoRA fine-tuning (Parameter-Efficient)
- Hoặc full fine-tuning
- Multi-task learning (transcribe + translate)

**c) Conversion to CTranslate2**
```bash
# Convert HuggingFace model to CTranslate2
ct2-transformers-converter \
    --model models/finetuned/whisper-mixed \
    --output_dir models/final/whisper-vi-en-ct2 \
    --quantization float16
```

#### 5. **Compute Types**

- **float16**: GPU inference (nhanh, độ chính xác tốt)
- **int8**: CPU inference (tiết kiệm memory)
- **int8_float16**: Hybrid (encoder int8, decoder float16)

### Triển Khai

#### 1. **Model Loading với Caching**

```python
_MODEL_CACHE = {}

def _get_model(size, device, compute):
    key = f"ct2:{size}:{device}:{compute}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            str(size),
            device=device,
            compute_type=compute
        )
    return _MODEL_CACHE[key]
```

#### 2. **Inference**

```python
# Transcribe với VAD (Voice Activity Detection)
segments, info = model.transcribe(
    audio_path,
    language="vi",  # hoặc None cho auto-detect
    vad_filter=True,
    beam_size=5
)

# Collect results
text = " ".join([seg.text for seg in segments])
```

#### 3. **API Integration**

```python
@app.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile,
    language: Optional[str] = None,
    model_size: str = "small"
):
    # Auto-preprocess audio
    tmp_path = auto_preprocess_audio(uploaded_file)
    
    # Transcribe
    result = transcribe(
        tmp_path,
        size="/path/to/whisper-vi-en-ct2",
        lang=language,
        device="auto",
        compute="float16"
    )
    
    return TranscribeResponse(**result)
```

#### 4. **Performance Optimization**

- **Model caching**: Load một lần, dùng nhiều lần
- **Batch processing**: Xử lý nhiều files cùng lúc
- **VAD filtering**: Bỏ qua silence, tăng tốc độ
- **Beam search**: Cân bằng tốc độ và độ chính xác

---

## 🔊 Model 3: Whisper Original (HuggingFace)

### Công Nghệ Sử Dụng

#### 1. **HuggingFace Transformers**

- **Library**: transformers
- **Model**: openai/whisper-small
- **Format**: HuggingFace (PyTorch)
- **Size**: ~244M parameters

#### 2. **Model Architecture**

```
Audio Input (16kHz)
    ↓
Feature Extractor (Mel Spectrogram)
    ↓
Encoder (Transformer)
    ├─ Multi-Head Attention
    ├─ Feed-Forward Network
    └─ Layer Normalization
    ↓
Decoder (Transformer)
    ├─ Self-Attention
    ├─ Cross-Attention
    └─ Feed-Forward Network
    ↓
Text Output
```

#### 3. **Auto-detect Format**

Code tự động phát hiện format:
- CTranslate2: Có `model.bin`
- HuggingFace: Có `config.json` + `model.safetensors`

### Triển Khai

#### 1. **Model Loading**

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load HuggingFace model
processor = WhisperProcessor.from_pretrained("models/base/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("models/base/whisper-small")
model.eval()

if device == "cuda":
    model = model.cuda()
```

#### 2. **Inference**

```python
# Process audio
input_features = processor.feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features.to(model.device)

# Generate
with torch.no_grad():
    generated_ids = model.generate(
        input_features,
        max_length=448,
        language="vi",
        task="transcribe",
        num_beams=5
    )

# Decode
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

---

## 🎯 Model 4: CRNN-CTC ASR

### Công Nghệ Sử Dụng

#### 1. **Kiến Trúc CRNN-CTC**

```
Input Audio (16kHz mono)
    ↓
Log Mel-Spectrogram (80 bins)
    ↓
CNN Encoder (2 layers)
    ├─ Conv1d(80 → 128, kernel=5)
    ├─ ReLU
    ├─ Conv1d(128 → 128, kernel=5)
    └─ ReLU
    ↓
Bidirectional LSTM (3 layers)
    ├─ Hidden size: 256 per direction
    ├─ Output: 512 (256×2)
    └─ 3 layers
    ↓
Linear Head (512 → vocab_size)
    ↓
CTC Logits (T, B, V)
    ↓
CTC Decoding (Greedy)
    ↓
Text Output
```

#### 2. **CTC Loss**

- **Connectionist Temporal Classification**
- Không cần alignment giữa audio và text
- Tự động học alignment trong quá trình training
- Phù hợp cho sequence-to-sequence tasks

#### 3. **Model Specifications**

```
Total Parameters: ~4.1M
CNN Channels: 128
RNN Hidden: 256 (bidirectional → 512)
RNN Layers: 3
Vocab Size: 107 tokens
Input: 80 mel spectrogram bins
Output: CTC logits
```

#### 4. **Training**

- **Loss**: CTC Loss
- **Optimizer**: Adam
- **Decoder**: Greedy decoding (không dùng beam search)
- **Dataset**: LibriSpeech (EN) + VietSpeech (VI)

### Triển Khai

#### 1. **Model Definition**

```python
class CRNNCTC(nn.Module):
    def __init__(self, n_mels=80, vocab_size=107):
        super().__init__()
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        # Linear Head
        self.head = nn.Linear(512, vocab_size)
```

#### 2. **Inference**

```python
# Extract features
wav, sr = torchaudio.load(audio_path)
wav, sr = ensure_mono16k(wav, sr)
feats = wav_to_logmelspec(wav, sr).unsqueeze(0)

# Forward pass
logits, lens = model(feats, torch.tensor([feats.shape[1]]))

# CTC Decoding (Greedy)
text = greedy_decode(logits.cpu(), vocab_dict)[0]
```

#### 3. **API Endpoint**

```python
@app.post("/transcribe")
async def transcribe(file: UploadFile):
    # Preprocess audio
    tmp_path = save_uploaded_file(file)
    tmp_path = auto_preprocess_audio(tmp_path)
    
    # Load audio
    wav, sr = torchaudio.load(tmp_path)
    wav, sr = ensure_mono16k(wav, sr)
    feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits, lens = MODEL(feats, torch.tensor([feats.shape[1]], device=device))
    
    # Decode
    text = greedy_decode(logits.cpu(), ITOS)[0]
    
    return {"text": text}
```

---

## 🎵 Audio Processing Pipeline

### Công Nghệ Sử Dụng

#### 1. **Automatic Preprocessing**

```python
def auto_preprocess_audio(
    audio_path: str,
    sample_rate: int = 16000,
    model_type: str = 'ai-llm'
) -> str:
    """
    Tự động preprocessing audio cho các model khác nhau.
    
    Steps:
    1. Load audio (hỗ trợ nhiều format: WAV, MP3, FLAC, etc.)
    2. Resample về 16kHz
    3. Convert sang mono
    4. Noise reduction (optional)
    5. Normalize
    6. Save thành WAV format (PCM 16-bit)
    """
```

#### 2. **Audio Enhancement**

- **Noise Reduction**: Spectral subtraction, Wiener filtering
- **Speech Enhancement**: Spectral gating, adaptive filtering
- **Normalization**: Peak normalization, RMS normalization
- **Filtering**: High-pass, low-pass filters

#### 3. **Feature Extraction**

**a) Mel Spectrogram**
```python
# Parameters
sample_rate = 16000
n_mels = 80
n_fft = 400
hop_length = 160
win_length = 400

# Extract
mel_spec = librosa.feature.melspectrogram(
    audio,
    sr=sample_rate,
    n_mels=n_mels,
    n_fft=n_fft,
    hop_length=hop_length
)
log_mel = np.log(mel_spec + 1e-8)
```

**b) Audio Augmentation (Training)**
- Time stretching
- Pitch shifting
- Speed perturbation
- Volume perturbation
- Background noise addition

---

## 🔍 Decoding Algorithms

### 1. **Greedy Decoding**

- Đơn giản, nhanh
- Chọn token có probability cao nhất tại mỗi time step
- Phù hợp cho CTC models

```python
def greedy_decode(logits, vocab):
    # logits: (T, B, V)
    predictions = torch.argmax(logits, dim=-1)  # (T, B)
    # CTC decoding: remove blanks, merge duplicates
    text = ctc_decode(predictions, vocab)
    return text
```

### 2. **Beam Search Decoding**

- Tìm kiếm nhiều hypotheses
- Cân bằng tốc độ và độ chính xác
- Phù hợp cho Transformer models

```python
class BeamSearchDecoder:
    def __init__(self, beam_width=5, length_penalty=0.6):
        self.beam_width = beam_width
        self.length_penalty = length_penalty
    
    def decode(self, logits):
        # Maintain beam_width hypotheses
        # Expand each hypothesis
        # Prune to top beam_width
        # Apply length penalty
        return best_hypothesis
```

### 3. **Language Model Decoding (KenLM)**

- Kết hợp với n-gram language model
- Cải thiện độ chính xác
- Tốc độ chậm hơn

```python
class LMBeamSearchDecoder:
    def __init__(self, vocab, lm_path):
        self.lm = kenlm.Model(lm_path)
    
    def decode(self, logits):
        # Combine acoustic model scores với LM scores
        # Beam search với combined scores
        return best_hypothesis
```

---

## 🎓 Training Technologies

### 1. **Mixed Precision Training**

```python
# Automatic Mixed Precision (AMP)
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    logits = model(features)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Lợi ích**:
- Giảm memory usage ~50%
- Tăng tốc độ training ~1.5-2x
- Numerical stability với bfloat16

### 2. **Gradient Accumulation**

```python
# Accumulate gradients over multiple batches
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Lợi ích**:
- Effective batch size lớn hơn
- Phù hợp cho GPU nhỏ

### 3. **Learning Rate Scheduling**

```python
# Warmup + Cosine Annealing
warmup_steps = total_steps * 0.03  # 3% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### 4. **Curriculum Learning**

- Bắt đầu với câu ngắn, dễ
- Tăng dần độ khó
- Cải thiện convergence

### 5. **Auto-Rollback**

- Phát hiện model collapse (loss tăng đột ngột)
- Tự động rollback về checkpoint tốt nhất
- Tránh lãng phí thời gian

---

## 🚀 Deployment Architecture

### 1. **API Server Architecture**

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  - CORS enabled                          │
│  - File upload support                   │
│  - Error handling                        │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼──────┐
│ Model Cache │  │ Audio Proc  │
│ (Lazy Load) │  │ (Preprocess)│
└──────┬──────┘  └─────┬───────┘
       │                │
┌──────▼────────────────▼──────┐
│      Model Inference           │
│  - GPU/CPU auto-detect         │
│  - Batch processing            │
└──────┬─────────────────────────┘
       │
┌──────▼────────┐
│   Decoding    │
│  - Beam Search│
│  - Greedy     │
└──────┬────────┘
       │
┌──────▼────────┐
│  Response     │
│  - Text       │
│  - Confidence │
│  - Timestamps │
└───────────────┘
```

### 2. **Model Caching Strategy**

```python
# Lazy loading: Load model khi cần
models_cache = {}

def get_model(model_name):
    if model_name not in models_cache:
        models_cache[model_name] = load_model(model_name)
    return models_cache[model_name]
```

### 3. **Multi-Model Support**

- Mỗi model chạy trên port riêng
- Frontend có thể chọn model
- Models có thể chạy song song

### 4. **Performance Optimization**

**a) GPU Optimization**
- Mixed precision inference
- Batch processing
- Model quantization (int8)

**b) CPU Optimization**
- Multi-threading
- Efficient data loading
- Model quantization

**c) Memory Optimization**
- Model caching
- Gradient checkpointing
- Efficient batching

---

## 📊 So Sánh Công Nghệ

| Aspect | AI2Text | Whisper CT2 | Whisper HF | CTC ASR |
|--------|---------|-------------|------------|---------|
| **Framework** | PyTorch | CTranslate2 | Transformers | PyTorch |
| **Parameters** | ~30M | ~244M | ~244M | ~4.1M |
| **Speed** | Medium | Very Fast | Fast | Fast |
| **Memory** | Medium | Low | High | Low |
| **Accuracy** | High (VI) | High (EN+VI) | High | Medium |
| **Training** | Custom | Fine-tune | Fine-tune | From scratch |
| **Best For** | Vietnamese | Fast inference | General | Lightweight |

---

## 🔧 Configuration Files

### AI2Text Config (`configs/default.yaml`)

```yaml
# Model architecture
d_model: 256
num_encoder_layers: 14
num_decoder_layers: 6
num_heads: 8
d_ff: 2048

# Training
batch_size: 64
gradient_accumulation_steps: 4
learning_rate: 0.0003
use_amp: true
use_bf16: true
```

### Whisper Config

```python
# Environment variables
ASR_MODEL=/path/to/model
ASR_DEVICE=auto
ASR_COMPUTE=float16  # hoặc int8
```

---

## 📝 Kết Luận

Hệ thống AI2Text sử dụng **nhiều công nghệ hiện đại** để đạt được:

1. **Độ chính xác cao**: Transformer với attention mechanism
2. **Tốc độ nhanh**: CTranslate2 optimization
3. **Memory hiệu quả**: Mixed precision, quantization
4. **Linh hoạt**: Nhiều model cho nhiều use case

Tất cả các models đều được **tối ưu cho RTX 5060 Ti 16GB** và **Ryzen 9 9900X**, đảm bảo performance tốt nhất trên hardware này.

---

**Last Updated**: 2024-12-09

