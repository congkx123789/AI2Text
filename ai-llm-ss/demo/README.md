# ASR Model Demo Package

Gói demo này cho phép bạn chạy model ASR (Automatic Speech Recognition) trên máy khác mà không cần toàn bộ codebase.

## 📦 Nội dung

- `checkpoint_epoch_12.pt` - Model checkpoint đã được train
- `vocab.json` - Vocabulary file (từ điển ký tự)
- `model_code/` - Code model cần thiết (CRNNCTC, features, decoder)
- `demo_inference.py` - Script chạy inference đơn giản
- `requirements.txt` - Dependencies cần thiết

## 🚀 Cài đặt

### 1. Cài đặt Python dependencies

**Cho CPU:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

**Cho GPU (CUDA 11.8):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy
```

**Hoặc dùng requirements.txt:**
```bash
pip install -r requirements.txt
```

**Lưu ý:** Bạn cần cài đúng version PyTorch phù hợp với hệ thống của mình (CPU hoặc CUDA).

## 📝 Sử dụng

### Chạy inference trên file audio

**Cách đơn giản nhất (khuyến nghị):**
```bash
# Chỉ định ngôn ngữ để model chạy chuẩn hơn
python3 demo_inference.py audio.wav vi
python3 demo_inference.py audio.wav en

# Hoặc dùng run.py
python3 run.py audio.wav vi
python3 run.py audio.wav en
```

**Cách dùng với flags:**
```bash
# Chạy trên CPU
python3 demo_inference.py --audio test_audio.wav --language vi

# Chạy trên GPU (nếu có)
python3 demo_inference.py --audio test_audio.wav --language vi --device cuda

# Chỉ định checkpoint và vocab cụ thể
python3 demo_inference.py --audio test_audio.wav --language vi --checkpoint checkpoint_epoch_12.pt --vocab vocab.json
```

### Các tham số

**Positional arguments (đối số vị trí - cách đơn giản):**
- `<audio_file>`: Đường dẫn đến file audio (wav, mp3, flac, etc.)
- `[language]`: Mã ngôn ngữ (`vi` cho Tiếng Việt, `en` cho Tiếng Anh) - **Khuyến nghị chỉ định để model chạy chuẩn hơn**

**Optional arguments (tham số tùy chọn):**
- `--audio`: Đường dẫn đến file audio (thay thế cho positional argument)
- `--language` hoặc `--lang`: Mã ngôn ngữ (`vi`, `en`) - giúp model validate và chạy tốt hơn
- `--checkpoint`: Đường dẫn đến checkpoint file (mặc định: `checkpoint_epoch_12.pt`)
- `--vocab`: Đường dẫn đến vocabulary file (mặc định: `vocab.json`)
- `--device`: Device để chạy (`auto`, `cpu`, hoặc `cuda`, mặc định: `auto`)
- `--detect_language`: Bật language detection (phát hiện ngôn ngữ từ text sau khi transcribe)

### Chỉ định ngôn ngữ (Khuyến nghị)

**Quan trọng:** Chỉ định ngôn ngữ giúp model chạy chuẩn hơn và validate kết quả:

```bash
# Chỉ định Tiếng Việt
python3 demo_inference.py audio.wav vi
python3 run.py audio.wav vi

# Chỉ định Tiếng Anh
python3 demo_inference.py audio.wav en
python3 run.py audio.wav en

# Dùng flag
python3 demo_inference.py --audio audio.wav --language vi
```

Khi chỉ định ngôn ngữ, model sẽ:
- Hiển thị ngôn ngữ đã chỉ định
- Tự động detect ngôn ngữ từ kết quả transcription
- Cảnh báo nếu ngôn ngữ detect khác với ngôn ngữ chỉ định

### Language Detection (Phát hiện ngôn ngữ tự động)

Model có khả năng tự động phát hiện ngôn ngữ (Tiếng Việt hoặc Tiếng Anh) từ text sau khi transcribe:

```bash
# Bật language detection (không chỉ định ngôn ngữ trước)
python3 demo_inference.py audio.wav --detect_language

# Hoặc chỉ định ngôn ngữ (tự động bật detection để validate)
python3 demo_inference.py audio.wav vi
```

**Cách hoạt động:**
1. **Model tag extraction**: Model tự động output language tag (`<|vi|>` hoặc `<|en|>`) nếu được train với tags
2. **Text-based detection**: Nếu không có tag, sử dụng thư viện `langdetect` hoặc `langid` để detect từ text
3. **Heuristic fallback**: Kiểm tra ký tự tiếng Việt trong text

**Output với language detection:**
```
============================================================
TRANSCRIPTION RESULT
============================================================
Audio file: test_audio.wav
Transcribed text: <|vi|> xin chào đây là tiếng việt

Language Detection:
  Language: Vietnamese (vi)
  Confidence: 100.00%
  Method: tag_extraction
  Source: model_tag

Cleaned text (no tags): xin chào đây là tiếng việt
============================================================
```

## 🎯 Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **PyTorch**: 2.0.0 trở lên
- **torchaudio**: 2.0.0 trở lên
- **RAM**: Tối thiểu 2GB (khuyến nghị 4GB+)
- **GPU**: Tùy chọn (CUDA 11.8+ nếu dùng GPU)

## 📊 Thông tin model

- **Architecture**: CRNN-CTC (Convolutional Recurrent Neural Network với Connectionist Temporal Classification)
- **Input**: Audio mono, 16kHz sample rate
- **Features**: Log Mel Spectrogram (80 mel bands)
- **Vocabulary size**: 107 ký tự
- **Model parameters**: ~4.1M parameters

## 🔧 Xử lý lỗi

### Lỗi: "CUDA out of memory"
- Giảm batch size hoặc chạy trên CPU: `--device cpu`

### Lỗi: "No module named 'torch'"
- Cài đặt PyTorch: `pip install torch torchaudio`

### Lỗi: "Audio file not found"
- Kiểm tra đường dẫn file audio có đúng không
- Đảm bảo file audio tồn tại và có quyền đọc

### Lỗi: "Checkpoint not found"
- Đảm bảo file `checkpoint_epoch_12.pt` nằm trong cùng thư mục với script

## 📝 Ví dụ output

**Ví dụ với ngôn ngữ được chỉ định:**
```
Using device: cpu

Loading vocabulary from vocab.json...
Vocabulary size: 107
Loading model from checkpoint_epoch_12.pt...
Loaded checkpoint from epoch 12
Model parameters: 4,132,715

Loading audio: test_audio.wav
Expected language: Vietnamese (vi)
Running inference...
✓ Language matches expected: vi

============================================================
TRANSCRIPTION RESULT
============================================================
Audio file: test_audio.wav
Specified language: Vietnamese (vi)
Transcribed text: thiền học truyền thống thường chọn đối tượng thiền quán

Language Detection:
  Language: Vietnamese (vi)
  Confidence: 100.00%
  Method: tag_extraction
  Source: model_tag

Cleaned text (no tags): thiền học truyền thống thường chọn đối tượng thiền quán
============================================================
```

**Ví dụ đơn giản:**
```
Loading vocabulary from vocab.json...
Vocabulary size: 107
Loading model from checkpoint_epoch_12.pt...
Loaded checkpoint from epoch 12
Model parameters: 4,132,715

Loading audio: test_audio.wav
Running inference...

============================================================
TRANSCRIPTION RESULT
============================================================
Audio file: test_audio.wav
Transcribed text: thiền học truyền thống thường chọn đối tượng thiền quán
============================================================
```

## 🔗 Tích hợp vào code Python

Nếu muốn tích hợp vào code của bạn:

```python
import torch
from model_code.model import CRNNCTC
from model_code.features import wav_to_logmelspec, ensure_mono16k
from model_code.decode import greedy_decode
import json
import torchaudio

# Load vocabulary
with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
itos = {i: c for i, c in enumerate(vocab)}

# Load model
model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
checkpoint = torch.load('checkpoint_epoch_12.pt', map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Load and preprocess audio
waveform, sr = torchaudio.load('audio.wav')
waveform, sr = ensure_mono16k(waveform, sr)
features = wav_to_logmelspec(waveform, sr)
features = features.unsqueeze(0)  # Add batch dimension
lengths = torch.tensor([features.shape[1]])

# Inference
with torch.no_grad():
    logits, _ = model(features, lengths)
    text = greedy_decode(logits, itos)[0]

print(f"Transcribed: {text}")
```

## 📄 License

Model và code này thuộc về dự án ASR của bạn.

## ❓ Hỗ trợ

Nếu gặp vấn đề, vui lòng kiểm tra:
1. Python version (>= 3.8)
2. PyTorch version (>= 2.0.0)
3. File paths có đúng không
4. Audio format có được hỗ trợ không (wav, mp3, flac)

