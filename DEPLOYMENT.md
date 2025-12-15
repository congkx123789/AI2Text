# Hướng Dẫn Triển Khai AI2Text trên Máy Mới

Hướng dẫn chi tiết để cài đặt và chạy hệ thống AI2Text trên một máy tính mới.

## 📋 Mục Lục

- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt Cơ Bản](#cài-đặt-cơ-bản)
- [Cài Đặt Dependencies](#cài-đặt-dependencies)
- [Cấu Hình Môi Trường](#cấu-hình-môi-trường)
- [Chạy Hệ Thống](#chạy-hệ-thống)
- [Kiểm Tra](#kiểm-tra)
- [Troubleshooting](#troubleshooting)

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu

- **CPU**: 4 cores trở lên (khuyến nghị: 8+ cores)
- **RAM**: 8GB (khuyến nghị: 16GB+)
- **GPU**: Tùy chọn, nhưng khuyến nghị cho training và inference nhanh
  - NVIDIA GPU với CUDA support (khuyến nghị: 6GB+ VRAM)
- **Ổ cứng**: 20GB+ dung lượng trống (cho models và data)

### Phần Mềm

- **OS**: Linux (Ubuntu 20.04+), macOS, hoặc Windows 10/11
- **Python**: 3.9, 3.10, hoặc 3.11
- **CUDA**: 11.8+ (nếu dùng GPU)
- **Git**: Để clone repository

## 🔧 Cài Đặt Cơ Bản

### 1. Clone Repository

```bash
# Clone project
git clone <repository-url> AI2Text
cd AI2Text

# Hoặc nếu đã có code, copy toàn bộ thư mục
```

### 2. Cài Đặt Python và Virtual Environment

#### Linux/macOS:

```bash
# Kiểm tra Python version
python3 --version  # Cần >= 3.9

# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
source venv/bin/activate
```

#### Windows:

```powershell
# Kiểm tra Python version
python --version  # Cần >= 3.9

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
venv\Scripts\activate
```

### 3. Cài Đặt System Dependencies

#### Ubuntu/Debian:

```bash
# Cài đặt FFmpeg (cần cho audio processing)
sudo apt-get update
sudo apt-get install -y ffmpeg

# Cài đặt build tools
sudo apt-get install -y build-essential

# Cài đặt Tesseract OCR (nếu dùng OCR)
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie
```

#### macOS:

```bash
# Sử dụng Homebrew
brew install ffmpeg
brew install tesseract tesseract-lang
```

#### Windows:

- Tải và cài FFmpeg từ: https://ffmpeg.org/download.html
- Tải và cài Tesseract từ: https://github.com/UB-Mannheim/tesseract/wiki

## 📦 Cài Đặt Dependencies

### 1. Cài Đặt Python Packages

```bash
# Đảm bảo đã kích hoạt virtual environment
pip install --upgrade pip

# Cài đặt dependencies chính
pip install -r requirements.txt
```

### 2. Cài Đặt Dependencies cho từng Sub-project

#### AI2Text (Transformer ASR):

```bash
# Dependencies đã có trong requirements.txt
# Không cần cài thêm
```

#### ai-llm (Whisper + LLM):

```bash
cd ai-llm

# Tạo virtual environment riêng (tùy chọn)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# hoặc .venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt  # Nếu có
# Hoặc cài thủ công:
pip install torch torchvision torchaudio
pip install transformers faster-whisper
pip install fastapi uvicorn
pip install sentence-transformers
pip install peft  # Cho LoRA models
```

#### ai-llm-ss (CTC ASR):

```bash
cd ai-llm-ss

# Cài đặt dependencies
pip install -r requirements.txt  # Nếu có
# Hoặc cài thủ công:
pip install torch torchaudio
pip install fastapi uvicorn
pip install librosa soundfile
```

### 3. Cài Đặt CUDA (Nếu dùng GPU)

#### Linux:

```bash
# Kiểm tra GPU
nvidia-smi

# Cài PyTorch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Windows:

- Tải CUDA Toolkit từ: https://developer.nvidia.com/cuda-downloads
- Cài PyTorch với CUDA:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ⚙️ Cấu Hình Môi Trường

### 1. Tạo File .env (Tùy chọn)

```bash
# Tạo file .env trong thư mục gốc
cat > .env << EOF
# ASR Configuration
ASR_MODEL=small
ASR_DEVICE=auto
ASR_COMPUTE=float16

# API Ports
AI2TEXT_PORT=8000
WHISPER_FINETUNE_PORT=8002
WHISPER_ORIGINAL_PORT=8003
CTC_ASR_PORT=8001
FRONTEND_PORT=5500

# Paths
PROJECT_ROOT=/path/to/AI2Text
MODEL_PATH=/path/to/models
DATA_PATH=/path/to/data
EOF
```

### 2. Tạo Thư Mục Cần Thiết

```bash
# Tạo các thư mục cho models và data
mkdir -p AI2Text/checkpoints
mkdir -p AI2Text/data
mkdir -p ai-llm/models/base
mkdir -p ai-llm/models/final
mkdir -p ai-llm-ss/data/results
mkdir -p logs
```

### 3. Download Models (Nếu cần)

#### Whisper Models:

```bash
cd ai-llm

# Download Whisper base model (tự động khi chạy lần đầu)
# Hoặc download thủ công:
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
WhisperProcessor.from_pretrained('openai/whisper-small'); \
WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')"
```

#### AI2Text Models:

- Copy `checkpoints/best_model.pt` từ máy cũ
- Hoặc train model mới (xem hướng dẫn training)

#### CTC Models:

- Copy `ai-llm-ss/data/results/asr_ctc.pt` từ máy cũ
- Hoặc train model mới

## 🚀 Chạy Hệ Thống

### Cách 1: Chạy Tất Cả Services (Khuyến nghị)

```bash
# Từ thư mục gốc
python start_all_services.py
```

### Cách 2: Chạy Từng Service Riêng

#### 1. AI2Text API (Port 8000):

```bash
cd AI2Text
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

#### 2. Whisper Fine-tuned API (Port 8002):

```bash
cd ai-llm
source .venv/bin/activate  # Linux/macOS
# hoặc .venv\Scripts\activate  # Windows

# Đảm bảo có model fine-tuned
export ASR_MODEL=/path/to/ai-llm/models/final/whisper-vi-en-ct2
export ASR_DEVICE=auto
export ASR_COMPUTE=float16  # hoặc int8 cho CPU

uvicorn src.api.server:app --host 0.0.0.0 --port 8002
```

#### 3. Whisper Original API (Port 8003):

```bash
cd ai-llm
source .venv/bin/activate

export ASR_MODEL=/path/to/ai-llm/models/base/whisper-small
export ASR_DEVICE=auto
export ASR_COMPUTE=int8  # CPU

uvicorn src.api.server:app --host 0.0.0.0 --port 8003
```

#### 4. CTC ASR API (Port 8001):

```bash
cd ai-llm-ss
python scripts/serve_asr.py --host 0.0.0.0 --port 8001
```

#### 5. Frontend (Port 5500):

```bash
cd frontend
python3 -m http.server 5500
```

### Cách 3: Sử Dụng Scripts

#### Linux/macOS:

```bash
# Start all services
./start_all.sh

# Stop all services
./stop_all.sh
```

#### Windows:

```powershell
# Start all services
.\start_all.sh

# Stop all services
.\stop_all.sh
```

## ✅ Kiểm Tra

### 1. Kiểm Tra Health Check

```bash
# AI2Text API
curl http://localhost:8000/health

# Whisper APIs
curl http://localhost:8002/health
curl http://localhost:8003/health

# CTC ASR API
curl http://localhost:8001/health
```

### 2. Kiểm Tra Models

```bash
# List models trong AI2Text
curl http://localhost:8000/models

# List models trong Whisper (nếu có endpoint)
```

### 3. Test Transcription

```bash
# Test với AI2Text
curl -X POST "http://localhost:8000/transcribe" \
     -F "audio=@test_audio.wav" \
     -F "model_name=best_model"

# Test với Whisper
curl -X POST "http://localhost:8002/transcribe/upload" \
     -F "file=@test_audio.wav" \
     -F "language=vi"
```

### 4. Kiểm Tra Frontend

Mở trình duyệt và truy cập: `http://localhost:5500`

## 🔍 Troubleshooting

### Lỗi: ModuleNotFoundError

**Nguyên nhân**: Thiếu dependencies

**Giải pháp**:
```bash
# Cài lại dependencies
pip install -r requirements.txt

# Hoặc cài thủ công module bị thiếu
pip install <module_name>
```

### Lỗi: CUDA out of memory

**Nguyên nhân**: GPU không đủ VRAM

**Giải pháp**:
```bash
# Dùng CPU thay vì GPU
export ASR_DEVICE=cpu
export ASR_COMPUTE=int8

# Hoặc giảm batch size trong code
```

### Lỗi: Model not found

**Nguyên nhân**: Đường dẫn model sai hoặc model chưa được copy

**Giải pháp**:
```bash
# Kiểm tra đường dẫn model
ls -la AI2Text/checkpoints/
ls -la ai-llm/models/final/
ls -la ai-llm-ss/data/results/

# Copy models từ máy cũ nếu thiếu
```

### Lỗi: Port already in use

**Nguyên nhân**: Port đã được sử dụng bởi process khác

**Giải pháp**:
```bash
# Tìm process đang dùng port
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Hoặc đổi port trong code/config
```

### Lỗi: FFmpeg not found

**Nguyên nhân**: Chưa cài FFmpeg

**Giải pháp**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Tải từ https://ffmpeg.org/download.html
```

### Lỗi: Permission denied

**Nguyên nhân**: Không có quyền truy cập file/thư mục

**Giải pháp**:
```bash
# Linux/macOS: Thay đổi quyền
chmod +x scripts/*.sh
chmod -R 755 checkpoints/

# Hoặc chạy với sudo (không khuyến nghị)
```

### Lỗi: ImportError với transformers

**Nguyên nhân**: Version không tương thích

**Giải pháp**:
```bash
# Cài đúng version
pip install transformers==4.44.2
pip install torch torchvision torchaudio
```

## 📝 Checklist Triển Khai

Trước khi chạy, đảm bảo:

- [ ] Python 3.9+ đã được cài đặt
- [ ] Virtual environment đã được tạo và kích hoạt
- [ ] Tất cả dependencies đã được cài đặt
- [ ] FFmpeg đã được cài đặt
- [ ] Models đã được copy hoặc download
- [ ] Các thư mục cần thiết đã được tạo
- [ ] Ports 8000, 8001, 8002, 8003, 5500 đã được mở
- [ ] CUDA đã được cài đặt (nếu dùng GPU)
- [ ] Health check endpoints hoạt động
- [ ] Frontend có thể truy cập được

## 🔄 Cập Nhật Từ Máy Cũ

### Copy Models:

```bash
# Từ máy cũ, copy models
scp -r user@old-machine:/path/to/AI2Text/checkpoints ./AI2Text/
scp -r user@old-machine:/path/to/ai-llm/models ./ai-llm/
scp -r user@old-machine:/path/to/ai-llm-ss/data/results ./ai-llm-ss/data/
```

### Copy Config:

```bash
# Copy config files
scp user@old-machine:/path/to/AI2Text/configs/* ./AI2Text/configs/
```

## 📞 Hỗ Trợ

Nếu gặp vấn đề:

1. Kiểm tra logs trong thư mục `logs/`
2. Kiểm tra output của các API endpoints
3. Xem lại phần Troubleshooting
4. Kiểm tra version của Python và các packages

---

**Lưu ý**: Đảm bảo đường dẫn trong code phù hợp với cấu trúc thư mục trên máy mới. Nếu cần, cập nhật các đường dẫn tuyệt đối trong code.

