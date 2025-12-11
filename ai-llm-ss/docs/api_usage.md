## ASR API Usage

Hướng dẫn chạy API server để load model và phục vụ inference.

### 1) Chuẩn bị môi trường
```bash
cd /home/alida/Documents/Cursor/AI2Text/ai-llm-ss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Kiểm tra model và vocab
- Model mặc định: `data/results/asr_ctc.pt`
- Checkpoint mới nhất: `data/results/checkpoints/checkpoint_epoch_12.pt`
- Vocab: `data/processed/vocab.json`

**Danh sách model sẵn có**
- `data/results/asr_ctc.pt` (mặc định, CTC)
- `data/results/checkpoints/checkpoint_epoch_12.pt` (checkpoint fine-tune mới nhất)

Bạn có thể đổi model/vocab bằng cách truyền tham số cho API server (xem bên dưới).

### 3) Chạy API server
```bash
python3 scripts/serve_asr.py --host 0.0.0.0 --port 8001
```

Tùy chọn:
- `--model-path`: đường dẫn model hoặc checkpoint (mặc định: `data/results/asr_ctc.pt`)
- `--vocab-path`: đường dẫn vocab (mặc định: `data/processed/vocab.json`)
- `--device`: `cpu`, `cuda`, hoặc `auto` (mặc định: `auto`)

Ví dụ đổi checkpoint:
```bash
python3 scripts/serve_asr.py \
  --host 0.0.0.0 --port 8001 \
  --model-path data/results/checkpoints/checkpoint_epoch_12.pt \
  --vocab-path data/processed/vocab.json \
  --device auto
```

### 4) Các endpoint chính
- `GET /health`: kiểm tra server và model đã load
- `GET /model/info`: thông tin model (vocab size, device, đường dẫn model)
- `POST /transcribe`: nhận file audio và trả về transcript

### 5) Gửi request mẫu
#### Curl
```bash
curl -X POST "http://localhost:8001/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"
```

#### Python requests
```python
import requests

url = "http://localhost:8001/transcribe"
with open("audio.wav", "rb") as f:
    resp = requests.post(url, files={"file": ("audio.wav", f, "audio/wav")})
print(resp.json())
```

### 6) Input audio yêu cầu
- 16 kHz mono là tốt nhất; server sẽ tự resample và chuyển mono nếu cần.
- Định dạng: WAV/FLAC/MP3/OGG (torchaudio + soundfile fallback).

### 7) Ghi chú hiệu năng
- Device mặc định `auto` (ưu tiên GPU nếu có).
- Greedy decoding (không LM), nên nhẹ và nhanh.

### 8) Khởi động lại server khi đổi model
Mỗi lần thay đổi `--model-path` hoặc `--vocab-path`, hãy dừng server cũ (`Ctrl+C`) và chạy lại lệnh khởi động với tham số mới.

