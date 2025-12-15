# AI2Text API Usage Guide

## 1) Khởi chạy server
```bash
cd /home/alida/Documents/Cursor/AI2Text/AI2Text
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Server dùng `checkpoints/best_model.pt` làm mặc định.

Kiểm tra health:
```bash
curl -s http://localhost:8000/health | jq .
```
Kỳ vọng: `status=healthy`, tokenizer/processor = true.

## 2) Gọi thử transcribe (greedy)
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@data/processed/full_merged_dataset/val/audio/047_000002198.wav" \
  -F "model_name=default" \
  -F "use_beam_search=false"
```
Kết quả trả về `text`, `confidence` (tạm 0.0), `processing_time`.

## 3) Tùy chọn beam search
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@path/to/audio.wav" \
  -F "model_name=default" \
  -F "use_beam_search=true" \
  -F "beam_width=5"
```

## 4) Chọn checkpoint khác (nếu có)
- Mặc định: `checkpoints/best_model.pt`.
- Có thể chỉ định `model_name` nếu bạn đặt file khác trong `checkpoints/` (ví dụ `checkpoints/my_model.pt` → `model_name=my_model.pt`). API sẽ thử các đường dẫn sau:
  - `checkpoints/best_model.pt` (ưu tiên)
  - `checkpoints/<model_name>.pt`
  - `checkpoints/<model_name>/best_model.pt`
  - `checkpoints/<model_name>/<model_name>.pt`
  - hoặc best_model mới nhất trong `checkpoints/` nếu không tìm thấy.

## 5) Lưu ý âm thanh đầu vào
- Bắt buộc 16 kHz mono; API tự resample nếu cần.
- Hỗ trợ WAV/MP3/FLAC/OGG... (torchaudio + librosa fallback).

## 6) Dừng server
```bash
pkill -f "uvicorn api.app:app"
```

## 7) Ghi chú phiên bản model
- Best checkpoint hiện tại: `checkpoints/best_model.pt` (epoch 17, val_loss≈4.285).
- Cấu hình: d_model=256, d_ff=2048, encoder_layers=14, decoder_layers=6, num_heads=8, vocab=3500 (SentencePiece).


