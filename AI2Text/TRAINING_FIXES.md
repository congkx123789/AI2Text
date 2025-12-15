# 🔧 CÁC FIXES ĐÃ ÁP DỤNG

## Vấn đề: Model bị "Ảo giác" (Hallucination)
Model đang học văn bản thay vì nghe âm thanh, tạo ra các câu generic không liên quan đến audio input.

## ✅ Các Fixes Đã Implement

### 1. **Hybrid CTC/Attention Loss** ✅
- **File**: `utils/ctc_loss.py`, `models/asr_base.py`, `training/train.py`
- **Mục đích**: Ép Encoder phải học alignment giữa audio và text
- **Cách hoạt động**: 
  - Thêm CTC projection từ Encoder output
  - Loss = 0.7 * Attention_Loss + 0.3 * CTC_Loss
- **Config**: `use_ctc_loss: true`, `ctc_weight: 0.3`

### 2. **Scheduled Sampling** ✅
- **File**: `training/scheduled_sampling.py`, `training/train.py`
- **Mục đích**: Giảm Teacher Forcing, buộc model dùng encoder output
- **Cách hoạt động**: 
  - Bắt đầu: 100% teacher forcing
  - Kết thúc: 50% teacher forcing (từ từ giảm)
- **Config**: `use_scheduled_sampling: true`, `teacher_forcing_initial: 1.0`, `teacher_forcing_final: 0.5`

### 3. **WER/CER Monitoring Mỗi 100 Batch** ✅
- **File**: `training/train.py`
- **Mục đích**: Theo dõi chất lượng model trong quá trình training
- **Cách hoạt động**: Tính WER/CER trên 8 samples mỗi 100 batch và log vào progress bar + log file

### 4. **Data Validation Scripts** ✅
- **File**: `scripts/validate_data_alignment.py`, `scripts/quick_check_data.py`
- **Mục đích**: Kiểm tra audio-text alignment trước khi train
- **Cách dùng**: 
  ```bash
  # Quick check
  python scripts/quick_check_data.py --split train --indices 0 1 2 10 100
  
  # Full validation
  python scripts/validate_data_alignment.py --split train --num-samples 100
  ```

## 🚀 QUY TRÌNH TRAIN LẠI TỪ ĐẦU

### Bước 1: Dừng Training Hiện Tại
```bash
# Nếu đang chạy, nhấn Ctrl+C để dừng
```

### Bước 2: Kiểm Tra Data Alignment (BẮT BUỘC)
```bash
# Quick check - kiểm tra nhanh một vài samples
python scripts/quick_check_data.py --split train --indices 0 1 2 10 100 500

# Nếu thấy lỗi, FIX NGAY trước khi train!
```

### Bước 3: Xóa Checkpoints Cũ
```bash
# Backup checkpoints cũ (optional)
mv checkpoints checkpoints_old_backup

# Hoặc xóa hẳn
rm -rf checkpoints/*
```

### Bước 4: Kiểm Tra Config
Đảm bảo `configs/default.yaml` có:
```yaml
# Hybrid CTC/Attention
use_ctc_loss: true
ctc_weight: 0.3

# Scheduled Sampling
use_scheduled_sampling: true
teacher_forcing_initial: 1.0
teacher_forcing_final: 0.5

# Giảm num_workers để tránh BrokenPipeError
num_workers: 2  # Thay vì 12
```

### Bước 5: Train Lại Từ Đầu
```bash
python train.py --config configs/default.yaml
```

## 📊 Theo Dõi Training

### WER/CER Mỗi 100 Batch
Trong log và progress bar, bạn sẽ thấy:
```
Batch 100/3033 | Loss: 4.1234 | WER: 95.2% | CER: 87.5%
```

### Dấu Hiệu Tốt:
- WER/CER giảm dần theo thời gian
- Predictions bắt đầu liên quan đến audio (không còn generic text)
- Loss giảm ổn định

### Dấu Hiệu Xấu:
- WER/CER không giảm sau 5-10 epochs
- Vẫn còn generic text như "and the oldest of the future..."
- Loss không giảm hoặc tăng

## ⚠️ LƯU Ý QUAN TRỌNG

1. **PHẢI kiểm tra data alignment trước khi train** - Đây là nguyên nhân #1 gây ra hallucination
2. **Xóa checkpoints cũ** - Model cũ đã học sai, không nên resume
3. **Theo dõi WER/CER** - Nếu sau 5 epochs vẫn > 100%, có thể cần điều chỉnh thêm
4. **Kiên nhẫn** - Model cần thời gian để học lại từ đầu với các fixes mới

## 🔍 Debug Nếu Vẫn Có Vấn Đề

1. **Kiểm tra data loader**:
   ```python
   # Trong training/dataset.py, thêm print trong __getitem__
   print(f"Loading: {audio_path} -> {transcript}")
   ```

2. **Kiểm tra CTC loss**:
   - Xem log có "✅ CTC Loss enabled" không
   - Loss có giảm không

3. **Kiểm tra scheduled sampling**:
   - Xem log có "✅ Scheduled Sampling enabled" không
   - Teacher forcing probability có giảm theo epoch không

## 📝 Files Đã Thay Đổi

- `models/asr_base.py` - Thêm CTC output
- `training/train.py` - Thêm CTC loss, scheduled sampling, WER/CER monitoring
- `training/scheduled_sampling.py` - Scheduled sampling utility
- `utils/ctc_loss.py` - CTC loss implementation
- `configs/default.yaml` - Config cho các tính năng mới
- `scripts/validate_data_alignment.py` - Data validation script
- `scripts/quick_check_data.py` - Quick data check script

