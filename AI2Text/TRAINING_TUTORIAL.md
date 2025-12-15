# 🚀 Training Tutorial - Hướng Dẫn Chạy Training

## 📋 Yêu Cầu Trước Khi Bắt Đầu

1. **Đã cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Đã có dataset:**
   - Dataset nằm tại: `data/processed/full_merged_dataset/`
   - Có file `manifest.csv` trong `train/` và `val/`

3. **Đã có tokenizer:**
   - File tokenizer: `models/tokenizer_vi_en_3500.model`

## 🎯 Các Lệnh Training Cơ Bản

### 1. Training Từ Đầu (Fresh Start)

```bash
cd /home/alida/Documents/Cursor/AI2Text/AI2Text
python training/train.py --config configs/default.yaml
```

**Lưu ý:** Training sẽ bắt đầu từ epoch 0 và tạo checkpoint mới.

---

### 2. Resume Training Từ Checkpoint

#### Resume từ checkpoint mới nhất:
```bash
cd /home/alida/Documents/Cursor/AI2Text/AI2Text
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt
```

#### Resume từ checkpoint cụ thể:
```bash
python training/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_5.pt
```

#### Tự động tìm checkpoint mới nhất:
```bash
NEWEST=$(python -c "import os; checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]; checkpoints.sort(key=lambda x: os.path.getmtime(f'checkpoints/{x}'), reverse=True); print(checkpoints[0] if checkpoints else 'best_model.pt')")
python training/train.py --config configs/default.yaml --resume "checkpoints/$NEWEST"
```

---

### 3. Training Và Lưu Log

```bash
python training/train.py --config configs/default.yaml 2>&1 | tee training_output.log
```

Hoặc với resume:
```bash
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt 2>&1 | tee training_output.log
```

---

### 4. Training Chạy Nền (Background)

```bash
# Chạy training ở background và lưu log
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt 2>&1 | tee training_output.log &
```

**Kiểm tra process:**
```bash
ps aux | grep train.py
```

**Dừng training:**
```bash
pkill -f "train.py"
```

---

## 📊 Kiểm Tra Trạng Thái Training

### Xem log real-time:
```bash
tail -f training_output.log
# hoặc
tail -f logs/training.log
```

### Xem checkpoint đã có:
```bash
ls -lht checkpoints/*.pt
```

### Xem thông tin checkpoint:
```bash
python -c "import torch; ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False); print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}'); print(f'Best Val Loss: {ckpt.get(\"best_val_loss\", \"N/A\")}')"
```

---

## 🧪 Quick Test Training

### Test trên 1 file audio, 20 epochs:
```bash
python test_quick_train.py
```

---

## 🔍 Đánh Giá Model

### Test WER/CER nhanh (3 samples):
```bash
python quick_test_wer.py --checkpoint checkpoints/best_model.pt --num-samples 3
```

### Đánh giá đầy đủ:
```bash
python evaluate_checkpoint.py --checkpoint checkpoints/best_model.pt --max-batches 3
```

---

## ⚙️ Cấu Hình Training

File cấu hình: `configs/default.yaml`

### Các tham số quan trọng:

```yaml
# Training
num_epochs: 50              # Tổng số epoch
batch_size: 64              # Batch size
learning_rate: 0.0003       # Learning rate
gradient_accumulation_steps: 4  # Effective batch = 64 * 4 = 256

# Mixed Precision
use_amp: true               # Bật mixed precision
use_bf16: true              # Sử dụng bfloat16 (tốt hơn float16)

# Checkpointing
save_every: 1               # Lưu checkpoint mỗi epoch

# Validation
calculate_val_wer: false    # Tắt tính WER để validation nhanh hơn
```

---

## 📈 Theo Dõi Training

### Metrics hiển thị:
- **Loss**: Loss hiện tại và trung bình
- **LR**: Learning rate hiện tại
- **BestVal**: Best validation loss
- **Progress**: Tiến độ epoch (batch/total)

### Ví dụ output:
```
🚀 Epoch 7/50:   5%|██▌                    | 150/3033 [02:15<42:21, 1.19batch/s]
Loss: 5.1234 | Avg: 5.2345 | LR: 3.00e-04 | BestVal: 4.9994
```

---

## 🛠️ Troubleshooting

### 1. GPU Out of Memory
```bash
# Giảm batch_size trong configs/default.yaml
batch_size: 32  # Thay vì 64
```

### 2. Training bị dừng đột ngột
```bash
# Kiểm tra log
tail -100 logs/training.log

# Resume từ checkpoint cuối cùng
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt
```

### 3. Muốn dừng training
```bash
# Tìm process
ps aux | grep train.py

# Dừng process
pkill -f "train.py"
# hoặc
kill <PID>
```

---

## 📝 Workflow Đề Xuất

### 1. Lần đầu training:
```bash
# Bước 1: Kiểm tra dataset
ls data/processed/full_merged_dataset/train/

# Bước 2: Bắt đầu training
python training/train.py --config configs/default.yaml 2>&1 | tee training_output.log

# Bước 3: Theo dõi (terminal khác)
tail -f training_output.log
```

### 2. Tiếp tục training:
```bash
# Bước 1: Kiểm tra checkpoint mới nhất
ls -lht checkpoints/*.pt | head -1

# Bước 2: Resume training
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt 2>&1 | tee training_output.log

# Bước 3: Theo dõi
tail -f training_output.log
```

### 3. Đánh giá model:
```bash
# Sau khi training xong một vài epoch
python quick_test_wer.py --checkpoint checkpoints/best_model.pt --num-samples 5
```

---

## 🎓 Tips

1. **Luôn lưu log:** Dùng `tee` để lưu output vào file
2. **Kiểm tra checkpoint thường xuyên:** Đảm bảo checkpoint được lưu đúng
3. **Monitor GPU:** Dùng `nvidia-smi` để theo dõi GPU usage
4. **Backup checkpoint:** Copy checkpoint quan trọng ra nơi khác
5. **Resume đúng checkpoint:** Luôn resume từ checkpoint mới nhất hoặc best model

---

## 📞 Lệnh Nhanh (Quick Reference)

```bash
# Training từ đầu
python training/train.py --config configs/default.yaml

# Resume từ best model
python training/train.py --config configs/default.yaml --resume checkpoints/best_model.pt

# Resume từ checkpoint mới nhất
NEWEST=$(python -c "import os; checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]; checkpoints.sort(key=lambda x: os.path.getmtime(f'checkpoints/{x}'), reverse=True); print(checkpoints[0] if checkpoints else 'best_model.pt')")
python training/train.py --config configs/default.yaml --resume "checkpoints/$NEWEST"

# Xem log
tail -f training_output.log

# Kiểm tra checkpoint
ls -lht checkpoints/*.pt

# Test WER/CER
python quick_test_wer.py --checkpoint checkpoints/best_model.pt --num-samples 3

# Dừng training
pkill -f "train.py"
```

---

**Chúc bạn training thành công! 🚀**

