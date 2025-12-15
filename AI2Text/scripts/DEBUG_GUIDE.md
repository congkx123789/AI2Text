# 🔍 Hướng Dẫn Debug ASR Model

## ⚠️ Cực Kỳ Quan Trọng: Kiểm Tra Sample Rate

Mô hình ASR (như Wav2Vec2, Whisper, HuBERT) thường yêu cầu đầu vào chuẩn **16,000 Hz**.

### Rủi Ro Nếu Không Resample

Nếu file gốc là 44.1kHz hoặc 48kHz mà bạn đẩy thẳng vào model 16kHz, model sẽ nghe tiếng người nói như **"người ngoài hành tinh"** (tua chậm/nhanh), dẫn đến **WER 100% mãi mãi**.

### ✅ Kiểm Tra Code Của Bạn

Code đã có resample trong `preprocessing/audio_processing.py`:
- Dòng 85-88: Tự động resample nếu sample rate khác 16kHz
- Sử dụng `torchaudio.transforms.Resample` hoặc `librosa.load(sr=16000)`

## 📋 Scripts Debug

### 1. Kiểm Tra Sample Rate

#### Kiểm tra 1 file:
```bash
python scripts/check_sample_rate.py data/audio/sample.wav
```

#### Kiểm tra nhiều file:
```bash
python scripts/check_sample_rate.py data/audio/*.wav
```

#### Kiểm tra toàn bộ dataset:
```bash
# Kiểm tra tất cả file trong dataset
python scripts/check_sample_rate.py --dataset data/processed/full_merged_dataset/train

# Chỉ kiểm tra 100 file đầu tiên (để test nhanh)
python scripts/check_sample_rate.py --dataset data/processed/full_merged_dataset/train --sample 100
```

**Kết quả mong đợi:**
- ✅ Tất cả file phải là 16kHz
- ⚠️ Nếu có file khác 16kHz, cần kiểm tra lại code resample

### 2. Debug Predictions

#### Quick Debug (Nhanh, 5-10 samples):
```bash
python scripts/quick_debug_prediction.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default.yaml \
    --num_samples 10
```

#### Full Debug (Chi tiết hơn):
```bash
python scripts/debug_predictions.py \
    --model checkpoints/best_model.pt \
    --dataset data/processed/full_merged_dataset/val \
    --manifest data/processed/full_merged_dataset/val/manifest.csv \
    --num_samples 20
```

#### Tự động debug khi evaluate:
Script `evaluate_checkpoint.py` đã được cập nhật để tự động in 5 predictions đầu tiên khi evaluate.

## 🎯 Phân Tích Kết Quả

### Kịch Bản TỐT (Model đang học):

```
Ref: "xin chào việt nam"
Pred: "xxiin cchhaààoo vviiệệtt nnaamm" (Bị lặp ký tự)
hoặc
Pred: "sin chao viet nam" (Sai dấu)
```

**→ Tiếp tục train.** CTC sẽ sửa lỗi này sau.

### Kịch Bản XẤU 1 (Lỗi Blank/Tokenizer):

```
Ref: "xin chào việt nam"
Pred: "" (Rỗng tuếch)
```

**→ Dừng ngay!** Kiểm tra lại:
- Learning rate (có thể quá cao/thấp)
- Tokenizer (có đúng không?)
- Model architecture

### Kịch Bản XẤU 2 (Lỗi Sample Rate):

```
Ref: "xin chào việt nam"
Pred: "xin ch" (Bị cắt cụt)
hoặc
Pred: "x i n c h a o v i e t n a m a b c..." (Dài lê thê vô nghĩa)
```

**→ Dừng ngay!** Fix lại code xử lý âm thanh:
1. Chạy `check_sample_rate.py` để kiểm tra
2. Đảm bảo `AudioProcessor.load_audio()` resample về 16kHz
3. Kiểm tra lại trong `training/dataset.py` dòng 306

## 🔧 Checklist Khi Training

### Trước khi bắt đầu training:

- [ ] **Kiểm tra sample rate**: Chạy `check_sample_rate.py` trên một vài file
- [ ] **Kiểm tra code resample**: Đảm bảo `AudioProcessor.load_audio()` có resample
- [ ] **Test load 1 sample**: Đảm bảo audio được load và resample đúng

### Trong quá trình training:

- [ ] **Debug predictions**: Sau mỗi vài epoch, chạy `quick_debug_prediction.py`
- [ ] **Kiểm tra loss**: Loss phải giảm dần, không bị NaN
- [ ] **Kiểm tra WER/CER**: Phải giảm dần, không bị stuck ở 100%

### Khi có vấn đề:

1. **WER/CER = 100% mãi mãi:**
   - ✅ Chạy `check_sample_rate.py` → Kiểm tra sample rate
   - ✅ Chạy `quick_debug_prediction.py` → Xem predictions
   - ✅ Kiểm tra learning rate
   - ✅ Kiểm tra tokenizer

2. **Predictions rỗng:**
   - ✅ Kiểm tra learning rate (có thể quá cao)
   - ✅ Kiểm tra tokenizer
   - ✅ Kiểm tra model architecture

3. **Predictions bất thường về độ dài:**
   - ✅ Chạy `check_sample_rate.py` → Có thể do sample rate
   - ✅ Kiểm tra audio processing pipeline

## 📊 Ví Dụ Output

### Sample Rate Check:
```
📁 File: sample.wav
   Sample Rate: 16000 Hz
   Duration: 7.57 seconds
   ✅ Sample rate đúng (16kHz)
```

### Debug Prediction:
```
Sample #1
🎯 Reference: xin chào việt nam
🤖 Prediction: xxiin cchhaààoo vviiệệtt nnaamm
✅ KỊCH BẢN TỐT: Model đang học (lặp ký tự - bình thường với CTC)
```

## 🚀 Quick Start

1. **Kiểm tra sample rate của dataset:**
   ```bash
   python scripts/check_sample_rate.py --dataset data/processed/full_merged_dataset/train --sample 100
   ```

2. **Debug predictions sau training:**
   ```bash
   python scripts/quick_debug_prediction.py \
       --checkpoint checkpoints/best_model.pt \
       --config configs/default.yaml \
       --num_samples 10
   ```

3. **Evaluate với auto-debug:**
   ```bash
   python evaluate_checkpoint.py checkpoints/best_model.pt
   # Sẽ tự động in 5 predictions đầu tiên
   ```

## 📝 Notes

- Tất cả scripts đều có `--help` để xem chi tiết
- Scripts tự động fallback sang librosa nếu torchaudio không hoạt động
- Có thể chạy trên CPU hoặc GPU (tự động detect)

