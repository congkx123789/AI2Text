# Báo Cáo Test Dataset và Kết Quả Model

**Ngày tạo:** 2024-12-12  
**Model:** checkpoint_epoch_12.pt  
**Epoch:** 12

---

## 📊 Tổng Quan Test Dataset

### Thông Tin Cơ Bản

| Thông số | Giá trị |
|----------|---------|
| **Tổng số samples** | **32,818** |
| **Số file audio** | 32,891 |
| **Kích thước audio** | 6.9 GB |
| **Format manifest** | CSV với 5 cột |
| **Format audio** | WAV files |
| **Có timestamps** | ✅ Có (word-level alignments) |

### Cấu Trúc Dataset

```
data/processed/test/
├── manifest.csv              # 32,819 dòng (32,818 samples + 1 header)
├── manifest.csv.backup       # Backup file
├── timestamps.json           # Word-level alignments cho tất cả files
└── audio/                    # 32,891 WAV files (6.9 GB)
    ├── 083_000007975.wav
    ├── 6689-64264-0014.wav
    └── ...
```

### Format Manifest CSV

File `manifest.csv` chứa các cột sau:

| Cột | Mô tả | Ví dụ |
|-----|-------|-------|
| `id` | Unique identifier | `083_000007975` |
| `transcript` | Text transcription với language tag | `<|vi|> tiến vào thế kỷ hai mươi` |
| `audio_path` | Đường dẫn tương đối đến file audio | `audio/083_000007975.wav` |
| `duration` | Độ dài audio (giây) | `5.076` |
| `words_json` | JSON array chứa word-level timestamps | `[{"word": "tiến", "start": 0.0, "end": 0.443}, ...]` |

### Language Tags

Dataset hỗ trợ 2 ngôn ngữ với language tags:
- **Tiếng Việt**: `<|vi|>` prefix trong transcript
- **Tiếng Anh**: `<|en|>` prefix trong transcript

### Timestamps.json

File `timestamps.json` chứa word-level alignments cho mỗi audio file:

```json
{
  "083_000007975.wav": {
    "duration": 5.056,
    "text": "<|vi|> tiến vào thế kỷ hai mươi",
    "segments": [
      {
        "word": "tiến",
        "start": 0.0,
        "end": 0.443,
        "score": 0.011
      },
      ...
    ]
  }
}
```

### Thống Kê Audio

- **Tổng thời lượng audio:** 227,287.43 giây (~63.1 giờ)
- **Độ dài trung bình:** 6.98 giây/sample
- **Độ dài ngắn nhất:** 5.02 giây
- **Độ dài dài nhất:** 13.61 giây

### Ví Dụ Samples

#### Sample 1: Tiếng Việt
- **ID:** `083_000007975`
- **Transcript:** `tiến vào thế kỷ hai mươi`
- **Duration:** 5.076 giây
- **Audio:** `audio/083_000007975.wav`

#### Sample 2: Tiếng Anh
- **ID:** `6689-64264-0014`
- **Transcript:** `when turnbull in the narrow space behind the counter would push his way past her without other pretense of apology than something like a sneer she did feel for a moment as if evil were about to have the victory over her`
- **Duration:** 13.44 giây
- **Audio:** `audio/6689-64264-0014.wav`

---

## 🎯 Kết Quả Model trên Test Dataset

### Model Information

| Thông số | Giá trị |
|----------|---------|
| **Checkpoint** | `data/results/checkpoints/checkpoint_epoch_12.pt` |
| **Epoch** | 12 |
| **Training Loss** | 0.5933 |
| **Model Parameters** | 4,132,715 (~4.1M) |
| **Model Size** | ~47 MB |

### Metrics Tổng Quan

#### Accuracy Metrics

| Metric | Giá trị | Mô tả |
|--------|---------|-------|
| **WER (Word Error Rate)** | **47.37%** | Tỷ lệ lỗi từ - chỉ số quan trọng nhất |
| **CER (Character Error Rate)** | **21.04%** | Tỷ lệ lỗi ký tự - quan trọng cho Tiếng Việt |
| **SER (Sentence Error Rate)** | **99.61%** | Tỷ lệ câu có ít nhất một lỗi |
| **Exact Match Accuracy** | **0.41%** | Tỷ lệ câu hoàn toàn đúng |
| **Word-level Accuracy** | **0.39%** | Tỷ lệ câu đúng từng từ |

#### Performance Metrics

| Metric | Giá trị | Đánh giá |
|--------|---------|----------|
| **RTF (Real-Time Factor)** | **0.0007** | ✅ Rất nhanh (1440.9x real-time) |
| **Total Audio Duration** | 227,287.43 giây (~63.1 giờ) | |
| **Total Inference Time** | 157.74 giây (~2.6 phút) | |
| **Tốc độ xử lý** | **1440.9x** thời gian thực | ✅ Xuất sắc |

### Chi Tiết Kết Quả

#### Tổng Số Samples Tested

- **Total samples:** 32,818
- **Exact matches:** 134 (0.41%)
- **Word matches:** 128 (0.39%)
- **Sentences with errors:** 32,690 (99.61%)

#### Phân Tích Lỗi

**WER = 47.37%** có nghĩa là:
- Trung bình, model sai khoảng **47 từ trên 100 từ**
- Đây là mức WER cao, cần cải thiện

**CER = 21.04%** có nghĩa là:
- Trung bình, model sai khoảng **21 ký tự trên 100 ký tự**
- Đối với Tiếng Việt, CER này cho thấy còn nhiều lỗi về dấu và ký tự

**SER = 99.61%** có nghĩa là:
- Hầu hết các câu (99.61%) đều có ít nhất một lỗi
- Chỉ có 0.39% câu hoàn toàn đúng

### Ví Dụ Predictions

#### ✅ Câu Đúng (Exact Match)

**Sample 4:**
- **Ground Truth:** `thiền học truyền thống thường chọn đối tượng thiền quán là các công án thiền`
- **Prediction:** `thiền học truyền thống thường chọn đối tượng thiền quán là các công án thiền`
- **Kết quả:** ✅ Hoàn toàn đúng

#### ❌ Câu Có Lỗi (Typical Errors)

**Sample 1:**
- **Ground Truth:** `tiến vào thế kỷ hai mươi`
- **Prediction:** `tìm vô thế kì ha mốt`
- **Lỗi:** 
  - `tiến` → `tìm` (substitution)
  - `vào` → `vô` (substitution)
  - `kỷ` → `kì` (substitution, mất dấu)
  - `hai mươi` → `ha mốt` (substitution + deletion)

**Sample 2 (English):**
- **Ground Truth:** `when turnbull in the narrow space behind the counter would push his way past her without other pretense of apology than something like a sneer she did feel for a moment as if evil were about to have the victory over her`
- **Prediction:** `whenturnble in the maro space beind the couner woild plush hes way pousher without other preaten to tepoligy then something like a sneer she ded the ol for amoment as if ele worbot to hape the dictory over her`
- **Lỗi:**
  - Missing spaces: `whenturnble`, `amoment`
  - Substitutions: `turnbull` → `turnble`, `narrow` → `maro`, `counter` → `couner`
  - Deletions: `did` → `ded`, `feel` → `ol`, `victory` → `dictory`

**Sample 3:**
- **Ground Truth:** `giống hệt như năm xưa bà nội và mẹ thằng thanh cứ cãi nhau ầm ĩ`
- **Prediction:** `long hẹt tê nong xư bạn nội và mẹ chẳng thanh co cả hau ở mí`
- **Lỗi:** Nhiều substitutions và deletions

### Phân Tích Theo Ngôn Ngữ

Dataset bao gồm cả Tiếng Việt và Tiếng Anh. Dựa trên sample analysis (1,000 samples đầu tiên):

- **Tiếng Việt:** ~83.9% (839/1000 samples)
- **Tiếng Anh:** ~16.1% (161/1000 samples)

**Tổng dataset (32,818 samples):**
- **Tiếng Việt:** Ước tính ~27,500 samples (~84%)
- **Tiếng Anh:** Ước tính ~5,300 samples (~16%)

**Lỗi thường gặp:**
- **Tiếng Việt:**
  - Mất dấu (ví dụ: `kỷ` → `kì`)
  - Substitution từ (ví dụ: `tiến` → `tìm`)
  - Deletion từ (ví dụ: `hai mươi` → `ha mốt`)
  
- **Tiếng Anh:**
  - Missing spaces (ví dụ: `whenturnble`)
  - Spelling errors (ví dụ: `turnbull` → `turnble`)
  - Word substitutions (ví dụ: `narrow` → `maro`)

---

## 📈 So Sánh với Tiêu Chuẩn

### Benchmark Comparison

| Model | WER | CER | RTF | Ghi chú |
|-------|-----|-----|-----|---------|
| **Model hiện tại (Epoch 12)** | **47.37%** | **21.04%** | **0.0007** | Model nhỏ, rất nhanh |
| Whisper Base | ~15-20% | ~5-8% | ~0.1-0.3 | Model lớn, chậm hơn |
| Production ASR | <10% | <3% | <0.5 | Yêu cầu cho production |

### Đánh Giá

#### ✅ Điểm Mạnh

1. **Tốc độ xử lý xuất sắc:**
   - RTF = 0.0007 (1440.9x real-time)
   - Xử lý 63 giờ audio trong chỉ 2.6 phút
   - Phù hợp cho ứng dụng real-time và batch processing

2. **Model nhỏ gọn:**
   - Chỉ ~4.1M parameters
   - Checkpoint size: ~47 MB
   - Dễ deploy trên edge devices

3. **Hiệu quả memory:**
   - Xử lý được batch lớn
   - Inference nhanh và ổn định

#### ⚠️ Điểm Cần Cải Thiện

1. **WER còn cao (47.37%):**
   - Cần thêm dữ liệu training
   - Có thể cần tinh chỉnh hyperparameters
   - Có thể thử các kiến trúc model khác

2. **CER (21.04%):**
   - Đối với Tiếng Việt, cần cải thiện nhận diện dấu
   - Cần tăng cường dữ liệu Tiếng Việt có dấu đầy đủ

3. **SER rất cao (99.61%):**
   - Hầu hết các câu đều có lỗi
   - Cần cải thiện độ chính xác tổng thể

---

## 📁 Files và Outputs

### Input Files

1. **Test Dataset:**
   - `data/processed/test/manifest.csv` - Manifest file với 32,818 samples
   - `data/processed/test/audio/` - 32,891 WAV files (6.9 GB)
   - `data/processed/test/timestamps.json` - Word-level alignments

2. **Model:**
   - `data/results/checkpoints/checkpoint_epoch_12.pt` - Model checkpoint

3. **Vocabulary:**
   - `data/processed/vocab.json` - Vocabulary file (107 tokens)

### Output Files

1. **Predictions:**
   - `experiments/reports/all_predictions_epoch12.json` - Tất cả 32,818 predictions
     - Format: `[{"ground_truth": "...", "prediction": "..."}, ...]`

2. **Metrics:**
   - `experiments/reports/metrics.json` - Metrics summary
     - Chứa WER, CER, SER, RTF và các metrics khác

3. **Visualizations:**
   - `experiments/reports/training_report.png` - Training report
   - `experiments/reports/log_scale_metrics.png` - Log scale metrics
   - `experiments/reports/combined_log_metrics.png` - Combined metrics

4. **Reports:**
   - `experiments/reports/evaluation_report.md` - Báo cáo đánh giá chi tiết
   - `experiments/reports/TEST_DATASET_AND_RESULTS.md` - File này

---

## 🔍 Chi Tiết Kỹ Thuật

### Test Configuration

- **Script:** `scripts/test_full_dataset.py`
- **Batch size:** 16
- **Device:** CUDA
- **Decoder:** Greedy decoding
- **Metrics library:** jiwer

### Test Process

1. Load model từ checkpoint
2. Load vocabulary (107 tokens)
3. Load test dataset từ manifest.csv
4. Process từng batch với batch_size=16
5. Decode predictions với greedy decoding
6. Tính toán metrics (WER, CER, SER, RTF)
7. Lưu tất cả predictions và metrics

### Metrics Calculation

- **WER/CER:** Sử dụng thư viện `jiwer`
- **SER:** Tỷ lệ câu có `ground_truth != prediction` (case-insensitive)
- **RTF:** `total_inference_time / total_audio_duration`
- **Exact Match:** `ground_truth.lower() == prediction.lower()`
- **Word-level Accuracy:** Tỷ lệ câu đúng từng từ (case-insensitive)

---

## 📊 Thống Kê Chi Tiết

### Audio Duration Distribution

- **Tổng thời lượng:** 227,287.43 giây (~63.1 giờ)
- **Trung bình:** 6.98 giây/sample
- **Tối thiểu:** 5.02 giây
- **Tối đa:** 13.61 giây
- **Tổng số samples:** 32,818

### Inference Performance

- **Tổng thời gian inference:** 157.74 giây (~2.6 phút)
- **RTF:** 0.0007 (1440.9x real-time)
- **Throughput:** ~208 samples/giây
- **Average time per sample:** ~4.8 ms

### Error Distribution

- **Sentences with errors:** 32,690 (99.61%)
- **Perfect matches:** 134 (0.41%)
- **Word-level matches:** 128 (0.39%)

---

## 🎯 Khuyến Nghị

### Ngắn Hạn

1. ✅ Model đã sẵn sàng cho demo và testing
2. ⚠️ Cần cải thiện độ chính xác trước khi deploy production

### Dài Hạn

1. **Tăng dữ liệu training:**
   - Đặc biệt là dữ liệu Tiếng Việt có dấu đầy đủ
   - Cân bằng hơn giữa Tiếng Việt và Tiếng Anh

2. **Data augmentation:**
   - Thêm noise, speed variation, pitch shift
   - SpecAugment cho audio features

3. **Fine-tuning:**
   - Tinh chỉnh trên domain cụ thể nếu có
   - Transfer learning từ model lớn hơn

4. **Model improvements:**
   - Thử các kiến trúc khác (Transformer, Conformer)
   - Beam search decoding thay vì greedy
   - Language model integration

5. **Ensemble:**
   - Kết hợp nhiều model để giảm WER
   - Voting hoặc weighted averaging

---

## 📝 Ghi Chú

- Tất cả predictions được lưu trong `all_predictions_epoch12.json`
- Metrics được tính toán tự động bằng `jiwer`
- Test được chạy trên GPU (CUDA) với batch_size=16
- Model sử dụng greedy decoding (có thể cải thiện bằng beam search)

---

*Báo cáo được tạo tự động từ kết quả test và dataset information.*

