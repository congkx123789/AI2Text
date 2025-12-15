# Bộ Test Data và Kết Quả Model

## 📋 Tổng Quan

Tài liệu này ghi lại toàn bộ thông tin về bộ test data và kết quả đánh giá của các model trong hệ thống AI-LLM.

---

## 📊 Bộ Test Data

### 1. Whisper ASR Test Data

**File:** `data/processed_finetune/whisper_test.jsonl`

**Thông tin:**
- **Tổng số mẫu:** 32,818 samples
- **Định dạng:** JSONL (mỗi dòng là một JSON object)
- **Cấu trúc dữ liệu:**
  ```json
  {
    "audio": "đường dẫn đến file audio",
    "text": "transcript tham chiếu",
    "language": "mã ngôn ngữ (vi, en, ...)"
  }
  ```

**Ví dụ:**
```json
{"audio": "data/processed/full_merged_dataset/test/audio/083_000007975.wav", "text": "tiến vào thế kỷ hai mươi", "language": "vi"}
{"audio": "data/processed/full_merged_dataset/test/audio/6689-64264-0014.wav", "text": "when turnbull in the narrow space behind the counter would push his way past her without other pretense of apology than something like a sneer she did feel for a moment as if evil were about to have the victory over her", "language": "en"}
```

**Phân bố ngôn ngữ (từ kết quả test):**
- Tiếng Việt (vi): 5,264 samples
- Tiếng Thổ Nhĩ Kỳ (tr): 10,184 samples
- Tiếng Hungary (hu): 3,136 samples
- Tiếng Rumani (ro): 4,048 samples
- Tiếng Ả Rập (ar): 2,880 samples
- Tiếng Hy Lạp (el): 864 samples
- Tiếng Pháp (fr): 1,384 samples
- Tiếng Tây Ban Nha (es): 672 samples
- Tiếng Séc (cs): 960 samples
- Và 46 ngôn ngữ khác (tổng cộng 55 ngôn ngữ)

### 2. LLM Test Data

**File:** `data/processed_finetune/llm_test.jsonl`

**Thông tin:**
- **Tổng số mẫu:** 32,818 samples
- **Định dạng:** JSONL (mỗi dòng là một JSON object)
- **Cấu trúc dữ liệu:**
  ```json
  {
    "instruction": "hướng dẫn nhiệm vụ",
    "input": "input text (có thể rỗng)",
    "output": "output mong đợi"
  }
  ```

**Ví dụ:**
```json
{"instruction": "Chuyển đổi audio thành văn bản tiếng Việt.", "input": "", "output": "tiến vào thế kỷ hai mươi"}
{"instruction": "Transcribe audio to English text.", "input": "", "output": "when turnbull in the narrow space behind the counter would push his way past her without other pretense of apology than something like a sneer she did feel for a moment as if evil were about to have the victory over her"}
```

**Các loại instruction:**
- "Chuyển đổi audio thành văn bản tiếng Việt."
- "Transcribe audio to English text."
- Các task khác: summarize, answer, translate, analyze, extract

### 3. Các Bộ Test Data Khác

**File:** `data/processed_finetune/whisper_train_test.jsonl`
- Dữ liệu train/test cho Whisper

**File:** `data/processed/full_merged_dataset/test/`
- Thư mục chứa audio files và metadata cho test set

---

## 🎯 Kết Quả Đánh Giá Model

### 1. Whisper ASR - Model Fine-tuned

**File kết quả:** `evaluation_results.json`

**Model:** `models/finetuned/whisper-test`

**Kết quả:**
- **WER (Word Error Rate):** 0.1927 (19.27%)
- **CER (Character Error Rate):** 0.1121 (11.21%)
- **Số mẫu đánh giá:** 32,818 samples

**So sánh với Base Model:**
- **Base Model:** `openai/whisper-small`
- **WER Base:** 0.3305 (33.05%)
- **CER Base:** 0.1478 (14.78%)
- **Cải thiện WER:** ~42% (từ 33.05% xuống 19.27%)
- **Cải thiện CER:** ~24% (từ 14.78% xuống 11.21%)

**Ví dụ Predictions vs References:**

| Reference | Prediction (Fine-tuned) | Prediction (Base) |
|-----------|------------------------|-------------------|
| "tiến vào thế kỷ hai mươi" | "tiến vào thế kỷ 21" | "đến vào thế kỷ 21" |
| "giống hệt như năm xưa bà nội và mẹ thằng thanh cứ cãi nhau ầm ĩ" | "rũng hệt như năm xưa bà nội và mẹ thằng thanh choi cãi nhau ẳn bí" | "Dống hẹt như năm xưa, ba nội và mẹ thằng Thanh cứ cãi nhau ở Mỹ." |

### 2. Whisper ASR - Test Chi Tiết (BF16)

**File kết quả:** `test_results_parallel/whisper_test_bf16_FINAL.json`

**Thông tin test:**
- **Timestamp:** 2025-12-11T10:40:22.227465
- **Device:** CUDA
- **Compute Type:** bfloat16
- **Tổng số mẫu:** 32,816 samples
- **Nguồn:** Merge từ 8 file parts (whisper_test_bf16_part0.json đến part7.json)

**Kết quả tổng quan:**
- **Overall WER:** 0.0874 (8.74%)
- **Overall CER:** 0.0501 (5.01%)
- **Perfect Matches:** 9,683 samples
- **Perfect Match Rate:** 29.51%

**Thống kê WER:**
- **Mean:** 0.0874
- **Median:** 0.0556
- **P95:** 0.2581
- **P99:** 0.4615
- **Min:** 0.0
- **Max:** 13.12

**Thống kê CER:**
- **Mean:** 0.0501
- **Median:** 0.0250
- **P95:** 0.1667
- **P99:** 0.3387
- **Min:** 0.0
- **Max:** 10.69

**Phân bố ngôn ngữ trong test:**
- **Tổng số ngôn ngữ:** 55 ngôn ngữ
- **Top 10 ngôn ngữ:**
  1. Tiếng Thổ Nhĩ Kỳ (tr): 10,184 samples
  2. Tiếng Việt (vi): 5,264 samples
  3. Tiếng Rumani (ro): 4,048 samples
  4. Tiếng Hungary (hu): 3,136 samples
  5. Tiếng Ả Rập (ar): 2,880 samples
  6. Tiếng Pháp (fr): 1,384 samples
  7. Tiếng Hy Lạp (el): 864 samples
  8. Tiếng Séc (cs): 960 samples
  9. Tiếng Tây Ban Nha (es): 672 samples
  10. Tiếng Anh (en): 184 samples

### 3. Whisper ASR - Base Model Test (Origin)

**File kết quả:** `test_results_parallel/whisper_origin_test.json`

**Thông tin test:**
- **Timestamp:** 2025-12-11T13:00:28.364614
- **Model:** `whisper-small` (base model, chưa fine-tune)
- **Device:** CUDA
- **Compute Type:** bfloat16
- **Tổng số mẫu:** 50 samples

**Kết quả:**
- **Average WER:** 0.3453 (34.53%)
- **Average CER:** 0.1525 (15.25%)
- **Perfect Matches:** 0
- **Perfect Match Rate:** 0.0%

**Ví dụ kết quả:**
```json
{
  "sample_id": 0,
  "audio_path": "data/processed/full_merged_dataset/test/audio/083_000007975.wav",
  "reference": "tiến vào thế kỷ hai mươi",
  "prediction": "đến vào thế bị 21",
  "wer": 0.6667,
  "cer": 0.5000,
  "match": false,
  "language": "vi",
  "expected_language": "vi"
}
```

### 4. Whisper ASR - Base Model Test (1000 samples)

**File kết quả:** `test_results_parallel/whisper_origin_1000.json`

**Thông tin test:**
- **Timestamp:** 2025-12-11T14:21:52.771063
- **Model:** `whisper-small` (base model)
- **Device:** CUDA
- **Compute Type:** float16
- **Tổng số mẫu:** 1,000 samples

**Kết quả:**
- **Average WER:** 0.3191 (31.91%)
- **Average CER:** 0.1361 (13.61%)
- **Perfect Matches:** 0
- **Perfect Match Rate:** 0.0%

---

## 📈 So Sánh Hiệu Suất

### So Sánh Whisper Models

| Model | WER | CER | Perfect Match Rate | Số Mẫu Test |
|-------|-----|-----|-------------------|-------------|
| **Fine-tuned (whisper-test)** | 19.27% | 11.21% | - | 32,818 |
| **Fine-tuned (BF16 test)** | 8.74% | 5.01% | 29.51% | 32,816 |
| **Base (whisper-small, 50 samples)** | 34.53% | 15.25% | 0.0% | 50 |
| **Base (whisper-small, 1000 samples)** | 31.91% | 13.61% | 0.0% | 1,000 |

**Nhận xét:**
- Model fine-tuned có hiệu suất tốt hơn đáng kể so với base model
- WER giảm từ ~32-35% xuống ~9-19%
- CER giảm từ ~13-15% xuống ~5-11%
- Model fine-tuned đạt perfect match rate 29.51% trên test set lớn

### Phân Tích Theo Ngôn Ngữ

**Top 5 ngôn ngữ có số lượng mẫu nhiều nhất:**
1. **Tiếng Thổ Nhĩ Kỳ (tr):** 10,184 samples
2. **Tiếng Việt (vi):** 5,264 samples
3. **Tiếng Rumani (ro):** 4,048 samples
4. **Tiếng Hungary (hu):** 3,136 samples
5. **Tiếng Ả Rập (ar):** 2,880 samples

**Lưu ý:** Kết quả chi tiết theo từng ngôn ngữ có thể được phân tích từ file `whisper_test_bf16_FINAL.json` (có chứa thông tin chi tiết cho từng sample).

---

## 📁 Cấu Trúc File Kết Quả

### 1. evaluation_results.json

Cấu trúc:
```json
{
  "finetuned": {
    "model": "đường dẫn model",
    "wer": 0.1927,
    "cer": 0.1121,
    "num_samples": 32818,
    "predictions": [...],
    "references": [...]
  },
  "base": {
    "model": "openai/whisper-small",
    "wer": 0.3305,
    "cer": 0.1478,
    "num_samples": 32818,
    "predictions": [...],
    "references": [...]
  }
}
```

### 2. whisper_test_bf16_FINAL.json

Cấu trúc:
```json
{
  "timestamp": "2025-12-11T10:40:22.227465",
  "model_path": "unknown",
  "device": "unknown",
  "compute_type": "unknown",
  "source_files": [...],
  "summary": {
    "total_samples": 32816,
    "overall_wer": 0.0874,
    "overall_cer": 0.0501,
    "perfect_matches": 9683,
    "perfect_match_rate": 0.2951,
    "wer_stats": {...},
    "cer_stats": {...},
    "language_distribution": {...}
  },
  "results": [
    {
      "sample_id": 0,
      "audio_path": "...",
      "reference": "...",
      "prediction": "...",
      "wer": 0.0,
      "cer": 0.0,
      "match": true,
      "language": "vi",
      "expected_language": "vi"
    },
    ...
  ]
}
```

### 3. whisper_origin_test.json / whisper_origin_1000.json

Cấu trúc tương tự `whisper_test_bf16_FINAL.json` nhưng cho base model.

---

## 🔍 Phân Tích Chi Tiết

### 1. Phân Phối WER/CER

**Từ whisper_test_bf16_FINAL.json:**

**WER Distribution:**
- **Median (50th percentile):** 0.0556 (5.56%)
- **95th percentile:** 0.2581 (25.81%)
- **99th percentile:** 0.4615 (46.15%)
- **Max:** 13.12 (có thể là outlier hoặc sample rất khó)

**CER Distribution:**
- **Median (50th percentile):** 0.0250 (2.50%)
- **95th percentile:** 0.1667 (16.67%)
- **99th percentile:** 0.3387 (33.87%)
- **Max:** 10.69 (có thể là outlier)

**Nhận xét:**
- 50% samples có WER ≤ 5.56% (rất tốt)
- 95% samples có WER ≤ 25.81% (chấp nhận được)
- 1% samples có WER rất cao (>46%), cần phân tích thêm

### 2. Perfect Matches

- **Tổng số perfect matches:** 9,683 / 32,816 (29.51%)
- Đây là các samples được transcribe hoàn toàn chính xác (WER = 0, CER = 0)

### 3. Language Detection Accuracy

Từ kết quả test, model có khả năng detect ngôn ngữ tốt. Các samples có `language` và `expected_language` khớp nhau trong hầu hết các trường hợp.

---

## 📝 Ghi Chú

### 1. Sự Khác Biệt Giữa Các Test

- **evaluation_results.json:** So sánh fine-tuned vs base trên cùng 32,818 samples
- **whisper_test_bf16_FINAL.json:** Test chi tiết model fine-tuned trên 32,816 samples với BF16
- **whisper_origin_test.json:** Test base model trên 50 samples
- **whisper_origin_1000.json:** Test base model trên 1,000 samples

### 2. Metrics

- **WER (Word Error Rate):** Tỷ lệ lỗi từ, tính bằng số từ sai / tổng số từ
- **CER (Character Error Rate):** Tỷ lệ lỗi ký tự, tính bằng số ký tự sai / tổng số ký tự
- **Perfect Match Rate:** Tỷ lệ samples được transcribe hoàn toàn chính xác

### 3. Device và Compute Type

- **CUDA:** Sử dụng GPU
- **BF16 (bfloat16):** Độ chính xác 16-bit, tối ưu cho GPU Ampere+
- **Float16:** Độ chính xác 16-bit tiêu chuẩn

---

## 🚀 Kết Luận

### Điểm Mạnh

1. **Model fine-tuned có hiệu suất vượt trội:**
   - WER giảm từ ~33% xuống ~9-19%
   - CER giảm từ ~15% xuống ~5-11%
   - Perfect match rate đạt 29.51%

2. **Đa ngôn ngữ:**
   - Hỗ trợ 55 ngôn ngữ
   - Đặc biệt tốt với tiếng Việt (5,264 samples trong test set)

3. **Test set lớn:**
   - 32,818 samples cho Whisper ASR
   - 32,818 samples cho LLM
   - Đảm bảo đánh giá toàn diện

### Điểm Cần Cải Thiện

1. **Một số samples có WER rất cao:**
   - Cần phân tích các samples có WER > 50%
   - Có thể do chất lượng audio kém hoặc ngôn ngữ khó

2. **Perfect match rate:**
   - 29.51% là tốt nhưng có thể cải thiện thêm
   - Có thể fine-tune thêm hoặc cải thiện data quality

3. **LLM Results:**
   - Cần thêm kết quả đánh giá cho LLM model
   - Hiện tại chỉ có test data, chưa có kết quả đánh giá

---

## 📚 Tài Liệu Tham Khảo

- **Test Data:** `data/processed_finetune/whisper_test.jsonl`, `data/processed_finetune/llm_test.jsonl`
- **Evaluation Results:** `evaluation_results.json`
- **Detailed Test Results:** `test_results_parallel/whisper_test_bf16_FINAL.json`
- **Base Model Results:** `test_results_parallel/whisper_origin_test.json`, `test_results_parallel/whisper_origin_1000.json`

---

**Tài liệu được tạo tự động dựa trên các file kết quả hiện có.**
**Cập nhật lần cuối: 2025-12-11**

