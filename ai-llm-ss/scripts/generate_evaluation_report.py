#!/usr/bin/env python3
"""
Tạo báo cáo đánh giá model dạng text và markdown.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def load_metrics(metrics_path):
    """Load metrics từ file."""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_number(num, decimals=2):
    """Format số với dấu phẩy."""
    if isinstance(num, float):
        return f"{num:.{decimals}f}"
    return f"{num:,}"

def generate_markdown_report(metrics_path, checkpoint_dir, output_path=None):
    """Tạo báo cáo markdown."""
    
    metrics_data = load_metrics(metrics_path)
    metrics = metrics_data['metrics']
    
    # Tính toán thêm một số chỉ số
    total_samples = metrics['total_samples']
    exact_matches = metrics.get('exact_matches', 0)
    word_matches = metrics.get('word_matches', 0)
    
    # Tạo markdown report
    report = f"""# Báo Cáo Đánh Giá Model ASR

**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Thông Tin Model

- **Checkpoint:** `{metrics_data['checkpoint']}`
- **Epoch:** {metrics_data['epoch']}
- **Training Loss:** {metrics_data['loss']:.4f}
- **Tổng số tham số:** ~4.1M

---

## Kết Quả Test trên Tập Dữ Liệu Độc Lập

### Chỉ Số Độ Chính Xác (Accuracy Metrics)

| Chỉ Số | Giá Trị | Mô Tả |
|--------|--------|-------|
| **WER (Word Error Rate)** | **{metrics['wer']*100:.2f}%** | Tỷ lệ lỗi từ - chỉ số quan trọng nhất |
| **CER (Character Error Rate)** | **{metrics['cer']*100:.2f}%** | Tỷ lệ lỗi ký tự - quan trọng cho Tiếng Việt |
| **SER (Sentence Error Rate)** | **{metrics['sentence_error_rate']:.2f}%** | Tỷ lệ câu có lỗi |
| **Exact Match Accuracy** | **{metrics['exact_accuracy']:.2f}%** | Tỷ lệ câu hoàn toàn đúng |
| **Word-level Accuracy** | **{metrics['word_accuracy']:.2f}%** | Tỷ lệ câu đúng từng từ |

### Chi Tiết

- **Tổng số mẫu test:** {format_number(total_samples)}
- **Số câu hoàn toàn đúng:** {format_number(exact_matches)} ({metrics['exact_accuracy']:.2f}%)
- **Số câu đúng từng từ:** {format_number(word_matches)} ({metrics['word_accuracy']:.2f}%)
- **Số câu có lỗi:** {format_number(metrics['sentence_errors'])} ({metrics['sentence_error_rate']:.2f}%)

---

## Chỉ Số Hiệu Năng (Performance Metrics)

| Chỉ Số | Giá Trị | Đánh Giá |
|--------|--------|----------|
| **RTF (Real-Time Factor)** | **{metrics['rtf']:.4f}** | {'✅ Rất nhanh (RTF < 1)' if metrics['rtf'] < 1 else '⚠️ Chậm hơn thời gian thực'} |
| **Tổng thời lượng audio:** | {metrics['total_audio_seconds']/3600:.2f} giờ | |
| **Tổng thời gian inference:** | {metrics['total_inference_seconds']/60:.2f} phút | |
| **Tốc độ xử lý:** | ~{metrics['total_audio_seconds']/metrics['total_inference_seconds']:.1f}x thời gian thực | |

---

## Phân Tích Kết Quả

### Điểm Mạnh

1. **Tốc độ xử lý xuất sắc:** RTF = {metrics['rtf']:.4f} cho thấy model xử lý nhanh hơn thời gian thực rất nhiều ({metrics['total_audio_seconds']/metrics['total_inference_seconds']:.1f}x), phù hợp cho ứng dụng real-time.

2. **Model nhỏ gọn:** ~4.1M tham số, dễ triển khai trên thiết bị edge hoặc mobile.

### Điểm Cần Cải Thiện

1. **WER còn cao ({metrics['wer']*100:.2f}%):** Model còn mắc nhiều lỗi nhận diện từ. Cần:
   - Tăng thêm dữ liệu training
   - Tinh chỉnh hyperparameters
   - Thử các kiến trúc model khác

2. **CER ({metrics['cer']*100:.2f}%):** Đối với Tiếng Việt, CER này cho thấy còn nhiều lỗi về dấu và ký tự. Cần:
   - Tăng cường dữ liệu Tiếng Việt có dấu
   - Cải thiện tokenizer

3. **SER rất cao ({metrics['sentence_error_rate']:.2f}%):** Hầu hết các câu đều có ít nhất một lỗi. Điều này phù hợp với WER cao.

---

## Khuyến Nghị

### Ngắn Hạn
- ✅ Model đã sẵn sàng cho demo và testing
- ⚠️ Cần cải thiện độ chính xác trước khi deploy production

### Dài Hạn
1. **Tăng dữ liệu training:** Đặc biệt là dữ liệu Tiếng Việt có dấu đầy đủ
2. **Data augmentation:** Thêm noise, speed variation, pitch shift
3. **Fine-tuning:** Tinh chỉnh trên domain cụ thể nếu có
4. **Ensemble:** Kết hợp nhiều model để giảm WER

---

## So Sánh với Tiêu Chuẩn Ngành

| Model | WER | CER | RTF | Ghi Chú |
|-------|-----|-----|-----|---------|
| **Model hiện tại** | **{metrics['wer']*100:.2f}%** | **{metrics['cer']*100:.2f}%** | **{metrics['rtf']:.4f}** | Model nhỏ, nhanh |
| Whisper Base | ~15-20% | ~5-8% | ~0.1-0.3 | Model lớn, chậm hơn |
| Production ASR | <10% | <3% | <0.5 | Yêu cầu cho production |

---

*Báo cáo được tạo tự động từ kết quả test.*
"""
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "experiments" / "reports" / "evaluation_report.md"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Báo cáo markdown đã được lưu tại: {output_path}")
    
    # In summary ra console
    print("\n" + "="*80)
    print("TÓM TẮT KẾT QUẢ ĐÁNH GIÁ MODEL")
    print("="*80)
    print(f"\n📊 Chỉ Số Chính:")
    print(f"   • WER:  {metrics['wer']*100:.2f}%")
    print(f"   • CER:  {metrics['cer']*100:.2f}%")
    print(f"   • SER:  {metrics['sentence_error_rate']:.2f}%")
    print(f"   • RTF:  {metrics['rtf']:.4f} ({metrics['total_audio_seconds']/metrics['total_inference_seconds']:.1f}x real-time)")
    print(f"\n📈 Chi Tiết:")
    print(f"   • Tổng mẫu test: {format_number(total_samples)}")
    print(f"   • Câu đúng hoàn toàn: {format_number(exact_matches)} ({metrics['exact_accuracy']:.2f}%)")
    print(f"   • Câu đúng từng từ: {format_number(word_matches)} ({metrics['word_accuracy']:.2f}%)")
    print(f"\n✅ Điểm Mạnh: Tốc độ xử lý rất nhanh ({metrics['total_audio_seconds']/metrics['total_inference_seconds']:.1f}x real-time)")
    print(f"⚠️  Cần Cải Thiện: WER còn cao ({metrics['wer']*100:.2f}%), cần thêm dữ liệu training")
    print("="*80 + "\n")
    
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tạo báo cáo đánh giá model")
    parser.add_argument("--metrics", default="experiments/reports/metrics.json",
                      help="File metrics từ test")
    parser.add_argument("--output", default="experiments/reports/evaluation_report.md",
                      help="Đường dẫn lưu báo cáo markdown")
    args = parser.parse_args()
    
    generate_markdown_report(args.metrics, None, args.output)

if __name__ == "__main__":
    main()

