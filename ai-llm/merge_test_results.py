"""
Gộp tất cả kết quả test từ các parallel processes thành một file tổng hợp
"""
import json
import argparse
from pathlib import Path
from glob import glob
from datetime import datetime
from collections import defaultdict


def merge_results(input_patterns, output_file):
    """Gộp tất cả kết quả từ các file JSON"""
    all_results = []
    all_summaries = []
    
    # Tìm tất cả các file JSON
    json_files = []
    for pattern in input_patterns:
        json_files.extend(glob(pattern))
    
    json_files = sorted(set(json_files))
    
    if not json_files:
        print(f"❌ Không tìm thấy file nào khớp với pattern: {input_patterns}")
        return
    
    print(f"📁 Tìm thấy {len(json_files)} file kết quả:")
    for f in json_files:
        print(f"   - {f}")
    
    # Đọc tất cả các file
    total_samples = 0
    for json_file in json_files:
        print(f"\n📖 Đang đọc: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get("results", [])
            summary = data.get("summary", {})
            
            all_results.extend(results)
            if summary:
                all_summaries.append(summary)
            
            total_samples += len(results)
            print(f"   ✅ Đã thêm {len(results)} samples")
        except Exception as e:
            print(f"   ❌ Lỗi khi đọc {json_file}: {e}")
    
    # Tính toán tổng hợp
    print(f"\n📊 Đang tính toán tổng hợp từ {total_samples} samples...")
    
    total_wer = 0.0
    total_cer = 0.0
    perfect_matches = 0
    wer_values = []
    cer_values = []
    language_stats = defaultdict(int)
    
    for result in all_results:
        wer = result.get("wer", 0.0)
        cer = result.get("cer", 0.0)
        total_wer += wer
        total_cer += cer
        wer_values.append(wer)
        cer_values.append(cer)
        
        if result.get("match", False):
            perfect_matches += 1
        
        lang = result.get("language", "unknown")
        language_stats[lang] += 1
    
    # Tính toán thống kê
    avg_wer = total_wer / len(all_results) if all_results else 0.0
    avg_cer = total_cer / len(all_results) if all_results else 0.0
    perfect_match_rate = perfect_matches / len(all_results) if all_results else 0.0
    
    # Tính percentile
    wer_values_sorted = sorted(wer_values)
    cer_values_sorted = sorted(cer_values)
    
    def percentile(data, p):
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(data):
            return data[f] * (1 - c) + data[f + 1] * c
        return data[f]
    
    merged_summary = {
        "total_samples": len(all_results),
        "overall_wer": avg_wer,
        "overall_cer": avg_cer,
        "average_wer": avg_wer,
        "average_cer": avg_cer,
        "perfect_matches": perfect_matches,
        "perfect_match_rate": perfect_match_rate,
        "wer_stats": {
            "mean": avg_wer,
            "median": percentile(wer_values_sorted, 50),
            "p95": percentile(wer_values_sorted, 95),
            "p99": percentile(wer_values_sorted, 99),
            "min": min(wer_values) if wer_values else 0.0,
            "max": max(wer_values) if wer_values else 0.0,
        },
        "cer_stats": {
            "mean": avg_cer,
            "median": percentile(cer_values_sorted, 50),
            "p95": percentile(cer_values_sorted, 95),
            "p99": percentile(cer_values_sorted, 99),
            "min": min(cer_values) if cer_values else 0.0,
            "max": max(cer_values) if cer_values else 0.0,
        },
        "language_distribution": dict(language_stats),
    }
    
    # Tạo file tổng hợp
    merged_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": all_summaries[0].get("model_path", "unknown") if all_summaries else "unknown",
        "device": all_summaries[0].get("device", "unknown") if all_summaries else "unknown",
        "compute_type": all_summaries[0].get("compute_type", "unknown") if all_summaries else "unknown",
        "source_files": json_files,
        "summary": merged_summary,
        "results": all_results,
    }
    
    # Lưu file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã gộp thành công!")
    print(f"📁 File tổng hợp: {output_file}")
    print(f"📊 Tổng số samples: {len(all_results)}")
    print(f"\n📈 THỐNG KÊ TỔNG HỢP:")
    print(f"   WER trung bình: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"   CER trung bình: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"   Perfect matches: {perfect_matches}/{len(all_results)} ({perfect_match_rate*100:.2f}%)")
    print(f"\n   WER - Median: {merged_summary['wer_stats']['median']:.4f}, P95: {merged_summary['wer_stats']['p95']:.4f}, P99: {merged_summary['wer_stats']['p99']:.4f}")
    print(f"   CER - Median: {merged_summary['cer_stats']['median']:.4f}, P95: {merged_summary['cer_stats']['p95']:.4f}, P99: {merged_summary['cer_stats']['p99']:.4f}")
    
    if language_stats:
        print(f"\n🌍 Phân bố ngôn ngữ:")
        for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {lang}: {count} ({count/len(all_results)*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Gộp kết quả test từ các parallel processes")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["test_results_parallel/whisper_test_bf16_part*.json"],
        help="Pattern(s) để tìm các file JSON kết quả"
    )
    parser.add_argument(
        "--output",
        default="test_results_parallel/whisper_test_bf16_merged.json",
        help="File output tổng hợp"
    )
    
    args = parser.parse_args()
    
    merge_results(args.inputs, args.output)


if __name__ == "__main__":
    main()

