"""
View detailed test results từ log file
"""
import json
import argparse
from pathlib import Path


def view_results(log_file, show_all=False, show_worst=10, show_best=10):
    """View test results"""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print("=" * 80)
    print("📊 TEST RESULTS VIEWER")
    print("=" * 80)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    summary = data.get("summary", {})
    
    print(f"\n📁 Log file: {log_file}")
    print(f"📅 Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"🤖 Model: {data.get('model_path', 'unknown')}")
    print(f"📊 Total samples: {len(results)}")
    print()
    
    if summary:
        print("=" * 80)
        print("📈 SUMMARY METRICS")
        print("=" * 80)
        print(f"Overall WER: {summary.get('overall_wer', 0):.4f} ({summary.get('overall_wer', 0)*100:.2f}%)")
        print(f"Overall CER: {summary.get('overall_cer', 0):.4f} ({summary.get('overall_cer', 0)*100:.2f}%)")
        print(f"Average WER: {summary.get('average_wer', 0):.4f} ({summary.get('average_wer', 0)*100:.2f}%)")
        print(f"Average CER: {summary.get('average_cer', 0):.4f} ({summary.get('average_cer', 0)*100:.2f}%)")
        print(f"Perfect matches: {summary.get('perfect_matches', 0)}/{summary.get('total_samples', 0)} ({summary.get('perfect_match_rate', 0)*100:.2f}%)")
    
    if show_all:
        print("\n" + "=" * 80)
        print("📝 ALL RESULTS")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            match_icon = "✅" if result["match"] else "❌"
            print(f"\n{i}. Sample {result['sample_id']} {match_icon} WER: {result['wer']:.4f} CER: {result['cer']:.4f}")
            print(f"   Audio: {result['audio_path']}")
            print(f"   Language: {result.get('language', 'unknown')}")
            print(f"   Reference: {result['reference']}")
            print(f"   Prediction: {result['prediction']}")
    
    # Show worst samples
    if show_worst > 0:
        worst = sorted(results, key=lambda x: x["wer"], reverse=True)[:show_worst]
        print("\n" + "=" * 80)
        print(f"🔴 TOP {show_worst} WORST SAMPLES (Highest WER)")
        print("=" * 80)
        for i, result in enumerate(worst, 1):
            print(f"\n{i}. Sample {result['sample_id']} | WER: {result['wer']:.4f} | CER: {result['cer']:.4f}")
            print(f"   Audio: {result['audio_path']}")
            print(f"   Language: {result.get('language', 'unknown')}")
            print(f"   Reference: {result['reference']}")
            print(f"   Prediction: {result['prediction']}")
    
    # Show best samples
    if show_best > 0:
        best = sorted(results, key=lambda x: x["wer"])[:show_best]
        print("\n" + "=" * 80)
        print(f"🟢 TOP {show_best} BEST SAMPLES (Lowest WER)")
        print("=" * 80)
        for i, result in enumerate(best, 1):
            match_icon = "✅" if result["match"] else "⭐"
            print(f"\n{i}. Sample {result['sample_id']} {match_icon} | WER: {result['wer']:.4f} | CER: {result['cer']:.4f}")
            print(f"   Audio: {result['audio_path']}")
            print(f"   Language: {result.get('language', 'unknown')}")
            print(f"   Reference: {result['reference']}")
            print(f"   Prediction: {result['prediction']}")


def main():
    parser = argparse.ArgumentParser(description="View test results")
    parser.add_argument(
        "--log",
        default="whisper_full_test_bf16_log.json",
        help="Log file path"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all results"
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=10,
        help="Number of worst samples to show"
    )
    parser.add_argument(
        "--best",
        type=int,
        default=10,
        help="Number of best samples to show"
    )
    
    args = parser.parse_args()
    
    view_results(
        log_file=args.log,
        show_all=args.all,
        show_worst=args.worst,
        show_best=args.best
    )


if __name__ == "__main__":
    main()

