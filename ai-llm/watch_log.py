"""
Watch log file và hiển thị kết quả mới nhất real-time
"""
import json
import time
from pathlib import Path
import sys


def watch_log(log_file="whisper_full_test_bf16_log.json", show_count=5):
    """Watch log file và hiển thị kết quả mới nhất"""
    log_path = Path(log_file)
    
    print("=" * 80)
    print("👀 WATCHING TEST LOG - REAL-TIME")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Showing latest {show_count} predictions")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    last_count = 0
    
    try:
        while True:
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    results = data.get("results", [])
                    current = len(results)
                    total = data.get("total_samples", 0)
                    progress = data.get("current_progress", {})
                    metrics = data.get("current_metrics", {})
                    
                    if current > last_count:
                        # Clear screen
                        print("\033[2J\033[H", end="")
                        
                        print("=" * 80)
                        print("📊 TEST PROGRESS - REAL-TIME")
                        print("=" * 80)
                        print(f"Completed: {current}/{total} ({progress.get('percentage', 0):.2f}%)")
                        
                        if metrics:
                            print(f"Average WER: {metrics.get('average_wer', 0):.4f} ({metrics.get('average_wer', 0)*100:.2f}%)")
                            print(f"Average CER: {metrics.get('average_cer', 0):.4f} ({metrics.get('average_cer', 0)*100:.2f}%)")
                            print(f"Perfect matches: {metrics.get('perfect_matches', 0)}/{current} ({metrics.get('perfect_match_rate', 0)*100:.2f}%)")
                        
                        print()
                        print("=" * 80)
                        print(f"📝 LATEST {show_count} PREDICTIONS")
                        print("=" * 80)
                        
                        # Show latest predictions
                        for i, result in enumerate(results[-show_count:], 1):
                            match_icon = "✅" if result["match"] else "❌"
                            print(f"\n{i}. Sample {result['sample_id']} {match_icon} | WER: {result['wer']:.4f} | CER: {result['cer']:.4f}")
                            print(f"   Language: {result.get('language', 'unknown')}")
                            print(f"   Reference:  {result['reference']}")
                            print(f"   Prediction: {result['prediction']}")
                            print(f"   Audio: {result['audio_path']}")
                        
                        last_count = current
                        
                        if current >= total:
                            print("\n" + "=" * 80)
                            print("✅ TEST COMPLETED!")
                            print("=" * 80)
                            break
                    
                except (json.JSONDecodeError, KeyError) as e:
                    pass  # File might be incomplete or being written
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\n\n👀 Watching stopped")
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"\nFinal count: {len(data.get('results', []))}/{data.get('total_samples', 0)}")
            except:
                pass


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "whisper_full_test_bf16_log.json"
    show_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    watch_log(log_file, show_count)

