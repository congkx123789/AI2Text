"""
Monitor test progress và hiển thị kết quả từng sample
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta


def format_time(seconds):
    """Format seconds to human readable"""
    return str(timedelta(seconds=int(seconds)))


def monitor_test(log_file="whisper_full_test_bf16_log.json"):
    """Monitor test progress"""
    log_path = Path(log_file)
    
    print("=" * 80)
    print("📊 MONITORING TEST PROGRESS")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()
    
    last_count = 0
    start_time = None
    
    try:
        while True:
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    results = data.get("results", [])
                    total = data.get("total_samples", 0)
                    current = len(results)
                    
                    if current > 0:
                        if start_time is None:
                            start_time = time.time()
                        
                        # Calculate progress
                        progress = (current / total * 100) if total > 0 else 0
                        elapsed = time.time() - start_time if start_time else 0
                        
                        if current > last_count:
                            # Calculate speed
                            speed = current / elapsed if elapsed > 0 else 0
                            remaining = (total - current) / speed if speed > 0 else 0
                            
                            # Calculate metrics
                            avg_wer = sum(r["wer"] for r in results) / len(results)
                            avg_cer = sum(r["cer"] for r in results) / len(results)
                            perfect = sum(1 for r in results if r["match"])
                            
                            # Clear and print
                            print("\033[2J\033[H", end="")  # Clear screen
                            print("=" * 80)
                            print("📊 TEST PROGRESS")
                            print("=" * 80)
                            print(f"Progress: {current}/{total} ({progress:.2f}%)")
                            print(f"Speed: {speed:.2f} samples/sec")
                            print(f"Elapsed: {format_time(elapsed)}")
                            print(f"Remaining: {format_time(remaining)}")
                            print()
                            print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
                            print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
                            print(f"Perfect matches: {perfect}/{current} ({perfect/current*100:.2f}%)")
                            print()
                            
                            # Show latest 5 results
                            print("=" * 80)
                            print("📝 LATEST 5 RESULTS")
                            print("=" * 80)
                            for i, result in enumerate(results[-5:], 1):
                                match_icon = "✅" if result["match"] else "❌"
                                print(f"\n{i}. Sample {result['sample_id']} {match_icon} WER: {result['wer']:.4f}")
                                print(f"   Reference: {result['reference'][:80]}...")
                                print(f"   Prediction: {result['prediction'][:80]}...")
                            
                            if current >= total:
                                print("\n" + "=" * 80)
                                print("✅ TEST COMPLETED!")
                                print("=" * 80)
                                break
                            
                            last_count = current
                    
                except (json.JSONDecodeError, KeyError):
                    pass  # File might be incomplete
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\nCurrent progress: {len(data.get('results', []))}/{data.get('total_samples', 0)}")


if __name__ == "__main__":
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else "whisper_full_test_bf16_log.json"
    monitor_test(log_file)

