#!/bin/bash
# Quick status check cho test

cd "$(dirname "$0")/.."

echo "=" | head -c 80
echo ""
echo "📊 TEST STATUS"
echo "=" | head -c 80
echo ""

# Check if test is running
if pgrep -f test_full_dataset_bf16.py > /dev/null; then
    PID=$(pgrep -f test_full_dataset_bf16.py | head -1)
    echo "✅ Test is RUNNING (PID: $PID)"
else
    echo "❌ Test is NOT running"
fi

echo ""

# Check log file
if [ -f "whisper_full_test_bf16_log.json" ]; then
    python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

log = Path('whisper_full_test_bf16_log.json')
if log.exists():
    data = json.load(open(log))
    results = data.get('results', [])
    total = data.get('total_samples', 0)
    progress = data.get('current_progress', {})
    metrics = data.get('current_metrics', {})
    
    print(f"📁 Log file: whisper_full_test_bf16_log.json")
    print(f"📊 Progress: {len(results)}/{total} samples ({len(results)/total*100:.2f}%)")
    
    if metrics:
        print(f"📈 Current Metrics:")
        print(f"   Average WER: {metrics.get('average_wer', 0):.4f} ({metrics.get('average_wer', 0)*100:.2f}%)")
        print(f"   Average CER: {metrics.get('average_cer', 0):.4f} ({metrics.get('average_cer', 0)*100:.2f}%)")
        print(f"   Perfect matches: {metrics.get('perfect_matches', 0)}/{len(results)} ({metrics.get('perfect_match_rate', 0)*100:.2f}%)")
    
    if results:
        latest = results[-1]
        print(f"\n📝 Latest Sample:")
        print(f"   ID: {latest['sample_id']} | WER: {latest['wer']:.4f} | Match: {'✅' if latest['match'] else '❌'}")
        print(f"   Reference:  {latest['reference'][:70]}...")
        print(f"   Prediction: {latest['prediction'][:70]}...")
else:
    print("❌ Log file not found")
EOF
else
    echo "❌ Log file not found"
fi

echo ""
echo "=" | head -c 80
echo ""
echo "📋 Quick Commands:"
echo "   Watch: python3 watch_log.py whisper_full_test_bf16_log.json 10"
echo "   View:  python3 view_test_results.py --log whisper_full_test_bf16_log.json"
echo "   Log:   tail -f whisper_full_test.log"

