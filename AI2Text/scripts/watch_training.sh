#!/bin/bash
# Quick script to watch training progress in real-time

LOG_FILE="logs/training.log"

echo "=========================================="
echo "🚀 ASR Training Monitor"
echo "=========================================="
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Make sure training is running first!"
    exit 1
fi

echo "📊 Watching: $LOG_FILE"
echo "   Press Ctrl+C to exit"
echo ""
echo "=========================================="
echo ""

# Watch log file in real-time
tail -f "$LOG_FILE" | grep --line-buffered -E "Epoch|Batch|loss|WER|CER|Learning|Summary|Best|checkpoint" | while read line; do
    echo "$(date '+%H:%M:%S') | $line"
done

