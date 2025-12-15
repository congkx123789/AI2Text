#!/bin/bash
# Script to watch training metrics in real-time

LOG_FILE="${1:-logs/training_restart.log}"

echo "======================================================================"
echo "📊 WATCHING TRAINING METRICS - REAL-TIME"
echo "======================================================================"
echo ""
echo "💡 Metrics will appear after batch 50, 100, 150, 200, etc."
echo "💡 Currently watching: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================================"
echo ""

# Watch for metrics
tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Check for logged metrics
    if echo "$line" | grep -qE "(Batch.*Loss:|LR:|WER:|CER:)"; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "$line"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # Check for epoch summary
    elif echo "$line" | grep -qE "(EPOCH SUMMARY|Train Loss|Val Loss|Learning Rate|WER|CER|Best)"; then
        echo "$line"
    # Show progress every 10 batches
    elif echo "$line" | grep -qE "🚀 Epoch.*\|.*\[.*\]" && echo "$line" | grep -oP "\|\s+\K\d+(?=/)" | grep -qE "(00|50)$"; then
        # Extract batch number
        BATCH=$(echo "$line" | grep -oP "\|\s+\K\d+(?=/)" | head -1)
        if [ ! -z "$BATCH" ] && [ $((BATCH % 50)) -eq 0 ]; then
            echo "📍 Batch $BATCH - Metrics should appear soon..."
        fi
    fi
done

