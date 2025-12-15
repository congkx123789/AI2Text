#!/bin/bash
# Script to tail training log and show metrics

LOG_FILE="${1:-logs/training_restart.log}"

echo "======================================================================"
echo "📊 TRAINING METRICS - REAL-TIME"
echo "======================================================================"
echo ""

# Get latest progress line
LATEST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)

if [[ -z "$LATEST_LINE" ]]; then
    echo "❌ Log file not found or empty: $LOG_FILE"
    exit 1
fi

# Extract epoch and batch info
if echo "$LATEST_LINE" | grep -q "🚀 Epoch"; then
    EPOCH=$(echo "$LATEST_LINE" | grep -oP "Epoch \K\d+" | head -1)
    TOTAL_EPOCHS=$(echo "$LATEST_LINE" | grep -oP "Epoch \d+/\K\d+" | head -1)
    BATCH=$(echo "$LATEST_LINE" | grep -oP "\|\s+\K\d+(?=/)" | head -1)
    TOTAL_BATCHES=$(echo "$LATEST_LINE" | grep -oP "/\K\d+(?=\s+\[)" | head -1)
    PERCENT=$(echo "$LATEST_LINE" | grep -oP "\s+\K\d+(?=%\|)" | head -1)
    TIME_ELAPSED=$(echo "$LATEST_LINE" | grep -oP "\[\K\d+:\d+(?=<)" | head -1)
    TIME_REMAINING=$(echo "$LATEST_LINE" | grep -oP "<\K\d+:\d+(?=,)" | head -1)
    BATCH_SPEED=$(echo "$LATEST_LINE" | grep -oP ",\s+\K[\d.]+batch/s" | head -1)
    
    echo "🔄 Progress:"
    echo "   Epoch: $EPOCH/$TOTAL_EPOCHS"
    echo "   Batch: $BATCH/$TOTAL_BATCHES ($PERCENT%)"
    echo "   Time: $TIME_ELAPSED elapsed | $TIME_REMAINING remaining"
    echo "   Speed: $BATCH_SPEED"
    echo ""
fi

# Look for metrics in recent log lines (last 100 lines)
echo "📉 METRICS (from recent log entries):"
echo ""

# Check for logged metrics
METRICS_LINE=$(tail -100 "$LOG_FILE" | grep -E "(Loss:|LR:|WER:|CER:)" | tail -1)

if [[ -n "$METRICS_LINE" ]]; then
    echo "$METRICS_LINE" | sed 's/^/   /'
else
    echo "   ⚠️  Metrics not yet logged (will appear after batch 50)"
    echo "   Check progress bar above for current values"
fi

echo ""
echo "======================================================================"
echo "💡 Tip: Metrics are logged every 50 batches"
echo "💡 To watch real-time: tail -f $LOG_FILE | grep -E '(Loss|LR|WER|CER)'"
echo "======================================================================"

