#!/bin/bash
# Simple script to monitor training progress

LOG_FILE="logs/training.log"

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

echo "📊 Current Status:"
echo ""

# Get latest epoch info
tail -50 "$LOG_FILE" | grep -E "Epoch|loss|WER|CER|Learning rate" | tail -10

echo ""
echo "📈 Recent Progress:"
tail -20 "$LOG_FILE" | grep -E "batch|loss|step" | tail -5

echo ""
echo "💾 Checkpoints:"
ls -lh checkpoints/*.pt 2>/dev/null | tail -3 || echo "   No checkpoints yet"

echo ""
echo "📝 Full log: tail -f $LOG_FILE"

