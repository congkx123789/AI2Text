#!/bin/bash
# Simple script to watch training progress
# Usage: ./scripts/watch_training.sh

cd "$(dirname "$0")/.."

while true; do
    clear
    echo "============================================================"
    echo "TRAINING MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
    
    # Check if training is running
    PID=$(pgrep -f "train_ctc" | head -1)
    if [ -n "$PID" ]; then
        ETIME=$(ps -p "$PID" -o etime= | tr -d ' ')
        echo "✓ Training đang chạy (PID: $PID, Thời gian: $ETIME)"
    else
        echo "✗ Training không chạy"
        exit 1
    fi
    
    # Latest checkpoint
    LATEST_CHECKPOINT=$(ls -t data/results/checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        EPOCH=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint_epoch_\([0-9]*\)\.pt/\1/')
        MTIME=$(stat -c %y "$LATEST_CHECKPOINT" | cut -d' ' -f2 | cut -d'.' -f1)
        echo "📁 Epoch hiện tại: $EPOCH/20"
        echo "   Checkpoint lúc: $MTIME"
    fi
    
    # GPU info
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader,nounits | \
        awk -F', ' '{printf "📊 GPU: %dMB/%dMB (%d%%), Util: %d%%\n", $1, $2, ($1*100/$2), $3}'
    fi
    
    echo ""
    echo "============================================================"
    echo "Auto-refresh mỗi 5 giây. Nhấn Ctrl+C để dừng."
    sleep 5
done

