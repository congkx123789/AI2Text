#!/bin/bash

# Training Script Helper - AI2Text
# Usage: ./run_training.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="configs/default.yaml"
CHECKPOINT_DIR="checkpoints"
LOG_FILE="training_output.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_help() {
    echo -e "${GREEN}AI2Text Training Helper${NC}"
    echo ""
    echo "Usage: ./run_training.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start          - Start training from beginning"
    echo "  resume         - Resume from best_model.pt"
    echo "  resume-newest  - Resume from newest checkpoint"
    echo "  resume-latest  - Alias for resume-newest"
    echo "  status         - Check training status"
    echo "  stop           - Stop running training"
    echo "  logs           - View training logs (tail -f)"
    echo "  checkpoints    - List all checkpoints"
    echo "  test-wer       - Quick WER/CER test (3 samples)"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_training.sh start"
    echo "  ./run_training.sh resume"
    echo "  ./run_training.sh resume-newest"
    echo "  ./run_training.sh status"
    echo "  ./run_training.sh logs"
}

find_newest_checkpoint() {
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "best_model.pt"
        return
    fi
    
    python3 -c "
import os
checkpoints = [f for f in os.listdir('$CHECKPOINT_DIR') if f.endswith('.pt')]
if not checkpoints:
    print('best_model.pt')
else:
    checkpoints.sort(key=lambda x: os.path.getmtime(f'$CHECKPOINT_DIR/{x}'), reverse=True)
    print(checkpoints[0])
"
}

start_training() {
    echo -e "${GREEN}🚀 Starting training from beginning...${NC}"
    
    # Check if training is already running
    if pgrep -f "train.py" > /dev/null; then
        echo -e "${YELLOW}⚠️  Training is already running!${NC}"
        echo "Use './run_training.sh stop' to stop it first."
        exit 1
    fi
    
    python training/train.py --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE" &
    TRAIN_PID=$!
    echo -e "${GREEN}✅ Training started (PID: $TRAIN_PID)${NC}"
    echo "Log file: $LOG_FILE"
    echo "View logs: tail -f $LOG_FILE"
}

resume_training() {
    CHECKPOINT="${1:-best_model.pt}"
    echo -e "${GREEN}🔄 Resuming training from: $CHECKPOINT${NC}"
    
    # Check if training is already running
    if pgrep -f "train.py" > /dev/null; then
        echo -e "${YELLOW}⚠️  Training is already running!${NC}"
        echo "Use './run_training.sh stop' to stop it first."
        exit 1
    fi
    
    if [ ! -f "$CHECKPOINT_DIR/$CHECKPOINT" ]; then
        echo -e "${RED}❌ Checkpoint not found: $CHECKPOINT_DIR/$CHECKPOINT${NC}"
        exit 1
    fi
    
    python training/train.py --config "$CONFIG_FILE" --resume "$CHECKPOINT_DIR/$CHECKPOINT" 2>&1 | tee "$LOG_FILE" &
    TRAIN_PID=$!
    echo -e "${GREEN}✅ Training resumed (PID: $TRAIN_PID)${NC}"
    echo "Log file: $LOG_FILE"
    echo "View logs: tail -f $LOG_FILE"
}

resume_newest() {
    NEWEST=$(find_newest_checkpoint)
    echo -e "${GREEN}🔍 Found newest checkpoint: $NEWEST${NC}"
    resume_training "$NEWEST"
}

check_status() {
    echo -e "${GREEN}📊 Training Status${NC}"
    echo ""
    
    # Check if training is running
    if pgrep -f "train.py" > /dev/null; then
        echo -e "${GREEN}✅ Training is RUNNING${NC}"
        echo ""
        echo "Processes:"
        ps aux | grep -E "train.py" | grep -v grep | head -3
    else
        echo -e "${YELLOW}⏸️  Training is NOT running${NC}"
    fi
    
    echo ""
    echo "Latest checkpoints:"
    if [ -d "$CHECKPOINT_DIR" ]; then
        ls -lht "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -5 || echo "No checkpoints found"
    else
        echo "No checkpoints directory"
    fi
    
    echo ""
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 lines from log:"
        tail -5 "$LOG_FILE"
    fi
}

stop_training() {
    echo -e "${YELLOW}🛑 Stopping training...${NC}"
    
    if ! pgrep -f "train.py" > /dev/null; then
        echo -e "${YELLOW}⚠️  No training process found${NC}"
        return
    fi
    
    pkill -f "train.py"
    sleep 2
    
    if pgrep -f "train.py" > /dev/null; then
        echo -e "${RED}❌ Failed to stop training. Trying force kill...${NC}"
        pkill -9 -f "train.py"
        sleep 1
    fi
    
    if ! pgrep -f "train.py" > /dev/null; then
        echo -e "${GREEN}✅ Training stopped${NC}"
    else
        echo -e "${RED}❌ Could not stop training${NC}"
        exit 1
    fi
}

view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}❌ Log file not found: $LOG_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}📄 Viewing logs (Ctrl+C to exit)...${NC}"
    tail -f "$LOG_FILE"
}

list_checkpoints() {
    echo -e "${GREEN}📦 Checkpoints${NC}"
    echo ""
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo -e "${YELLOW}⚠️  Checkpoints directory not found${NC}"
        return
    fi
    
    if [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠️  No checkpoints found${NC}"
        return
    fi
    
    echo "All checkpoints (newest first):"
    ls -lht "$CHECKPOINT_DIR"/*.pt | awk '{printf "  %-30s %8s %s %s %s\n", $9, $5, $6, $7, $8}'
    
    echo ""
    echo "Checkpoint info:"
    python3 -c "
import torch
import os
checkpoints = [f for f in os.listdir('$CHECKPOINT_DIR') if f.endswith('.pt')]
checkpoints.sort(key=lambda x: os.path.getmtime(f'$CHECKPOINT_DIR/{x}'), reverse=True)
for f in checkpoints[:5]:
    try:
        ckpt = torch.load(f'$CHECKPOINT_DIR/{f}', map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 'N/A')
        val_loss = ckpt.get('best_val_loss', 'N/A')
        if isinstance(val_loss, float):
            val_loss = f'{val_loss:.4f}'
        print(f'  {f}: Epoch {epoch}, Best Val Loss: {val_loss}')
    except Exception as e:
        print(f'  {f}: Error loading ({e})')
"
}

test_wer() {
    NEWEST=$(find_newest_checkpoint)
    echo -e "${GREEN}🧪 Testing WER/CER with: $NEWEST${NC}"
    
    if [ ! -f "$CHECKPOINT_DIR/$NEWEST" ]; then
        echo -e "${RED}❌ Checkpoint not found: $CHECKPOINT_DIR/$NEWEST${NC}"
        exit 1
    fi
    
    python quick_test_wer.py --checkpoint "$CHECKPOINT_DIR/$NEWEST" --num-samples 3
}

# Main command handler
case "${1:-help}" in
    start)
        start_training
        ;;
    resume)
        resume_training "best_model.pt"
        ;;
    resume-newest|resume-latest)
        resume_newest
        ;;
    status)
        check_status
        ;;
    stop)
        stop_training
        ;;
    logs)
        view_logs
        ;;
    checkpoints)
        list_checkpoints
        ;;
    test-wer)
        test_wer
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac

