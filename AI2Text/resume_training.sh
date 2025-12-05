#!/bin/bash
# Resume training from the good checkpoint (Epoch 1) with stabilized settings

cd "$(dirname "$0")"

echo "🚀 Resuming training from best_model.pt (Epoch 1)"
echo "📋 Configuration changes:"
echo "   - Learning Rate: 0.0002 (was 0.0003)"
echo "   - Warmup: 10% (was 5%)"
echo "   - Timestamp Loss Weight: 0.01 (was 0.1)"
echo ""
echo "⏳ Starting training..."
echo ""

python3 training/train.py \
  --config configs/default.yaml \
  --resume checkpoints/best_model.pt

