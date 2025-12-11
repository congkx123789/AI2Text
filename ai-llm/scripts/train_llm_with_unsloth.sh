#!/bin/bash
# Script helper để chạy training với Unsloth và cấu hình tối ưu VRAM
# Sử dụng: ./scripts/train_llm_with_unsloth.sh

set -e  # Dừng nếu có lỗi

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cấu hình chống phân mảnh bộ nhớ PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kiểm tra xem unsloth đã được cài đặt chưa
if ! python -c "import unsloth" 2>/dev/null; then
    echo "⚠️  Unsloth chưa được cài đặt!"
    echo "Đang cài đặt Unsloth..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    echo "✅ Unsloth đã được cài đặt!"
fi

# Chạy training với các tham số tối ưu để tận dụng VRAM
# Batch size 16 + accumulation 1 = effective batch size 16 (tận dụng VRAM tối đa)
python3 scripts/train_llm.py \
  --dataset data/processed_finetune/llm_train_mixed.jsonl \
  --output-dir models/finetuned/qwen-mixed \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --num-epochs 1 \
  "$@"  # Cho phép override các tham số khác

echo "✅ Training hoàn tất!"

