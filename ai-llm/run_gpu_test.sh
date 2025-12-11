#!/bin/bash
# Script để chạy GPU test với LD_LIBRARY_PATH được set đúng

cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# Tìm và export LD_LIBRARY_PATH cho cuDNN và cuBLAS
CUDNN_PATH=$(dirname $(find .venv -name "libcudnn.so.8*" -o -name "libcudnn.so.9*" 2>/dev/null | head -n 1) 2>/dev/null)
CUBLAS_PATH=$(dirname $(find .venv -name "libcublas.so.11*" -o -name "libcublas.so.12*" 2>/dev/null | head -n 1) 2>/dev/null)

if [ -z "$CUDNN_PATH" ] || [ -z "$CUBLAS_PATH" ]; then
    echo "❌ Không tìm thấy cuDNN hoặc cuBLAS libraries"
    echo "   Cài đặt: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
    exit 1
fi

export LD_LIBRARY_PATH=$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH

echo "✅ LD_LIBRARY_PATH configured:"
echo "   CUDNN: $CUDNN_PATH"
echo "   CUBLAS: $CUBLAS_PATH"
echo ""

# Chạy test với GPU (bf16)
python3 test_full_dataset_bf16.py \
    --model ./models/final/whisper-vi-en-ct2 \
    --test-dataset data/processed_finetune/whisper_test.jsonl \
    --base-audio-dir data/processed/full_merged_dataset \
    --output whisper_full_test_bf16_log.json \
    --device cuda \
    --compute bfloat16

