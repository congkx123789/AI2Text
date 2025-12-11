#!/bin/bash
# Script để chạy GPU test song song với nhiều processes để tận dụng GPU tối đa

cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# Tìm và export LD_LIBRARY_PATH cho cuDNN và cuBLAS
CUDNN_PATH=$(dirname $(find .venv -name "libcudnn.so.8*" -o -name "libcudnn.so.9*" 2>/dev/null | head -n 1) 2>/dev/null)
CUBLAS_PATH=$(dirname $(find .venv -name "libcublas.so.11*" -o -name "libcublas.so.12*" 2>/dev/null | head -n 1) 2>/dev/null)

if [ -z "$CUDNN_PATH" ] || [ -z "$CUBLAS_PATH" ]; then
    echo "❌ Không tìm thấy cuDNN hoặc cuBLAS libraries"
    exit 1
fi

export LD_LIBRARY_PATH=$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH

# Số processes song song (có thể điều chỉnh)
NUM_PROCESSES=${1:-2}

echo "🚀 Starting parallel GPU test with $NUM_PROCESSES processes"
echo "📊 Configuration:"
echo "   Device: CUDA (GPU)"
echo "   Compute: bfloat16"
echo "   Parallel processes: $NUM_PROCESSES"
echo ""

# Tạo thư mục cho kết quả từng process
mkdir -p test_results_parallel

# Chạy song song nhiều processes
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    echo "Starting process $i..."
    python3 test_full_dataset_bf16.py \
        --model ./models/final/whisper-vi-en-ct2 \
        --test-dataset data/processed_finetune/whisper_test.jsonl \
        --base-audio-dir data/processed/full_merged_dataset \
        --output test_results_parallel/whisper_test_bf16_part${i}.json \
        --device cuda \
        --compute bfloat16 \
        --max-samples $((32818 / NUM_PROCESSES)) \
        > test_results_parallel/test_part${i}.log 2>&1 &
done

echo ""
echo "✅ All $NUM_PROCESSES processes started!"
echo "📝 Monitor logs:"
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    echo "   tail -f test_results_parallel/test_part${i}.log"
done
echo ""
echo "⏱️  Estimated time: ~20-30 minutes (with $NUM_PROCESSES processes)"

