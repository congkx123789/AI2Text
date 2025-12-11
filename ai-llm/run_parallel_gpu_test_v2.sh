#!/bin/bash
# Script để chạy GPU test song song - Version 2: Chia dataset và chạy song song

cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# Tìm và export LD_LIBRARY_PATH
CUDNN_PATH=$(dirname $(find .venv -name "libcudnn.so.8*" -o -name "libcudnn.so.9*" 2>/dev/null | head -n 1) 2>/dev/null)
CUBLAS_PATH=$(dirname $(find .venv -name "libcublas.so.11*" -o -name "libcublas.so.12*" 2>/dev/null | head -n 1) 2>/dev/null)
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH

# Số processes song song
NUM_PROCESSES=${1:-2}
TOTAL_SAMPLES=32818
SAMPLES_PER_PROCESS=$((TOTAL_SAMPLES / NUM_PROCESSES))

echo "🚀 Starting parallel GPU test with $NUM_PROCESSES processes"
echo "📊 Each process will handle ~$SAMPLES_PER_PROCESS samples"
echo ""

# Tạo thư mục kết quả
mkdir -p test_results_parallel

# Chạy song song
PIDS=()
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    START_IDX=$((i * SAMPLES_PER_PROCESS))
    END_IDX=$(((i + 1) * SAMPLES_PER_PROCESS))
    if [ $i -eq $((NUM_PROCESSES-1)) ]; then
        # Process cuối lấy phần còn lại
        END_IDX=$TOTAL_SAMPLES
    fi
    
    echo "Starting process $i: samples $START_IDX to $END_IDX"
    
    python3 test_full_dataset_bf16.py \
        --model ./models/final/whisper-vi-en-ct2 \
        --test-dataset data/processed_finetune/whisper_test.jsonl \
        --base-audio-dir data/processed/full_merged_dataset \
        --output test_results_parallel/whisper_test_bf16_part${i}.json \
        --device cuda \
        --compute bfloat16 \
        --max-samples $SAMPLES_PER_PROCESS \
        > test_results_parallel/test_part${i}.log 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "✅ All $NUM_PROCESSES processes started!"
echo "PIDs: ${PIDS[@]}"
echo ""
echo "📝 Monitor:"
echo "   # Watch all logs:"
echo "   tail -f test_results_parallel/test_part*.log"
echo ""
echo "   # Check progress:"
echo "   python3 -c \"import json, glob; files=glob.glob('test_results_parallel/whisper_test_bf16_part*.json'); total=sum(len(json.load(open(f)).get('results',[])) for f in files); print(f'Progress: {total}/32818 ({total/32818*100:.2f}%)')\""
echo ""
echo "⏱️  Estimated time: ~20-30 minutes (with $NUM_PROCESSES processes)"

