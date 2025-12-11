#!/usr/bin/env bash
# Quick start script cho training
set -euo pipefail

echo "🚀 AI-LLM Training Quick Start"
echo "================================"

# Check if venv is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Virtual environment chưa được activate"
    echo "   Chạy: source .venv/bin/activate"
    exit 1
fi

# Check dataset
if [[ ! -f "data/processed_finetune/whisper_train_mixed.jsonl" ]]; then
    echo "❌ Không tìm thấy dataset!"
    echo "   Chạy: python3 scripts/convert_csv_to_jsonl.py"
    exit 1
fi

echo ""
echo "📋 Chọn loại training:"
echo "1. Whisper ASR"
echo "2. LLM (QLoRA)"
echo "3. Cả hai"
echo ""
read -p "Nhập lựa chọn (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🎤 Bắt đầu training Whisper..."
        python3 scripts/train_whisper.py \
            --dataset data/processed_finetune/whisper_train_mixed.jsonl \
            --model-id openai/whisper-small \
            --output-dir models/finetuned/whisper-mixed \
            --batch-size 16 \
            --num-epochs 3 \
            --learning-rate 1e-5 \
            --fp16
        ;;
    2)
        echo ""
        echo "💬 Bắt đầu training LLM..."
        python3 scripts/train_llm.py \
            --dataset data/processed_finetune/llm_train_mixed.jsonl \
            --model-id Qwen/Qwen2.5-0.5B-Instruct \
            --output-dir models/finetuned/qwen-mixed \
            --batch-size 4 \
            --num-epochs 3 \
            --learning-rate 2e-4 \
            --lora-r 16 \
            --lora-alpha 32
        ;;
    3)
        echo ""
        echo "🎤 Bắt đầu training Whisper..."
        python3 scripts/train_whisper.py \
            --dataset data/processed_finetune/whisper_train_mixed.jsonl \
            --model-id openai/whisper-small \
            --output-dir models/finetuned/whisper-mixed \
            --batch-size 16 \
            --num-epochs 3 \
            --learning-rate 1e-5 \
            --fp16
        
        echo ""
        echo "💬 Bắt đầu training LLM..."
        python3 scripts/train_llm.py \
            --dataset data/processed_finetune/llm_train_mixed.jsonl \
            --model-id Qwen/Qwen2.5-0.5B-Instruct \
            --output-dir models/finetuned/qwen-mixed \
            --batch-size 4 \
            --num-epochs 3 \
            --learning-rate 2e-4 \
            --lora-r 16 \
            --lora-alpha 32
        ;;
    *)
        echo "❌ Lựa chọn không hợp lệ"
        exit 1
        ;;
esac

echo ""
echo "✅ Hoàn thành!"
echo ""
echo "📝 Để sử dụng model mới, cập nhật .env:"
echo "   ASR_FINETUNED_MODEL=./models/finetuned/whisper-mixed"
echo "   GEN_FINETUNED_MODEL=./models/finetuned/qwen-mixed"


