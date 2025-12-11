#!/bin/bash
# Ví dụ merge models

echo "=" | head -c 80
echo ""
echo "MERGE MODELS EXAMPLES"
echo "=" | head -c 80
echo ""

# 1. Merge Qwen LoRA adapter thành full model
echo "1. Merge Qwen LoRA adapter:"
echo "python3 scripts/merge_lora_adapter.py \\"
echo "  --adapter models/finetuned/qwen-mixed \\"
echo "  --output models/finetuned/qwen-mixed-merged"
echo ""

# 2. Merge Whisper models (weight averaging)
echo "2. Merge Whisper models (weight averaging):"
echo "python3 scripts/merge_whisper_models.py \\"
echo "  --base-model openai/whisper-small \\"
echo "  --finetuned-model models/finetuned/whisper-mixed \\"
echo "  --output models/finetuned/whisper-merged \\"
echo "  --alpha 0.5"
echo ""

# 3. Merge Whisper (use fine-tuned only - default)
echo "3. Merge Whisper (use fine-tuned only):"
echo "python3 scripts/merge_whisper_models.py \\"
echo "  --base-model openai/whisper-small \\"
echo "  --finetuned-model models/finetuned/whisper-mixed \\"
echo "  --output models/finetuned/whisper-merged \\"
echo "  --alpha 1.0"
echo ""

echo "After merging, update .env:"
echo "  GEN_FINETUNED_MODEL=./models/finetuned/qwen-mixed-merged"
echo "  ASR_FINETUNED_MODEL=./models/finetuned/whisper-merged"

