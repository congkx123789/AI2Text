#!/bin/bash
# Example script to resume training from a checkpoint
# Usage: ./scripts/resume_training.sh [checkpoint_path] [epochs]

CHECKPOINT=${1:-"data/results/checkpoints/checkpoint_epoch_1.pt"}
EPOCHS=${2:-2}

python3 -m src.asr.train_ctc \
    --manifest data/processed/merged_dataset/train/manifest.csv \
    --audio_root data/processed/merged_dataset/train \
    --timestamps data/processed/merged_dataset/train/timestamps.json \
    --trim_segments \
    --vocab data/processed/vocab.json \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr 0.001 \
    --device auto \
    --num_workers 0 \
    --amp \
    --log_interval 20 \
    --resume "$CHECKPOINT" \
    --checkpoint_dir data/results/checkpoints

