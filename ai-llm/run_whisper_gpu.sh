#!/bin/bash
# Wrapper script để chạy Whisper với GPU (bf16)
# Sử dụng: ./run_whisper_gpu.sh audio.wav

# Set environment variables để fix cuDNN issues
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export USE_GPU=1

# Chạy script với GPU
python run_whisper.py "$@"

