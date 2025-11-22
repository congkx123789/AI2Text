#!/usr/bin/env python3
"""
安装支持CUDA的PyTorch
MX330支持CUDA，需要安装CUDA版本的PyTorch
"""
import subprocess
import sys

print("=" * 60)
print("安装支持CUDA的PyTorch")
print("=" * 60)

# MX330通常支持CUDA 11.x或12.x
# 根据nvidia-smi显示CUDA Version: 13.0（这是驱动支持的最高版本）
# 但实际PyTorch可能支持CUDA 11.8或12.1

print("\n正在卸载CPU版本的PyTorch...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchaudio", "-y"])

print("\n正在安装CUDA 12.1版本的PyTorch...")
# 安装CUDA 12.1版本的PyTorch（兼容性较好）
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "torch", "torchaudio", 
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

print("\n安装完成！请运行 check_gpu.py 验证")




