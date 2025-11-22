#!/usr/bin/env python3
"""检查GPU和CUDA支持"""
import torch

print("=" * 60)
print("GPU和CUDA检查")
print("=" * 60)

print(f"\nPyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # 测试GPU计算
    print("\n测试GPU计算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  [OK] GPU计算测试成功！")
    except Exception as e:
        print(f"  [ERROR] GPU计算测试失败: {e}")
else:
    print("\n[WARN] CUDA不可用，将使用CPU训练")
    print("可能需要安装支持CUDA的PyTorch版本")

print("=" * 60)




