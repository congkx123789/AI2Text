#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("检查训练状态和Checkpoint")
print("=" * 60)

ckpt_base = Path('checkpoints')
training_dirs = [d for d in ckpt_base.iterdir() if d.is_dir() and 'training_' in d.name]

if training_dirs:
    print(f"\n找到 {len(training_dirs)} 个训练目录:")
    print("-" * 60)
    
    for d in sorted(training_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
        ckpts = list(d.glob('*.pt')) + list(d.glob('*.ckpt'))
        print(f"\n目录: {d.name}")
        print(f"  创建时间: {datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Checkpoint数量: {len(ckpts)}")
        
        if ckpts:
            latest = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            size_mb = latest.stat().st_size / (1024 * 1024)
            print(f"  最新checkpoint: {latest.name}")
            print(f"    大小: {size_mb:.2f} MB")
            print(f"    时间: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    路径: {latest.absolute()}")
else:
    print("\n未找到训练目录")

# 检查test_run目录
test_run = ckpt_base / 'test_run'
if test_run.exists():
    ckpts = list(test_run.glob('*.pt')) + list(test_run.glob('*.ckpt'))
    if ckpts:
        print(f"\n测试checkpoint目录 (test_run):")
        print(f"  Checkpoint数量: {len(ckpts)}")
        latest = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        size_mb = latest.stat().st_size / (1024 * 1024)
        print(f"  最新: {latest.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 60)




