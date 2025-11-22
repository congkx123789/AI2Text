#!/usr/bin/env python3
"""
列出所有checkpoint文件
"""
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("所有Checkpoint文件")
print("=" * 60)

ckpt_base = Path('checkpoints')
all_ckpts = []

# 遍历所有目录
for ckpt_dir in ckpt_base.iterdir():
    if ckpt_dir.is_dir():
        ckpts = list(ckpt_dir.glob('*.pt')) + list(ckpt_dir.glob('*.ckpt'))
        for ckpt in ckpts:
            all_ckpts.append({
                'file': ckpt,
                'dir': ckpt_dir.name,
                'time': datetime.fromtimestamp(ckpt.stat().st_mtime),
                'size': ckpt.stat().st_size / (1024 * 1024)
            })

# 按时间排序
all_ckpts.sort(key=lambda x: x['time'], reverse=True)

print(f"\n总共找到 {len(all_ckpts)} 个checkpoint文件\n")

for i, ckpt in enumerate(all_ckpts, 1):
    print(f"{i}. {ckpt['dir']}/{ckpt['file'].name}")
    print(f"   大小: {ckpt['size']:.2f} MB")
    print(f"   时间: {ckpt['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   路径: {ckpt['file'].absolute()}")
    print()

# 显示最新的
if all_ckpts:
    latest = all_ckpts[0]
    print("=" * 60)
    print("最新Checkpoint:")
    print(f"  文件: {latest['file'].name}")
    print(f"  目录: {latest['dir']}")
    print(f"  大小: {latest['size']:.2f} MB")
    print(f"  时间: {latest['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  完整路径: {latest['file'].absolute()}")
    print("=" * 60)




