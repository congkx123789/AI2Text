#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

ckpt_dir = Path('checkpoints/test_run')
ckpts = list(ckpt_dir.glob('*.pt')) + list(ckpt_dir.glob('*.ckpt'))

print(f"Found {len(ckpts)} checkpoint files")
print("=" * 60)

for c in sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True):
    size_mb = c.stat().st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(c.stat().st_mtime)
    print(f"File: {c.name}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Time: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Path: {c.absolute()}")
    print("-" * 60)

if ckpts:
    latest = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"\nLatest checkpoint: {latest.absolute()}")

