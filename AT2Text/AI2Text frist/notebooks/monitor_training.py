#!/usr/bin/env python3
"""
监控训练进度 - 检查checkpoint生成情况
"""
import time
from pathlib import Path
from datetime import datetime

print("监控训练进度...")
print("按 Ctrl+C 停止监控")
print("=" * 60)

ckpt_base = Path('checkpoints')
last_count = {}

try:
    while True:
        # 查找最新的训练目录
        training_dirs = [d for d in ckpt_base.iterdir() 
                        if d.is_dir() and 'training_' in d.name]
        
        if training_dirs:
            latest_dir = sorted(training_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            ckpts = list(latest_dir.glob('*.pt')) + list(latest_dir.glob('*.ckpt'))
            count = len(ckpts)
            
            if count != last_count.get(latest_dir.name, 0):
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {latest_dir.name}")
                print(f"  Checkpoint数量: {count}")
                
                if ckpts:
                    latest = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    size_mb = latest.stat().st_size / (1024 * 1024)
                    print(f"  最新: {latest.name} ({size_mb:.2f} MB)")
                    print(f"  路径: {latest.absolute()}")
                
                last_count[latest_dir.name] = count
        
        time.sleep(10)  # 每10秒检查一次
        
except KeyboardInterrupt:
    print("\n\n监控已停止")
    print("=" * 60)
    
    # 显示最终状态
    if training_dirs:
        latest_dir = sorted(training_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        ckpts = list(latest_dir.glob('*.pt')) + list(latest_dir.glob('*.ckpt'))
        print(f"\n最终状态 - {latest_dir.name}:")
        print(f"  总checkpoint数: {len(ckpts)}")
        if ckpts:
            latest = sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            print(f"  最新checkpoint: {latest.absolute()}")




