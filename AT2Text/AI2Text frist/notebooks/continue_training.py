#!/usr/bin/env python3
"""
继续训练模型 - 从现有checkpoint恢复或开始新训练
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 60)
print("继续训练模型")
print("=" * 60)

# ========================================
# 1. 检查现有checkpoint
# ========================================
print("\n1. 检查现有checkpoint...")
checkpoint_base = Path('checkpoints')
existing_checkpoints = []

# 查找所有checkpoint目录
for ckpt_dir in checkpoint_base.glob('*'):
    if ckpt_dir.is_dir():
        ckpt_files = list(ckpt_dir.glob('*.pt')) + list(ckpt_dir.glob('*.ckpt'))
        if ckpt_files:
            latest = sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            existing_checkpoints.append({
                'dir': ckpt_dir,
                'file': latest,
                'time': datetime.fromtimestamp(latest.stat().st_mtime),
                'size': latest.stat().st_size / (1024 * 1024)
            })

if existing_checkpoints:
    print(f"  找到 {len(existing_checkpoints)} 个checkpoint目录:")
    for i, ckpt in enumerate(existing_checkpoints, 1):
        print(f"    {i}. {ckpt['dir'].name}")
        print(f"       文件: {ckpt['file'].name}")
        print(f"       大小: {ckpt['size']:.2f} MB")
        print(f"       时间: {ckpt['time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 选择最新的
    latest_ckpt = sorted(existing_checkpoints, key=lambda x: x['time'], reverse=True)[0]
    resume_from = str(latest_ckpt['file'])
    print(f"\n  将从此checkpoint恢复: {resume_from}")
else:
    print("  未找到现有checkpoint，将从头开始训练")
    resume_from = None

# ========================================
# 2. 创建训练配置
# ========================================
print("\n2. 配置训练参数...")

# 读取默认配置
with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 创建新的checkpoint目录（带时间戳）
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
checkpoint_dir = checkpoint_base / f'training_{timestamp}'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# 更新配置
cfg['checkpoint_dir'] = str(checkpoint_dir)
cfg['num_epochs'] = 50  # 完整训练50个epoch
cfg['batch_size'] = 16  # 使用默认batch size
cfg['run_name'] = f'training_{timestamp}'  # 唯一的运行名称

# 保存训练配置
config_path = f'configs/training_{timestamp}.yaml'
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"  配置文件: {config_path}")
print(f"  Checkpoint目录: {checkpoint_dir.absolute()}")
print(f"  Epochs: {cfg['num_epochs']}")
print(f"  Batch size: {cfg['batch_size']}")
if resume_from:
    print(f"  从checkpoint恢复: {resume_from}")

# ========================================
# 3. 检查数据库
# ========================================
print("\n3. 检查数据库...")
db_path = cfg.get('database_path', 'database/asr_training.db')
if not os.path.isabs(db_path):
    db_path = os.path.join(project_root, db_path)

if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  [OK] 数据库存在: {db_path}")
    print(f"  大小: {size_mb:.2f} MB")
else:
    print(f"  [ERROR] 数据库不存在: {db_path}")
    print("  请先运行数据导入步骤")
    sys.exit(1)

# ========================================
# 4. 开始训练
# ========================================
print("\n" + "=" * 60)
print("4. 开始训练...")
print("=" * 60)
print(f"配置: {config_path}")
print(f"Checkpoint将保存到: {checkpoint_dir.absolute()}")
if resume_from:
    print(f"从checkpoint恢复: {resume_from}")
print("-" * 60)

# 运行训练
train_cmd = [sys.executable, 'training/train.py', '--config', config_path]
if resume_from:
    train_cmd.extend(['--resume', resume_from])

result = subprocess.run(
    train_cmd,
    cwd=project_root,
    env=dict(os.environ, PYTHONUNBUFFERED='1')
)

print("-" * 60)

# ========================================
# 5. 检查生成的checkpoint
# ========================================
print("\n5. 检查生成的checkpoint...")
checkpoint_files = list(checkpoint_dir.glob('*.pt')) + list(checkpoint_dir.glob('*.ckpt'))

if checkpoint_files:
    print(f"[OK] 找到 {len(checkpoint_files)} 个checkpoint文件:")
    for ckpt in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"  {ckpt.name}")
        print(f"    大小: {size_mb:.2f} MB")
        print(f"    时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    路径: {ckpt.absolute()}")
    
    latest = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"\n最新checkpoint: {latest.absolute()}")
else:
    print("[WARN] 未找到checkpoint文件")

print("\n" + "=" * 60)
if result.returncode == 0:
    print("[OK] 训练完成！")
    print(f"Checkpoint保存在: {checkpoint_dir.absolute()}")
else:
    print(f"[WARN] 训练可能未完成，退出码: {result.returncode}")
print("=" * 60)

