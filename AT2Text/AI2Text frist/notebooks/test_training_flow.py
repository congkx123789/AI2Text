#!/usr/bin/env python3
"""
测试训练流程 - 运行一个最小训练来验证并生成checkpoint
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("=" * 60)
print("测试训练流程")
print("=" * 60)

# 1. 检查数据库
print("\n1. 检查数据库...")
db_path = 'database/asr_training.db'
if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"   [OK] 数据库存在: {size_mb:.2f} MB")
    
    # 检查数据库是否有数据
    try:
        from database.db_utils import ASRDatabase
        db = ASRDatabase(db_path)
        stats = db.get_statistics()
        print(f"   音频文件数: {stats.get('total_audio_files', 0)}")
        print(f"   转录数: {stats.get('total_transcripts', 0)}")
        
        if stats.get('total_audio_files', 0) == 0:
            print("   [WARN] 数据库为空，需要先导入数据")
            print("   运行 notebook 的 Cell 8-9 来导入数据")
            sys.exit(1)
    except Exception as e:
        print(f"   [WARN] 无法读取数据库统计: {e}")
else:
    print(f"   [ERROR] 数据库不存在: {db_path}")
    print("   需要先运行数据导入步骤")
    sys.exit(1)

# 2. 创建测试配置
print("\n2. 创建测试配置...")
with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 修改为最小训练配置
test_cfg = cfg.copy()
test_cfg['num_epochs'] = 1  # 只训练1个epoch
test_cfg['batch_size'] = 2  # 小batch size
test_cfg['checkpoint_dir'] = 'checkpoints/test_run'

# 创建checkpoint目录
checkpoint_dir = Path(test_cfg['checkpoint_dir'])
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# 保存测试配置
test_config_path = 'configs/test_training.yaml'
with open(test_config_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(test_cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"   [OK] 测试配置已创建: {test_config_path}")
print(f"   Epochs: {test_cfg['num_epochs']}")
print(f"   Batch size: {test_cfg['batch_size']}")
print(f"   Checkpoint dir: {checkpoint_dir.absolute()}")

# 3. 检查训练脚本
print("\n3. 检查训练脚本...")
train_script = 'training/train.py'
if not os.path.exists(train_script):
    print(f"   [ERROR] {train_script} 不存在")
    sys.exit(1)
print(f"   [OK] {train_script} 存在")

# 4. 运行训练（测试模式）
print("\n" + "=" * 60)
print("4. 开始测试训练...")
print("=" * 60)
print("注意: 这将运行一个最小训练（1 epoch, batch_size=2）")
print("      用于验证流程是否能正常工作")
print("-" * 60)

import subprocess
result = subprocess.run(
    [sys.executable, train_script, '--config', test_config_path],
    cwd=project_root,
    env=dict(os.environ, PYTHONUNBUFFERED='1'),
    capture_output=False
)

print("-" * 60)

# 5. 检查checkpoint
print("\n5. 检查生成的checkpoint...")
checkpoint_files = list(checkpoint_dir.glob('*.pt')) + list(checkpoint_dir.glob('*.ckpt'))
if checkpoint_files:
    print(f"   [OK] 找到 {len(checkpoint_files)} 个checkpoint文件:")
    for ckpt in checkpoint_files:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"      {ckpt.name}: {size_mb:.2f} MB")
    
    # 列出所有checkpoint
    print(f"\n   Checkpoint 位置: {checkpoint_dir.absolute()}")
    print("\n   可以使用以下命令评估:")
    print(f"   python training/evaluate.py --config {test_config_path} --checkpoint {checkpoint_files[0]}")
else:
    print("   [WARN] 未找到checkpoint文件")
    if result.returncode != 0:
        print(f"   训练可能失败，退出码: {result.returncode}")

print("\n" + "=" * 60)
if result.returncode == 0 and checkpoint_files:
    print("[OK] 测试训练完成！Checkpoint已生成")
    print(f"Checkpoint保存在: {checkpoint_dir.absolute()}")
else:
    print("[WARN] 训练可能未完成或失败")
    print("请检查上面的错误信息")
print("=" * 60)

