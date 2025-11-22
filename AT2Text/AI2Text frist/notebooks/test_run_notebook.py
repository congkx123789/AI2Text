#!/usr/bin/env python3
"""
测试运行 notebook 流程 - 模拟 Jupyter 执行
检查每个步骤是否能正常运行
"""

import os
import sys
import json
from pathlib import Path

print("=" * 60)
print("测试 Notebook 执行流程")
print("=" * 60)

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print(f"\n项目目录: {project_root}")
print(f"当前目录: {os.getcwd()}")

# ========================================
# 测试 1: 检查路径
# ========================================
print("\n" + "=" * 60)
print("测试 1: 检查数据路径")
print("=" * 60)

BASE_DIR = r"G:\My Drive\datasets\bud500\data"
print(f"BASE_DIR: {BASE_DIR}")

if os.path.exists(BASE_DIR):
    print("[OK] 路径存在")
    import glob
    train_files = glob.glob(str(Path(BASE_DIR) / "train-*.parquet"))
    val_files = glob.glob(str(Path(BASE_DIR) / "validation-*.parquet"))
    test_files = glob.glob(str(Path(BASE_DIR) / "test-*.parquet"))
    
    print(f"  找到训练文件: {len(train_files)}")
    print(f"  找到验证文件: {len(val_files)}")
    print(f"  找到测试文件: {len(test_files)}")
    
    if len(train_files) > 0:
        print(f"  示例文件: {train_files[0]}")
else:
    print("[ERROR] 路径不存在")
    print("  请检查路径是否正确")

# ========================================
# 测试 2: 检查项目结构
# ========================================
print("\n" + "=" * 60)
print("测试 2: 检查项目结构")
print("=" * 60)

required_files = [
    'training/train.py',
    'models/asr_base.py',
    'configs/default.yaml',
    'database/db_utils.py',
    'preprocessing/audio_processing.py'
]

all_ok = True
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  [OK] {file_path}")
    else:
        print(f"  [ERROR] {file_path} 不存在")
        all_ok = False

if all_ok:
    print("[OK] 项目结构完整")
else:
    print("[ERROR] 项目结构不完整")

# ========================================
# 测试 3: 检查依赖
# ========================================
print("\n" + "=" * 60)
print("测试 3: 检查 Python 依赖")
print("=" * 60)

required_packages = [
    'torch', 'torchaudio', 'transformers', 'librosa',
    'soundfile', 'numpy', 'pandas', 'yaml', 'datasets'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [ERROR] {pkg} 未安装")
        missing.append(pkg)

if missing:
    print(f"\n[WARN] 缺少 {len(missing)} 个包，需要安装:")
    print(f"   pip install {' '.join(missing)}")
else:
    print("\n[OK] 所有依赖已安装")

# ========================================
# 测试 4: 检查配置文件
# ========================================
print("\n" + "=" * 60)
print("测试 4: 检查配置文件")
print("=" * 60)

try:
    import yaml
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print("[OK] 配置文件可读取")
    print(f"  batch_size: {cfg.get('batch_size', 'N/A')}")
    print(f"  num_epochs: {cfg.get('num_epochs', 'N/A')}")
    print(f"  database_path: {cfg.get('database_path', 'N/A')}")
except Exception as e:
    print(f"[ERROR] 配置文件错误: {e}")

# ========================================
# 测试 5: 检查数据库
# ========================================
print("\n" + "=" * 60)
print("测试 5: 检查数据库")
print("=" * 60)

db_path = cfg.get('database_path', 'database/asr_training.db')
if not os.path.isabs(db_path):
    db_path = os.path.join(project_root, db_path)

if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"[OK] 数据库存在: {db_path}")
    print(f"  大小: {size_mb:.2f} MB")
else:
    print(f"[WARN] 数据库不存在: {db_path}")
    print("  需要先运行数据导入步骤")

# ========================================
# 测试 6: 检查训练脚本
# ========================================
print("\n" + "=" * 60)
print("测试 6: 检查训练脚本")
print("=" * 60)

train_script = 'training/train.py'
if os.path.exists(train_script):
    print(f"[OK] {train_script} 存在")
    # 检查语法
    try:
        with open(train_script, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, train_script, 'exec')
        print("[OK] 训练脚本语法正确")
    except SyntaxError as e:
        print(f"[ERROR] 语法错误: {e}")
else:
    print(f"[ERROR] {train_script} 不存在")

# ========================================
# 测试 7: 检查 checkpoint 目录
# ========================================
print("\n" + "=" * 60)
print("测试 7: 检查 Checkpoint 目录")
print("=" * 60)

checkpoint_base = Path('checkpoints')
checkpoint_base.mkdir(exist_ok=True)
print(f"[OK] Checkpoint 目录: {checkpoint_base.absolute()}")

# 查找现有 checkpoints
existing_checkpoints = list(checkpoint_base.glob('*'))
if existing_checkpoints:
    print(f"  找到 {len(existing_checkpoints)} 个 checkpoint 目录:")
    for ckpt in existing_checkpoints[:5]:
        if ckpt.is_dir():
            ckpt_files = list(ckpt.glob('*.pt')) + list(ckpt.glob('*.ckpt'))
            print(f"    {ckpt.name}: {len(ckpt_files)} 个文件")
else:
    print("  暂无 checkpoint（训练后会生成）")

# ========================================
# 总结
# ========================================
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

if all_ok and not missing:
    print("[OK] 所有检查通过！")
    print("\n可以运行以下命令开始训练:")
    print("  python training/train.py --config configs/default.yaml")
    print("\n或者在 Jupyter 中运行 notebook cells")
else:
    print("[WARN] 发现一些问题，请先修复")
    if missing:
        print(f"\n需要安装: pip install {' '.join(missing)}")

print("=" * 60)

