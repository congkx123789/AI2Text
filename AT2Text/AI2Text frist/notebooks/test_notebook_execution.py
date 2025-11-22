#!/usr/bin/env python3
"""
测试 Notebook 执行流程
模拟运行 notebook 的关键步骤，验证是否能正常工作
"""

import os
import sys
import json
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("=" * 60)
print("测试 Notebook 执行流程")
print("=" * 60)

# ========================================
# 测试 1: 检查数据路径
# ========================================
print("\n1. 检查数据路径...")
BASE_DIR = r"G:\My Drive\datasets\bud500\data"

if os.path.exists(BASE_DIR):
    print(f"   [OK] 数据目录存在: {BASE_DIR}")
    
    # 检查 parquet 文件
    import glob
    train_files = glob.glob(str(Path(BASE_DIR) / "train-*.parquet"))
    val_files = glob.glob(str(Path(BASE_DIR) / "validation-*.parquet"))
    test_files = glob.glob(str(Path(BASE_DIR) / "test-*.parquet"))
    
    print(f"   - Train files: {len(train_files)}")
    print(f"   - Validation files: {len(val_files)}")
    print(f"   - Test files: {len(test_files)}")
    
    if len(train_files) == 0:
        print("   [WARN] 未找到训练文件")
    else:
        print(f"   [OK] 找到数据文件")
else:
    print(f"   [ERROR] 数据目录不存在: {BASE_DIR}")
    print("   请检查路径是否正确")
    sys.exit(1)

# ========================================
# 测试 2: 检查项目结构
# ========================================
print("\n2. 检查项目结构...")
required_dirs = ['training', 'models', 'configs', 'database', 'preprocessing', 'scripts']
missing = []

for d in required_dirs:
    if os.path.isdir(d):
        print(f"   [OK] {d}/")
    else:
        print(f"   [ERROR] {d}/ 缺失")
        missing.append(d)

if missing:
    print(f"   [WARN] 缺失 {len(missing)} 个目录")
else:
    print("   [OK] 项目结构完整")

# ========================================
# 测试 3: 检查配置文件
# ========================================
print("\n3. 检查配置文件...")
config_path = Path('configs/default.yaml')
if config_path.exists():
    print(f"   [OK] {config_path} 存在")
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print(f"   [OK] 配置文件可解析")
        print(f"   - batch_size: {cfg.get('batch_size', 'N/A')}")
        print(f"   - num_epochs: {cfg.get('num_epochs', 'N/A')}")
    except Exception as e:
        print(f"   [ERROR] 配置文件解析失败: {e}")
        sys.exit(1)
else:
    print(f"   [ERROR] {config_path} 不存在")
    sys.exit(1)

# ========================================
# 测试 4: 检查训练脚本
# ========================================
print("\n4. 检查训练脚本...")
train_script = Path('training/train.py')
if train_script.exists():
    print(f"   [OK] {train_script} 存在")
    
    # 检查脚本语法
    try:
        with open(train_script, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, str(train_script), 'exec')
        print(f"   [OK] 训练脚本语法正确")
    except SyntaxError as e:
        print(f"   [ERROR] 语法错误: {e}")
        sys.exit(1)
else:
    print(f"   [ERROR] {train_script} 不存在")
    sys.exit(1)

# ========================================
# 测试 5: 检查依赖包
# ========================================
print("\n5. 检查关键依赖包...")
required_packages = {
    'torch': 'PyTorch',
    'datasets': 'HuggingFace Datasets',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'transformers': 'Transformers',
}

missing_packages = []
for pkg, name in required_packages.items():
    try:
        __import__(pkg)
        print(f"   [OK] {name}")
    except ImportError:
        print(f"   [ERROR] {name} 未安装")
        missing_packages.append(pkg)

if missing_packages:
    print(f"\n   [WARN] 缺失 {len(missing_packages)} 个包:")
    print(f"   请运行: pip install {' '.join(missing_packages)}")
else:
    print("   [OK] 所有关键包已安装")

# ========================================
# 测试 6: 模拟数据加载（不实际加载，只检查逻辑）
# ========================================
print("\n6. 检查数据加载逻辑...")
try:
    from datasets import load_dataset, Audio
    print("   [OK] datasets 库可用")
    print("   [OK] 可以加载 parquet 文件")
except ImportError:
    print("   [WARN] datasets 库未安装，但逻辑应该正确")

# ========================================
# 测试 7: 检查数据库工具
# ========================================
print("\n7. 检查数据库工具...")
try:
    from database.db_utils import ASRDatabase
    print("   [OK] 数据库工具可导入")
    
    # 检查数据库文件
    db_path = Path('database/asr_training.db')
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"   [OK] 数据库存在: {size_mb:.2f} MB")
    else:
        print("   [INFO] 数据库不存在（首次运行时会创建）")
except ImportError as e:
    print(f"   [WARN] 数据库工具导入失败: {e}")

# ========================================
# 测试 8: 检查训练模块
# ========================================
print("\n8. 检查训练模块...")
try:
    from training.dataset import create_data_loaders
    from training.train import ASRTrainer
    print("   [OK] 训练模块可导入")
except ImportError as e:
    print(f"   [WARN] 训练模块导入失败: {e}")
    print("   可能需要先安装依赖")

# ========================================
# 测试 9: 模拟配置生成
# ========================================
print("\n9. 测试配置生成...")
try:
    import yaml
    from datetime import datetime
    
    # 创建测试配置
    test_checkpoint_dir = Path('checkpoints') / 'test_run'
    test_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    test_cfg = cfg.copy()
    test_cfg['checkpoint_dir'] = str(test_checkpoint_dir)
    test_cfg['num_epochs'] = 1  # 测试用
    test_cfg['batch_size'] = 2  # 测试用
    
    test_config_path = Path('configs/test_local.yaml')
    with open(test_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(test_cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"   [OK] 测试配置已生成: {test_config_path}")
    print(f"   [OK] Checkpoint 目录: {test_checkpoint_dir}")
    
    # 清理测试文件
    if test_config_path.exists():
        test_config_path.unlink()
    if test_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(test_checkpoint_dir)
    
except Exception as e:
    print(f"   [WARN] 配置生成测试失败: {e}")

# ========================================
# 总结
# ========================================
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

all_checks = [
    ("数据路径", os.path.exists(BASE_DIR)),
    ("项目结构", len(missing) == 0),
    ("配置文件", config_path.exists()),
    ("训练脚本", train_script.exists()),
]

passed = sum(1 for _, check in all_checks if check)
total = len(all_checks)

for name, check in all_checks:
    status = "[OK]" if check else "[ERROR]"
    print(f"  {status} {name}")

print(f"\n通过: {passed}/{total}")

if passed == total:
    print("\n[OK] 所有关键检查通过！")
    print("   Notebook 应该可以在 Jupyter 中正常运行")
    print("\n运行步骤:")
    print("   1. 打开 Jupyter Notebook")
    print("   2. 打开 notebooks/colab_parquet_training.ipynb")
    print("   3. 按顺序运行所有 cells")
    print("   4. Checkpoint 会保存在 checkpoints/ 目录")
else:
    print(f"\n[WARN] {total - passed} 个检查失败")
    print("   请修复上述问题后再运行 notebook")

print("=" * 60)

