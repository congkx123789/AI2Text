# 本地 Jupyter 运行指南

## 快速设置

在 Jupyter Notebook 中运行以下代码，替换 Cell 7 的内容：

```python
# 6) QUICK TEST config and list parquet shards
import os
import glob
from pathlib import Path

# QUICK TEST MODE
QUICK_TEST = True  # Set to False for full training
MAX_SHARDS_TRAIN = 1 if QUICK_TEST else None
MAX_SHARDS_VAL   = 1 if QUICK_TEST else None
MAX_SHARDS_TEST  = 1 if QUICK_TEST else None
MAX_EXAMPLES_TRAIN = 10 if QUICK_TEST else None
MAX_EXAMPLES_VAL   = 5  if QUICK_TEST else None
MAX_EXAMPLES_TEST  = 5  if QUICK_TEST else None

# ========================================
# AUTO-DETECT ENVIRONMENT AND SET BASE_DIR
# ========================================
# Detect if running locally (Jupyter) or on Colab
IS_COLAB = os.path.exists('/content')
IS_LOCAL = not IS_COLAB

if IS_LOCAL:
    # Local Jupyter environment - use your local Drive path
    BASE_DIR = r"G:\My Drive\datasets\bud500\data"
    print("=" * 60)
    print("Running in LOCAL Jupyter environment")
    print("=" * 60)
else:
    # Colab environment - use Drive mount path
    BASE_DIR = "/content/drive/MyDrive/bud500_data"
    print("=" * 60)
    print("Running in Google Colab")
    print("=" * 60)

print(f"\nLooking for parquet files...")
print(f"   Base directory: {BASE_DIR}")
print("=" * 60)

# Check if directory exists
if not os.path.exists(BASE_DIR):
    print(f"\nFolder not found: {BASE_DIR}")
    if IS_LOCAL:
        print("\nFor LOCAL Jupyter:")
        print("   1. Make sure Google Drive is synced locally")
        print("   2. Update BASE_DIR above with your actual local path")
        print("   3. Example: r\"G:\\My Drive\\datasets\\bud500\\data\"")
    else:
        print("\nFor Colab:")
        print("   1. Open the shared folder in your browser")
        print("   2. Add shortcut to Drive")
    raise FileNotFoundError(f"Please configure BASE_DIR correctly. Folder not found: {BASE_DIR}")
else:
    print(f"Folder found: {BASE_DIR}")

train_files = sorted(glob.glob(str(Path(BASE_DIR) / "train-*.parquet")))
val_files   = sorted(glob.glob(str(Path(BASE_DIR) / "validation-*.parquet")))
test_files  = sorted(glob.glob(str(Path(BASE_DIR) / "test-*.parquet")))

if not val_files and not test_files:
    all_files = sorted(glob.glob(str(Path(BASE_DIR) / "*.parquet")))
    val_files = all_files[:1]
    test_files = all_files[1:2]
    train_files = all_files[2:]

def _limit_shards(files, max_n):
    if max_n is not None and len(files) > max_n:
        return files[:max_n], True
    return files, False

train_files, cut_tr = _limit_shards(train_files, MAX_SHARDS_TRAIN)
val_files, cut_va   = _limit_shards(val_files, MAX_SHARDS_VAL)
test_files, cut_te  = _limit_shards(test_files, MAX_SHARDS_TEST)

print(f"Found {len(train_files)} train shards, {len(val_files)} val shards, {len(test_files)} test shards")
if QUICK_TEST:
    if cut_tr: print(f"QUICK_TEST: Using only first {MAX_SHARDS_TRAIN} train shards")
    if cut_va: print(f"QUICK_TEST: Using only first {MAX_SHARDS_VAL} validation shards")
    if cut_te: print(f"QUICK_TEST: Using only first {MAX_SHARDS_TEST} test shards")
    print("QUICK TEST MODE ENABLED - Training will be fast but limited!")
```

## 其他需要修改的 Cells

### Cell 5 (工作目录设置)
如果运行在本地，修改为：
```python
# 5) Set working directory and environment
import os
import sys

# For local Jupyter, use current directory
project_root = os.getcwd()
if 'notebooks' in project_root:
    # If running from notebooks folder, go up one level
    project_root = Path(project_root).parent

os.chdir(project_root)
print('Changed to project directory:', os.getcwd())

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print('Added project root to Python path')

# Verify key directories exist
required_dirs = ['training', 'models', 'configs', 'database', 'preprocessing']
missing = [d for d in required_dirs if not os.path.isdir(d)]
if missing:
    print('Missing directories:', missing)
else:
    print('All required directories found')
```

### Cell 11 (Checkpoint 配置)
对于本地运行，修改 checkpoint 目录：
```python
# 本地保存 checkpoint
checkpoint_dir = f'checkpoints/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
# 而不是 Drive 路径
```

## 运行步骤

1. 打开 Jupyter Notebook
2. 打开 `notebooks/colab_parquet_training.ipynb`
3. 修改 Cell 7 使用上面的代码
4. 修改 Cell 5 和 Cell 11（如需要）
5. 按顺序运行所有 cells

