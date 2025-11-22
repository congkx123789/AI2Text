#!/usr/bin/env python3
"""更新 notebook 中的 BASE_DIR 路径，支持本地环境"""

import json
from pathlib import Path

notebook_path = Path(__file__).parent / 'colab_parquet_training.ipynb'

# 读取 notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到包含 BASE_DIR 的 cell (应该是 cell 7，索引 7)
target_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'BASE_DIR' in source and 'CONFIGURE' in source:
            target_cell_idx = i
            break

if target_cell_idx is not None:
    cell = nb['cells'][target_cell_idx]
    source_lines = cell['source']
    
    # 创建新的 source，添加自动检测逻辑
    new_source = [
        "# 6) QUICK TEST config and list parquet shards\n",
        "import os\n",
        "import glob\n",
        "from pathlib import Path\n",
        "\n",
        "# QUICK TEST MODE\n",
        "QUICK_TEST = True  # Set to False for full training\n",
        "MAX_SHARDS_TRAIN = 1 if QUICK_TEST else None\n",
        "MAX_SHARDS_VAL   = 1 if QUICK_TEST else None\n",
        "MAX_SHARDS_TEST  = 1 if QUICK_TEST else None\n",
        "MAX_EXAMPLES_TRAIN = 10 if QUICK_TEST else None\n",
        "MAX_EXAMPLES_VAL   = 5  if QUICK_TEST else None\n",
        "MAX_EXAMPLES_TEST  = 5  if QUICK_TEST else None\n",
        "\n",
        "# ========================================\n",
        "# AUTO-DETECT ENVIRONMENT AND SET BASE_DIR\n",
        "# ========================================\n",
        "# Detect if running locally (Jupyter) or on Colab\n",
        "IS_COLAB = os.path.exists('/content')\n",
        "IS_LOCAL = not IS_COLAB\n",
        "\n",
        "if IS_LOCAL:\n",
        "    # Local Jupyter environment - use your local Drive path\n",
        "    BASE_DIR = r\"G:\\My Drive\\datasets\\bud500\\data\"\n",
        "    print(\"=\" * 60)\n",
        "    print(\"🖥️  Running in LOCAL Jupyter environment\")\n",
        "    print(\"=\" * 60)\n",
        "else:\n",
        "    # Colab environment - use Drive mount path\n",
        "    BASE_DIR = \"/content/drive/MyDrive/bud500_data\"\n",
        "    print(\"=\" * 60)\n",
        "    print(\"☁️  Running in Google Colab\")\n",
        "    print(\"=\" * 60)\n",
        "\n",
        "print(f\"\\n📁 Looking for parquet files...\")\n",
        "print(f\"   Base directory: {BASE_DIR}\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Check if directory exists\n",
        "if not os.path.exists(BASE_DIR):\n",
        "    print(f\"\\n❌ Folder not found: {BASE_DIR}\")\n",
        "    if IS_LOCAL:\n",
        "        print(\"\\n📋 For LOCAL Jupyter:\")\n",
        "        print(\"   1. Make sure Google Drive is synced locally\")\n",
        "        print(\"   2. Update BASE_DIR above with your actual local path\")\n",
        "        print(\"   3. Example: r\\\"G:\\\\My Drive\\\\datasets\\\\bud500\\\\data\\\"\")\n",
        "    else:\n",
        "        print(\"\\n📋 For Colab:\")\n",
        "        print(\"   1. Open the shared folder in your browser:\")\n",
        "        print(\"      https://drive.google.com/drive/folders/17iqsD7J2xi9_YLQGsqopR8XX-nVFN2y3\")\n",
        "        print(\"   2. Right-click on the folder → 'Add shortcut to Drive'\")\n",
        "        print(\"   3. Choose 'My Drive' as the location\")\n",
        "        print(\"   4. Update BASE_DIR above to match the folder name in your Drive\")\n",
        "    raise FileNotFoundError(f\"Please configure BASE_DIR correctly. Folder not found: {BASE_DIR}\")\n",
        "else:\n",
        "    print(f\"✓ Folder found: {BASE_DIR}\")\n",
        "\n",
        "train_files = sorted(glob.glob(str(Path(BASE_DIR) / \"train-*.parquet\")))\n",
        "val_files   = sorted(glob.glob(str(Path(BASE_DIR) / \"validation-*.parquet\")))\n",
        "test_files  = sorted(glob.glob(str(Path(BASE_DIR) / \"test-*.parquet\")))\n",
        "\n",
        "if not val_files and not test_files:\n",
        "    all_files = sorted(glob.glob(str(Path(BASE_DIR) / \"*.parquet\")))\n",
        "    val_files = all_files[:1]\n",
        "    test_files = all_files[1:2]\n",
        "    train_files = all_files[2:]\n",
        "\n",
        "def _limit_shards(files, max_n):\n",
        "    if max_n is not None and len(files) > max_n:\n",
        "        return files[:max_n], True\n",
        "    return files, False\n",
        "\n",
        "train_files, cut_tr = _limit_shards(train_files, MAX_SHARDS_TRAIN)\n",
        "val_files, cut_va   = _limit_shards(val_files, MAX_SHARDS_VAL)\n",
        "test_files, cut_te  = _limit_shards(test_files, MAX_SHARDS_TEST)\n",
        "\n",
        "print(f\"Found {len(train_files)} train shards, {len(val_files)} val shards, {len(test_files)} test shards\")\n",
        "if QUICK_TEST:\n",
        "    if cut_tr: print(f\"⚠️ QUICK_TEST: Using only first {MAX_SHARDS_TRAIN} train shards\")\n",
        "    if cut_va: print(f\"⚠️ QUICK_TEST: Using only first {MAX_SHARDS_VAL} validation shards\")\n",
        "    if cut_te: print(f\"⚠️ QUICK_TEST: Using only first {MAX_SHARDS_TEST} test shards\")\n",
        "    print(\"🚀 QUICK TEST MODE ENABLED - Training will be fast but limited!\")\n"
    ]
    
    cell['source'] = new_source
    
    # 保存
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Notebook updated: {notebook_path}")
    print("  Cell 7 will now auto-detect local/Colab environment")
else:
    print("[ERROR] Could not find cell with BASE_DIR")

