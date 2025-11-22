#!/usr/bin/env python3
"""
本地运行训练脚本 - 适配本地 Jupyter 环境
使用本地 Google Drive 路径: G:\My Drive\datasets\bud500\data
"""

import os
import sys
import glob
import csv
import shutil
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, Audio, DatasetDict

print("=" * 60)
print("🚀 ASR Training - Local Jupyter Environment")
print("=" * 60)

# ========================================
# 1. 配置路径
# ========================================
BASE_DIR = r"G:\My Drive\datasets\bud500\data"
QUICK_TEST = True  # 设置为 False 进行完整训练

MAX_SHARDS_TRAIN = 1 if QUICK_TEST else None
MAX_SHARDS_VAL   = 1 if QUICK_TEST else None
MAX_SHARDS_TEST  = 1 if QUICK_TEST else None
MAX_EXAMPLES_TRAIN = 10 if QUICK_TEST else None
MAX_EXAMPLES_VAL   = 5  if QUICK_TEST else None
MAX_EXAMPLES_TEST  = 5  if QUICK_TEST else None

print(f"\n📁 数据目录: {BASE_DIR}")
if not os.path.exists(BASE_DIR):
    print(f"❌ 目录不存在: {BASE_DIR}")
    print("   请检查路径是否正确")
    sys.exit(1)
print(f"✓ 目录存在")

# ========================================
# 2. 设置工作目录
# ========================================
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

print(f"\n📂 项目目录: {project_root}")

# ========================================
# 3. 查找 parquet 文件
# ========================================
print("\n" + "=" * 60)
print("🔍 查找 parquet 文件")
print("=" * 60)

train_files = sorted(glob.glob(str(Path(BASE_DIR) / "train-*.parquet")))
val_files   = sorted(glob.glob(str(Path(BASE_DIR) / "validation-*.parquet")))
test_files  = sorted(glob.glob(str(Path(BASE_DIR) / "test-*.parquet")))

if not val_files and not test_files:
    all_files = sorted(glob.glob(str(Path(BASE_DIR) / "*.parquet")))
    if all_files:
        val_files = all_files[:1]
        test_files = all_files[1:2] if len(all_files) > 1 else []
        train_files = all_files[2:] if len(all_files) > 2 else all_files

def _limit_shards(files, max_n):
    if max_n is not None and len(files) > max_n:
        return files[:max_n], True
    return files, False

train_files, cut_tr = _limit_shards(train_files, MAX_SHARDS_TRAIN)
val_files, cut_va   = _limit_shards(val_files, MAX_SHARDS_VAL)
test_files, cut_te  = _limit_shards(test_files, MAX_SHARDS_TEST)

print(f"找到:")
print(f"  - Train: {len(train_files)} shard(s)")
print(f"  - Validation: {len(val_files)} shard(s)")
print(f"  - Test: {len(test_files)} shard(s)")

if QUICK_TEST:
    print(f"\n⚠️ QUICK_TEST 模式启用")
    if cut_tr: print(f"  使用前 {MAX_SHARDS_TRAIN} 个 train shard")
    if cut_va: print(f"  使用前 {MAX_SHARDS_VAL} 个 validation shard")
    if cut_te: print(f"  使用前 {MAX_SHARDS_TEST} 个 test shard")

# ========================================
# 4. 加载数据集
# ========================================
print("\n" + "=" * 60)
print("📦 加载数据集")
print("=" * 60)

AUDIO_PATH_COL = "audio_path"
TEXT_COL = "text"
TARGET_SR = 16000

data_files = {}
if train_files: data_files["train"] = train_files
if val_files:   data_files["validation"] = val_files
if test_files:  data_files["test"] = test_files

print("加载 parquet 文件...")
raw_datasets = load_dataset("parquet", data_files=data_files)

def _limit_examples(ds, max_n):
    if max_n is not None and len(ds) > max_n:
        return ds.select(range(max_n))
    return ds

if QUICK_TEST:
    if "train" in raw_datasets:
        raw_datasets["train"] = _limit_examples(raw_datasets["train"], MAX_EXAMPLES_TRAIN)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = _limit_examples(raw_datasets["validation"], MAX_EXAMPLES_VAL)
    if "test" in raw_datasets:
        raw_datasets["test"] = _limit_examples(raw_datasets["test"], MAX_EXAMPLES_TEST)

def ensure_audio_column(ds):
    cols = ds.column_names
    if "audio" not in cols:
        assert AUDIO_PATH_COL in cols, f"缺少列 {AUDIO_PATH_COL}"
        ds = ds.rename_column(AUDIO_PATH_COL, "audio_path_tmp")
        ds = ds.map(lambda x: {"audio": {"path": x["audio_path_tmp"]}}, remove_columns=["audio_path_tmp"])
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    return ds

processed = {}
for split in raw_datasets.keys():
    processed[split] = ensure_audio_column(raw_datasets[split])

datasets = DatasetDict(processed)
print(f"✓ 数据集加载完成")
for split, ds in datasets.items():
    print(f"  {split}: {len(ds)} 个样本")

# ========================================
# 5. 导出 CSV
# ========================================
print("\n" + "=" * 60)
print("💾 导出 CSV 文件")
print("=" * 60)

EXPORT_BASE = Path('data/external')
EXPORT_BASE.mkdir(parents=True, exist_ok=True)
CSV_PATH = EXPORT_BASE / ('parquet_quick.csv' if QUICK_TEST else 'parquet_full.csv')

rows = []
copied_count = 0
skipped_count = 0

for split in ["train", "validation", "test"]:
    if split not in datasets:
        continue
    
    ds = datasets[split]
    print(f"处理 {split} split: {len(ds)} 个样本")
    
    for idx, ex in enumerate(ds):
        try:
            audio_info = ex.get("audio", None)
            if audio_info is None:
                skipped_count += 1
                continue
            
            if isinstance(audio_info, dict):
                src = audio_info.get("path", None)
            else:
                src = getattr(audio_info, 'path', None)
            
            if not src or not isinstance(src, str):
                skipped_count += 1
                continue
            
            txt = ex.get(TEXT_COL, None)
            if txt is None or not str(txt).strip():
                skipped_count += 1
                continue
            
            src_p = Path(src)
            if not src_p.exists():
                skipped_count += 1
                continue
            
            dst = EXPORT_BASE / src_p.name
            if not dst.exists() or dst.stat().st_size == 0:
                try:
                    shutil.copy2(src, dst)
                    copied_count += 1
                except Exception:
                    skipped_count += 1
                    continue
            
            rows.append({
                "file_path": dst.relative_to(EXPORT_BASE).as_posix(),
                "transcript": str(txt).strip()
            })
        except Exception as e:
            skipped_count += 1
            if idx < 3:
                print(f"  错误处理 {split}[{idx}]: {e}")

if rows:
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcript"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ CSV 已写入: {CSV_PATH}")
    print(f"  总行数: {len(rows)}")
    print(f"  复制文件: {copied_count}")
    print(f"  跳过: {skipped_count}")
else:
    print("❌ 没有有效数据可导出")
    sys.exit(1)

# ========================================
# 6. 导入数据库
# ========================================
print("\n" + "=" * 60)
print("🗄️  导入数据到数据库")
print("=" * 60)

result = subprocess.run([
    sys.executable, 'scripts/prepare_data.py',
    '--csv', str(CSV_PATH),
    '--audio_base', 'data/external',
    '--auto_split', '--skip_duplicates'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode != 0:
    print("❌ 数据导入失败")
    sys.exit(1)

print("✓ 数据导入成功")

# ========================================
# 7. 配置训练
# ========================================
print("\n" + "=" * 60)
print("⚙️  配置训练")
print("=" * 60)

# 本地保存checkpoint
checkpoint_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d-%H%M%S')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['checkpoint_dir'] = str(checkpoint_dir)

if QUICK_TEST:
    cfg['num_epochs'] = 1
    cfg['batch_size'] = 4
    print("⚠️ QUICK_TEST: num_epochs=1, batch_size=4")

with open('configs/local.yaml', 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"✓ 配置文件: configs/local.yaml")
print(f"✓ Checkpoint 目录: {checkpoint_dir}")

# ========================================
# 8. 开始训练
# ========================================
print("\n" + "=" * 60)
print("🚀 开始训练")
print("=" * 60)

print(f"配置: configs/local.yaml")
print(f"Epochs: {cfg.get('num_epochs', 'N/A')}")
print(f"Batch size: {cfg.get('batch_size', 'N/A')}")
print("-" * 60)

result = subprocess.run(
    [sys.executable, 'training/train.py', '--config', 'configs/local.yaml'],
    cwd=project_root,
    env=dict(os.environ, PYTHONUNBUFFERED='1')
)

print("-" * 60)
if result.returncode == 0:
    print("✅ 训练完成！")
    print(f"Checkpoints 保存在: {checkpoint_dir}")
else:
    print(f"❌ 训练失败，退出码: {result.returncode}")

sys.exit(result.returncode)

