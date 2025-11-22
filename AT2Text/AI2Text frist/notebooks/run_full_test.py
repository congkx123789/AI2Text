#!/usr/bin/env python3
"""
完整测试流程：数据导入 + 训练 + 获取checkpoint
模拟 notebook 的完整执行流程
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

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 60)
print("完整测试流程：数据导入 + 训练")
print("=" * 60)

# ========================================
# 步骤 1: 加载 parquet 数据
# ========================================
print("\n步骤 1: 加载 parquet 数据...")
BASE_DIR = r"G:\My Drive\datasets\bud500\data"
QUICK_TEST = True

if not os.path.exists(BASE_DIR):
    print(f"[ERROR] 路径不存在: {BASE_DIR}")
    sys.exit(1)

# 查找文件
train_files = sorted(glob.glob(str(Path(BASE_DIR) / "train-*.parquet")))[:1]  # QUICK_TEST: 只用1个
val_files = sorted(glob.glob(str(Path(BASE_DIR) / "validation-*.parquet")))[:1]
test_files = sorted(glob.glob(str(Path(BASE_DIR) / "test-*.parquet")))[:1]

print(f"  找到 {len(train_files)} 个训练文件")
print(f"  找到 {len(val_files)} 个验证文件")
print(f"  找到 {len(test_files)} 个测试文件")

if not train_files:
    print("[ERROR] 未找到训练文件")
    sys.exit(1)

    # 直接读取 parquet 文件，避免 Audio 解码
print("\n  读取 parquet 文件...")
try:
    import pandas as pd
    
    AUDIO_PATH_COL = "audio_path"
    TEXT_COL = "text"
    
    datasets = {}
    
    if train_files:
        df_train = pd.read_parquet(train_files[0])
        if QUICK_TEST:
            df_train = df_train.head(10)
        datasets["train"] = df_train
        print(f"  train: {len(df_train)} 个样本")
    
    if val_files:
        df_val = pd.read_parquet(val_files[0])
        if QUICK_TEST:
            df_val = df_val.head(5)
        datasets["validation"] = df_val
        print(f"  validation: {len(df_val)} 个样本")
    
    if test_files:
        df_test = pd.read_parquet(test_files[0])
        if QUICK_TEST:
            df_test = df_test.head(5)
        datasets["test"] = df_test
        print(f"  test: {len(df_test)} 个样本")
    
    print(f"  [OK] 数据集加载完成")
        
except Exception as e:
    print(f"  [ERROR] 加载数据集失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================================
# 步骤 2: 导出 CSV
# ========================================
print("\n步骤 2: 导出 CSV...")
EXPORT_BASE = Path('data/external')
EXPORT_BASE.mkdir(parents=True, exist_ok=True)
CSV_PATH = EXPORT_BASE / 'parquet_quick.csv'

rows = []
for split in ["train", "validation", "test"]:
    if split not in datasets:
        continue
    
    df = datasets[split]
    print(f"  处理 {split}: {len(df)} 个样本")
    
    # 从DataFrame获取列名
    cols = df.columns.tolist()
    
    # 找到音频路径列 - 可能是audio列（包含路径信息）
    audio_col = None
    for col in ["audio", AUDIO_PATH_COL, "audio_path", "path", "file_path"]:
        if col in cols:
            audio_col = col
            break
    
    if not audio_col:
        print(f"    [WARN] 未找到音频路径列，可用列: {cols}")
        continue
    
    # 找到文本列
    text_col = None
    for col in ["transcription", TEXT_COL, "text", "transcript", "sentence"]:
        if col in cols:
            text_col = col
            break
    
    if not text_col:
        print(f"    [WARN] 未找到文本列，可用列: {cols}")
        continue
    
    # 遍历DataFrame
    for idx, row in df.iterrows():
        try:
            # 处理audio列 - 可能是字典（包含bytes或path）
            audio_val = row[audio_col]
            audio_bytes = None
            src = None
            
            if isinstance(audio_val, dict):
                # 优先使用path，如果没有则使用bytes
                src = audio_val.get("path", None)
                if not src:
                    audio_bytes = audio_val.get("bytes", None)
            elif isinstance(audio_val, (bytes, bytearray)):
                audio_bytes = audio_val
            elif isinstance(audio_val, str):
                src = audio_val
            
            txt = str(row[text_col]) if pd.notna(row[text_col]) else None
            if not txt or not txt.strip():
                continue
            
            # 如果有路径，直接使用
            if src and isinstance(src, str) and src != 'nan' and src != 'None':
                src_p = Path(src)
                if src_p.exists():
                    dst = EXPORT_BASE / src_p.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    rows.append({
                        "file_path": dst.relative_to(EXPORT_BASE).as_posix(),
                        "transcript": txt.strip()
                    })
                    continue
            
            # 如果有bytes数据，保存为文件
            if audio_bytes:
                # 生成唯一文件名
                import hashlib
                audio_hash = hashlib.md5(audio_bytes[:1000] if len(audio_bytes) > 1000 else audio_bytes).hexdigest()
                dst = EXPORT_BASE / f"{split}_{idx}_{audio_hash}.wav"
                
                if not dst.exists():
                    # 保存bytes为wav文件
                    with open(dst, 'wb') as f:
                        f.write(audio_bytes)
                
                rows.append({
                    "file_path": dst.relative_to(EXPORT_BASE).as_posix(),
                    "transcript": txt.strip()
                })
                continue
                
        except Exception as e:
            if idx < 3:
                print(f"    错误处理 {split}[{idx}]: {e}")
            continue

if rows:
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcript"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] CSV已导出: {CSV_PATH} ({len(rows)} 行)")
else:
    print("  [ERROR] 没有数据可导出")
    sys.exit(1)

# ========================================
# 步骤 3: 导入数据库
# ========================================
print("\n步骤 3: 导入数据到数据库...")
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
    print("  [ERROR] 数据导入失败")
    sys.exit(1)

print("  [OK] 数据导入成功")

# ========================================
# 步骤 4: 配置训练
# ========================================
print("\n步骤 4: 配置训练...")
with open('configs/default.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['num_epochs'] = 1
cfg['batch_size'] = 2
cfg['checkpoint_dir'] = 'checkpoints/test_run'

checkpoint_dir = Path(cfg['checkpoint_dir'])
checkpoint_dir.mkdir(parents=True, exist_ok=True)

test_config_path = 'configs/test_training.yaml'
with open(test_config_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print(f"  [OK] 配置已保存: {test_config_path}")

# ========================================
# 步骤 5: 运行训练
# ========================================
print("\n" + "=" * 60)
print("步骤 5: 开始训练...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, 'training/train.py', '--config', test_config_path],
    cwd=project_root,
    env=dict(os.environ, PYTHONUNBUFFERED='1')
)

# ========================================
# 步骤 6: 检查 checkpoint
# ========================================
print("\n" + "=" * 60)
print("步骤 6: 检查 checkpoint...")
print("=" * 60)

checkpoint_files = list(checkpoint_dir.glob('*.pt')) + list(checkpoint_dir.glob('*.ckpt'))
if checkpoint_files:
    print(f"[OK] 找到 {len(checkpoint_files)} 个 checkpoint:")
    for ckpt in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"  {ckpt.name}")
        print(f"    大小: {size_mb:.2f} MB")
        print(f"    时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    路径: {ckpt.absolute()}")
    
    latest = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"\n最新 checkpoint: {latest.absolute()}")
else:
    print("[WARN] 未找到 checkpoint 文件")
    if result.returncode != 0:
        print(f"训练失败，退出码: {result.returncode}")

print("=" * 60)
if result.returncode == 0 and checkpoint_files:
    print("[OK] 测试完成！Checkpoint 已生成")
else:
    print("[WARN] 测试未完全成功")

