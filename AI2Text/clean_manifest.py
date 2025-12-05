"""
Script để tự động xóa các file lỗi khỏi manifest.csv
Dựa trên kết quả từ check_data.py
"""

import pandas as pd
from pathlib import Path
import argparse
import sys


def load_bad_files(bad_files_path: str) -> set:
    """Load danh sách file lỗi từ bad_files.txt"""
    bad_files = set()
    
    if not Path(bad_files_path).exists():
        print(f"❌ Không tìm thấy file: {bad_files_path}")
        print("   Hãy chạy check_data.py trước để tạo file bad_files.txt")
        return bad_files
    
    with open(bad_files_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Format: path,reason
                parts = line.split(',', 1)
                if len(parts) >= 1:
                    file_path = parts[0].strip()
                    bad_files.add(file_path)
    
    return bad_files


def clean_manifest(manifest_path: str, bad_files: set, backup: bool = True) -> int:
    """Xóa các file lỗi khỏi manifest và lưu backup"""
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        print(f"❌ Không tìm thấy manifest: {manifest_path}")
        return 0
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    original_count = len(df)
    
    # Xác định base directory để so sánh đường dẫn
    base_dir = manifest_path.parent
    
    # Tìm các dòng cần xóa
    rows_to_remove = []
    
    for idx, row in df.iterrows():
        # Lấy audio_path từ manifest
        if 'audio_path' in row:
            audio_path = row['audio_path']
        elif 'file_path' in row:
            audio_path = row['file_path']
        else:
            continue
        
        # Xây dựng đường dẫn đầy đủ để so sánh
        if Path(audio_path).is_absolute():
            full_path = Path(audio_path)
        else:
            full_path = base_dir / audio_path
        
        # So sánh với danh sách file lỗi
        if str(full_path) in bad_files:
            rows_to_remove.append(idx)
    
    if not rows_to_remove:
        print(f"✅ Không tìm thấy file lỗi nào trong manifest: {manifest_path}")
        return 0
    
    # Tạo backup nếu cần
    if backup:
        backup_path = manifest_path.with_suffix('.csv.backup')
        df.to_csv(backup_path, index=False)
        print(f"💾 Đã tạo backup: {backup_path}")
    
    # Xóa các dòng lỗi
    df_cleaned = df.drop(index=rows_to_remove).reset_index(drop=True)
    
    # Lưu manifest đã làm sạch
    df_cleaned.to_csv(manifest_path, index=False)
    
    removed_count = len(rows_to_remove)
    remaining_count = len(df_cleaned)
    
    print(f"✅ Đã xóa {removed_count} file lỗi khỏi manifest")
    print(f"   Trước: {original_count} files")
    print(f"   Sau: {remaining_count} files")
    print(f"   Đã lưu: {manifest_path}")
    
    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description='Xóa các file lỗi khỏi manifest.csv dựa trên bad_files.txt'
    )
    parser.add_argument(
        '--bad-files',
        type=str,
        default='bad_files.txt',
        help='Đường dẫn đến file bad_files.txt (mặc định: bad_files.txt)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        help='Đường dẫn đến manifest.csv cần làm sạch (mặc định: dùng manifest từ check_data.py)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='data/processed/merged_dataset',
        help='Root directory của dataset (mặc định: data/processed/merged_dataset)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Split cần làm sạch (mặc định: train)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Không tạo backup file'
    )
    parser.add_argument(
        '--all-splits',
        action='store_true',
        help='Làm sạch tất cả các splits (train, val, test)'
    )
    
    args = parser.parse_args()
    
    # Load danh sách file lỗi
    print(f"📖 Đang đọc danh sách file lỗi từ: {args.bad_files}")
    bad_files = load_bad_files(args.bad_files)
    
    if not bad_files:
        print("❌ Không có file lỗi nào để xóa")
        sys.exit(1)
    
    print(f"✅ Đã tải {len(bad_files)} file lỗi")
    
    # Xác định manifest cần làm sạch
    if args.manifest:
        manifest_paths = [args.manifest]
    elif args.all_splits:
        dataset_root = Path(args.dataset_root)
        manifest_paths = [
            dataset_root / 'train' / 'manifest.csv',
            dataset_root / 'val' / 'manifest.csv',
            dataset_root / 'test' / 'manifest.csv'
        ]
        # Chỉ xử lý các file tồn tại
        manifest_paths = [p for p in manifest_paths if p.exists()]
    else:
        dataset_root = Path(args.dataset_root)
        manifest_paths = [dataset_root / args.split / 'manifest.csv']
    
    # Làm sạch từng manifest
    total_removed = 0
    for manifest_path in manifest_paths:
        print(f"\n🔧 Đang làm sạch: {manifest_path}")
        removed = clean_manifest(
            str(manifest_path),
            bad_files,
            backup=not args.no_backup
        )
        total_removed += removed
    
    print(f"\n{'='*50}")
    print(f"✅ Hoàn thành! Đã xóa tổng cộng {total_removed} file lỗi")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

