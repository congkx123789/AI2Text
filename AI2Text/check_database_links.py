"""
Script để kiểm tra tất cả các link file trong database
Kiểm tra xem các file được tham chiếu trong database có tồn tại trên disk không
Kiểm tra logic: absolute vs relative paths, duplicates, normalization, etc.
"""

import sqlite3
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import os


def check_audio_files(db_path: str, project_root: Path = None) -> Dict:
    """Kiểm tra tất cả audio file paths trong database với logic checking"""
    results = {
        'total': 0,
        'exists': 0,
        'missing': 0,
        'missing_files': [],
        'absolute_paths': 0,
        'relative_paths': 0,
        'duplicate_paths': [],
        'normalized_issues': [],
        'symlinks': 0,
        'path_format_issues': []
    }
    
    if project_root is None:
        project_root = Path.cwd()
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Track paths for duplicate detection
    path_to_ids = defaultdict(list)
    normalized_paths = {}
    
    try:
        # Lấy tất cả audio files
        cursor = conn.execute("SELECT id, file_path FROM AudioFiles")
        rows = cursor.fetchall()
        results['total'] = len(rows)
        
        print(f"\n🔍 Kiểm tra {results['total']} audio files (với logic checking)...")
        
        for row in tqdm(rows, desc="Checking audio files"):
            file_path_str = row['file_path']
            audio_id = row['id']
            
            # Track for duplicates
            path_to_ids[file_path_str].append(audio_id)
            
            # Check absolute vs relative
            path_obj = Path(file_path_str)
            if path_obj.is_absolute():
                results['absolute_paths'] += 1
            else:
                results['relative_paths'] += 1
            
            # Normalize path and check for issues
            try:
                if path_obj.is_absolute():
                    normalized = path_obj.resolve()
                else:
                    normalized = (project_root / path_obj).resolve()
                
                normalized_str = str(normalized)
                
                # Only flag as issue if:
                # 1. Path contains .. or . components that were resolved
                # 2. Path has redundant separators or components
                # 3. Normalized path is significantly different (not just relative->absolute conversion)
                has_issues = False
                issue_reason = None
                
                # Check for .. or . in original path
                if '..' in file_path_str or (file_path_str.startswith('./') or '/./' in file_path_str):
                    # Check if normalization changed the path structure
                    original_clean = str(Path(file_path_str).as_posix()).replace('..', '').replace('./', '')
                    normalized_clean = normalized_str.replace(str(project_root), '').replace('//', '/')
                    if original_clean != normalized_clean.replace(str(project_root), ''):
                        has_issues = True
                        issue_reason = "Contains .. or . components"
                
                # Check for redundant path components
                if '//' in file_path_str or file_path_str.endswith('/'):
                    has_issues = True
                    issue_reason = "Redundant path separators"
                
                # Check if path resolution changed structure significantly
                if path_obj.is_absolute():
                    # For absolute paths, check if resolve() changed anything
                    if str(path_obj) != normalized_str and str(path_obj.resolve()) != normalized_str:
                        has_issues = True
                        issue_reason = "Absolute path resolution changed structure"
                else:
                    # For relative paths, only flag if there's an actual issue
                    # (not just the normal relative->absolute conversion)
                    pass  # Relative paths are expected to resolve differently
                
                if has_issues and file_path_str not in normalized_paths:
                    normalized_paths[file_path_str] = normalized_str
                    results['normalized_issues'].append({
                        'id': audio_id,
                        'original': file_path_str,
                        'normalized': normalized_str,
                        'reason': issue_reason
                    })
            except (OSError, ValueError) as e:
                # Path resolution failed
                results['path_format_issues'].append({
                    'id': audio_id,
                    'path': file_path_str,
                    'error': str(e)
                })
            
            # Check if file exists
            try:
                if path_obj.is_absolute():
                    exists = path_obj.exists()
                    check_path = path_obj
                else:
                    # Try relative to project root
                    check_path = project_root / path_obj
                    exists = check_path.exists()
                
                if exists:
                    results['exists'] += 1
                    
                    # Check if it's a symlink
                    if check_path.is_symlink():
                        results['symlinks'] += 1
                else:
                    results['missing'] += 1
                    results['missing_files'].append({
                        'type': 'audio',
                        'id': audio_id,
                        'path': file_path_str,
                        'is_absolute': path_obj.is_absolute()
                    })
            except (OSError, ValueError) as e:
                # Path check failed
                results['missing'] += 1
                results['missing_files'].append({
                    'type': 'audio',
                    'id': audio_id,
                    'path': file_path_str,
                    'is_absolute': path_obj.is_absolute(),
                    'error': str(e)
                })
        
        # Find duplicate paths
        for path_str, ids in path_to_ids.items():
            if len(ids) > 1:
                results['duplicate_paths'].append({
                    'path': path_str,
                    'count': len(ids),
                    'ids': ids[:5]  # Show first 5 IDs
                })
        
    finally:
        conn.close()
    
    return results


def check_model_checkpoints(db_path: str) -> Dict:
    """Kiểm tra tất cả model checkpoint paths"""
    results = {
        'total': 0,
        'exists': 0,
        'missing': 0,
        'missing_files': []
    }
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Lấy tất cả model checkpoints
        cursor = conn.execute("""
            SELECT id, model_name, checkpoint_path 
            FROM Models 
            WHERE checkpoint_path IS NOT NULL AND checkpoint_path != ''
        """)
        rows = cursor.fetchall()
        results['total'] = len(rows)
        
        if results['total'] > 0:
            print(f"\n🔍 Kiểm tra {results['total']} model checkpoints...")
            
            for row in tqdm(rows, desc="Checking checkpoints"):
                checkpoint_path = row['checkpoint_path']
                model_id = row['id']
                model_name = row['model_name']
                
                if Path(checkpoint_path).exists():
                    results['exists'] += 1
                else:
                    results['missing'] += 1
                    results['missing_files'].append({
                        'type': 'checkpoint',
                        'id': model_id,
                        'model_name': model_name,
                        'path': checkpoint_path
                    })
        else:
            print("\n⚠️  Không có model checkpoints nào trong database")
    
    finally:
        conn.close()
    
    return results


def check_orphaned_records(db_path: str) -> Dict:
    """Kiểm tra các record bị orphan (không có transcript, không có split)"""
    results = {
        'audio_without_transcript': 0,
        'audio_without_split': 0,
        'transcript_without_audio': 0,
        'split_without_audio': 0
    }
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Audio files không có transcript
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM AudioFiles af
            LEFT JOIN Transcripts t ON af.id = t.audio_file_id
            WHERE t.id IS NULL
        """)
        results['audio_without_transcript'] = cursor.fetchone()['count']
        
        # Audio files không có split assignment
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM AudioFiles af
            LEFT JOIN DataSplits ds ON af.id = ds.audio_file_id
            WHERE ds.id IS NULL
        """)
        results['audio_without_split'] = cursor.fetchone()['count']
        
        # Transcripts không có audio file (shouldn't happen due to FK, but check anyway)
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM Transcripts t
            LEFT JOIN AudioFiles af ON t.audio_file_id = af.id
            WHERE af.id IS NULL
        """)
        results['transcript_without_audio'] = cursor.fetchone()['count']
        
        # Splits không có audio file (shouldn't happen due to FK, but check anyway)
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM DataSplits ds
            LEFT JOIN AudioFiles af ON ds.audio_file_id = af.id
            WHERE af.id IS NULL
        """)
        results['split_without_audio'] = cursor.fetchone()['count']
        
    finally:
        conn.close()
    
    return results


def generate_report(audio_results: Dict, checkpoint_results: Dict, orphan_results: Dict, 
                   output_file: str = None):
    """Tạo báo cáo tổng hợp với logic checking"""
    print("\n" + "=" * 70)
    print("📊 BÁO CÁO KIỂM TRA DATABASE LINKS (VỚI LOGIC CHECKING)")
    print("=" * 70)
    
    # Audio Files - Basic stats
    print(f"\n🎵 AUDIO FILES - BASIC:")
    print(f"   Tổng số: {audio_results['total']}")
    print(f"   ✅ Tồn tại: {audio_results['exists']}")
    print(f"   ❌ Thiếu: {audio_results['missing']}")
    
    # Path type analysis
    print(f"\n🔗 PATH LOGIC ANALYSIS:")
    print(f"   Absolute paths: {audio_results['absolute_paths']}")
    print(f"   Relative paths: {audio_results['relative_paths']}")
    print(f"   Symlinks: {audio_results['symlinks']}")
    
    # Duplicate paths
    if audio_results['duplicate_paths']:
        print(f"\n   ⚠️  DUPLICATE PATHS: {len(audio_results['duplicate_paths'])}")
        print(f"   Top 5 duplicates:")
        for dup in audio_results['duplicate_paths'][:5]:
            print(f"      - Path: {dup['path']}")
            print(f"        Duplicated {dup['count']} times (IDs: {dup['ids']})")
    else:
        print(f"\n   ✅ No duplicate paths found")
    
    # Normalized path issues
    if audio_results['normalized_issues']:
        print(f"\n   ⚠️  PATH NORMALIZATION ISSUES: {len(audio_results['normalized_issues'])}")
        print(f"   Top 5 issues:")
        for issue in audio_results['normalized_issues'][:5]:
            print(f"      - ID {issue['id']}:")
            print(f"        Original: {issue['original']}")
            print(f"        Normalized: {issue['normalized']}")
    else:
        print(f"\n   ✅ No path normalization issues")
    
    # Path format issues
    if audio_results['path_format_issues']:
        print(f"\n   ⚠️  PATH FORMAT ISSUES: {len(audio_results['path_format_issues'])}")
        print(f"   Top 5 issues:")
        for issue in audio_results['path_format_issues'][:5]:
            print(f"      - ID {issue['id']}: {issue['path']}")
            print(f"        Error: {issue['error']}")
    else:
        print(f"\n   ✅ No path format issues")
    
    if audio_results['missing'] > 0:
        print(f"\n   ❌ MISSING FILES: {audio_results['missing']}")
        print(f"   Top 10 file thiếu:")
        for item in audio_results['missing_files'][:10]:
            path_type = "absolute" if item.get('is_absolute', False) else "relative"
            print(f"      - ID {item['id']} ({path_type}): {item['path']}")
    
    # Model Checkpoints
    if checkpoint_results['total'] > 0:
        print(f"\n💾 MODEL CHECKPOINTS:")
        print(f"   Tổng số: {checkpoint_results['total']}")
        print(f"   ✅ Tồn tại: {checkpoint_results['exists']}")
        print(f"   ❌ Thiếu: {checkpoint_results['missing']}")
        
        if checkpoint_results['missing'] > 0:
            print(f"\n   Checkpoints thiếu:")
            for item in checkpoint_results['missing_files']:
                print(f"      - Model '{item['model_name']}' (ID {item['id']}): {item['path']}")
    
    # Orphaned Records
    print(f"\n🔗 ORPHANED RECORDS:")
    print(f"   Audio files không có transcript: {orphan_results['audio_without_transcript']}")
    print(f"   Audio files không có split: {orphan_results['audio_without_split']}")
    print(f"   Transcripts không có audio (lỗi FK): {orphan_results['transcript_without_audio']}")
    print(f"   Splits không có audio (lỗi FK): {orphan_results['split_without_audio']}")
    
    # Tổng kết
    total_missing = audio_results['missing'] + checkpoint_results['missing']
    total_checked = audio_results['total'] + checkpoint_results['total']
    
    print(f"\n{'=' * 70}")
    print(f"📈 TỔNG KẾT:")
    print(f"   Tổng số file kiểm tra: {total_checked}")
    print(f"   File tồn tại: {audio_results['exists'] + checkpoint_results['exists']}")
    print(f"   File thiếu: {total_missing}")
    
    if total_missing > 0:
        print(f"\n   ⚠️  CẢNH BÁO: Có {total_missing} file không tồn tại!")
        print(f"   Hãy kiểm tra và cập nhật database hoặc khôi phục các file này.")
    else:
        print(f"\n   ✅ Tất cả file links đều hợp lệ!")
    
    print("=" * 70)
    
    # Lưu vào file nếu được yêu cầu
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO KIỂM TRA DATABASE LINKS (VỚI LOGIC CHECKING)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("AUDIO FILES - BASIC:\n")
            f.write(f"  Tổng số: {audio_results['total']}\n")
            f.write(f"  Tồn tại: {audio_results['exists']}\n")
            f.write(f"  Thiếu: {audio_results['missing']}\n\n")
            
            f.write("PATH LOGIC ANALYSIS:\n")
            f.write(f"  Absolute paths: {audio_results['absolute_paths']}\n")
            f.write(f"  Relative paths: {audio_results['relative_paths']}\n")
            f.write(f"  Symlinks: {audio_results['symlinks']}\n\n")
            
            if audio_results['duplicate_paths']:
                f.write(f"DUPLICATE PATHS ({len(audio_results['duplicate_paths'])}):\n")
                for dup in audio_results['duplicate_paths']:
                    f.write(f"  Path: {dup['path']}\n")
                    f.write(f"  Duplicated {dup['count']} times\n")
                    f.write(f"  IDs: {dup['ids']}\n\n")
            
            if audio_results['normalized_issues']:
                f.write(f"PATH NORMALIZATION ISSUES ({len(audio_results['normalized_issues'])}):\n")
                for issue in audio_results['normalized_issues']:
                    f.write(f"  ID {issue['id']}:\n")
                    f.write(f"    Original: {issue['original']}\n")
                    f.write(f"    Normalized: {issue['normalized']}\n\n")
            
            if audio_results['path_format_issues']:
                f.write(f"PATH FORMAT ISSUES ({len(audio_results['path_format_issues'])}):\n")
                for issue in audio_results['path_format_issues']:
                    f.write(f"  ID {issue['id']}: {issue['path']}\n")
                    f.write(f"    Error: {issue['error']}\n\n")
            
            if audio_results['missing'] > 0:
                f.write("MISSING FILES:\n")
                for item in audio_results['missing_files']:
                    path_type = "absolute" if item.get('is_absolute', False) else "relative"
                    f.write(f"  ID {item['id']} ({path_type}): {item['path']}\n")
            
            if checkpoint_results['total'] > 0:
                f.write("\nMODEL CHECKPOINTS:\n")
                f.write(f"  Tổng số: {checkpoint_results['total']}\n")
                f.write(f"  Tồn tại: {checkpoint_results['exists']}\n")
                f.write(f"  Thiếu: {checkpoint_results['missing']}\n\n")
                
                if checkpoint_results['missing'] > 0:
                    f.write("Danh sách checkpoint thiếu:\n")
                    for item in checkpoint_results['missing_files']:
                        f.write(f"  Model '{item['model_name']}' (ID {item['id']}): {item['path']}\n")
            
            f.write("\nORPHANED RECORDS:\n")
            f.write(f"  Audio files không có transcript: {orphan_results['audio_without_transcript']}\n")
            f.write(f"  Audio files không có split: {orphan_results['audio_without_split']}\n")
        
        print(f"\n💾 Báo cáo đã lưu vào: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Kiểm tra tất cả file links trong database'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='database/asr_training.db',
        help='Đường dẫn đến database file (mặc định: database/asr_training.db)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='database_links_report.txt',
        help='File output cho báo cáo (mặc định: database_links_report.txt)'
    )
    parser.add_argument(
        '--skip-checkpoints',
        action='store_true',
        help='Bỏ qua kiểm tra model checkpoints'
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    
    if not db_path.exists():
        print(f"❌ Không tìm thấy database: {db_path}")
        return
    
    print(f"📂 Database: {db_path.absolute()}")
    print(f"📏 Kích thước: {db_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Determine project root (parent of database directory)
    project_root = db_path.parent.parent if db_path.parent.name == 'database' else Path.cwd()
    print(f"📁 Project root: {project_root.absolute()}")
    
    # Kiểm tra audio files với logic checking
    audio_results = check_audio_files(str(db_path), project_root)
    
    # Kiểm tra model checkpoints
    checkpoint_results = {'total': 0, 'exists': 0, 'missing': 0, 'missing_files': []}
    if not args.skip_checkpoints:
        checkpoint_results = check_model_checkpoints(str(db_path))
    
    # Kiểm tra orphaned records
    orphan_results = check_orphaned_records(str(db_path))
    
    # Tạo báo cáo
    generate_report(audio_results, checkpoint_results, orphan_results, args.output)


if __name__ == "__main__":
    main()

