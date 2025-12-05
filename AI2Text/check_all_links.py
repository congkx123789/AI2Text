"""
Comprehensive script to check all links in:
- CSV manifests (full_merged_dataset and merged_dataset)
- JSON timestamp files
- Audio files referenced
- Database entries
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Set
from collections import defaultdict


def check_csv_manifest(manifest_path: Path, dataset_root: Path) -> Dict:
    """Check all audio file links in a CSV manifest"""
    results = {
        'total': 0,
        'exists': 0,
        'missing': 0,
        'missing_files': [],
        'errors': []
    }
    
    try:
        df = pd.read_csv(manifest_path)
        results['total'] = len(df)
        
        # Determine base directory
        base_dir = manifest_path.parent
        
        # Get audio path column
        audio_col = None
        if 'audio_path' in df.columns:
            audio_col = 'audio_path'
        elif 'file_path' in df.columns:
            audio_col = 'file_path'
        else:
            results['errors'].append(f"No audio_path or file_path column found")
            return results
        
        for idx, row in df.iterrows():
            audio_path = row[audio_col]
            
            if pd.isna(audio_path):
                results['missing'] += 1
                results['missing_files'].append({
                    'row': idx,
                    'path': 'NaN',
                    'reason': 'Empty path'
                })
                continue
            
            # Build full path
            if Path(audio_path).is_absolute():
                full_path = Path(audio_path)
            else:
                full_path = base_dir / audio_path
            
            if full_path.exists():
                results['exists'] += 1
            else:
                results['missing'] += 1
                results['missing_files'].append({
                    'row': idx,
                    'path': str(audio_path),
                    'full_path': str(full_path)
                })
    
    except Exception as e:
        results['errors'].append(f"Error reading CSV: {str(e)}")
    
    return results


def check_json_timestamps(json_path: Path, dataset_root: Path) -> Dict:
    """Check all audio file links in a JSON timestamps file"""
    results = {
        'total': 0,
        'exists': 0,
        'missing': 0,
        'missing_files': [],
        'errors': []
    }
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        base_dir = json_path.parent
        
        # JSON structure can vary, try to find audio file references
        if isinstance(data, dict):
            # Check if it's a dict with file paths as keys
            for key, value in data.items():
                results['total'] += 1
                # Try to construct audio path from key
                if key.endswith('.wav') or key.endswith('.mp3') or key.endswith('.flac'):
                    audio_path = base_dir / "audio" / Path(key).name
                else:
                    # Key might be an ID, try common patterns
                    audio_path = base_dir / "audio" / f"{key}.wav"
                
                if audio_path.exists():
                    results['exists'] += 1
                else:
                    results['missing'] += 1
                    results['missing_files'].append({
                        'key': key,
                        'path': str(audio_path)
                    })
        elif isinstance(data, list):
            # List of entries
            for entry in data:
                results['total'] += 1
                if isinstance(entry, dict):
                    # Try to find file path in entry
                    file_key = None
                    for k in ['file', 'file_path', 'audio_path', 'path', 'id']:
                        if k in entry:
                            file_key = entry[k]
                            break
                    
                    if file_key:
                        if Path(file_key).is_absolute():
                            audio_path = Path(file_key)
                        else:
                            audio_path = base_dir / "audio" / Path(file_key).name
                        
                        if audio_path.exists():
                            results['exists'] += 1
                        else:
                            results['missing'] += 1
                            results['missing_files'].append({
                                'entry': entry,
                                'path': str(audio_path)
                            })
    
    except Exception as e:
        results['errors'].append(f"Error reading JSON: {str(e)}")
    
    return results


def check_database_audio_files(db_path: Path) -> Dict:
    """Check all audio file links in database"""
    results = {
        'total': 0,
        'exists': 0,
        'missing': 0,
        'missing_files': [],
        'absolute_paths': 0,
        'relative_paths': 0
    }
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("SELECT id, file_path FROM AudioFiles")
        rows = cursor.fetchall()
        results['total'] = len(rows)
        
        project_root = db_path.parent.parent
        
        for row in rows:
            file_path_str = row['file_path']
            audio_id = row['id']
            
            path_obj = Path(file_path_str)
            if path_obj.is_absolute():
                results['absolute_paths'] += 1
                check_path = path_obj
            else:
                results['relative_paths'] += 1
                check_path = project_root / path_obj
            
            if check_path.exists():
                results['exists'] += 1
            else:
                results['missing'] += 1
                results['missing_files'].append({
                    'id': audio_id,
                    'path': file_path_str
                })
        
        conn.close()
    
    except Exception as e:
        results['errors'] = [str(e)]
    
    return results


def check_dataset(dataset_path: Path, dataset_name: str) -> Dict:
    """Check all files in a dataset directory"""
    results = {
        'dataset': dataset_name,
        'splits': {},
        'total_files': 0,
        'total_missing': 0
    }
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        
        split_results = {
            'csv': None,
            'json': None,
            'total_missing': 0
        }
        
        # Check CSV manifest
        csv_path = split_dir / 'manifest.csv'
        if csv_path.exists():
            print(f"  Checking {split}/manifest.csv...")
            csv_results = check_csv_manifest(csv_path, dataset_path)
            split_results['csv'] = csv_results
            split_results['total_missing'] += csv_results['missing']
            results['total_missing'] += csv_results['missing']
        
        # Check JSON timestamps
        json_path = split_dir / 'timestamps.json'
        if json_path.exists():
            print(f"  Checking {split}/timestamps.json...")
            json_results = check_json_timestamps(json_path, dataset_path)
            split_results['json'] = json_results
            split_results['total_missing'] += json_results['missing']
            results['total_missing'] += json_results['missing']
        
        results['splits'][split] = split_results
        results['total_files'] += split_results.get('csv', {}).get('total', 0)
    
    return results


def generate_comprehensive_report(datasets_results: Dict, db_results: Dict, output_file: str = None):
    """Generate comprehensive report"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE LINK CHECK REPORT")
    print("=" * 80)
    
    # Dataset results
    print("\n📁 DATASET MANIFESTS:")
    for dataset_name, dataset_result in datasets_results.items():
        print(f"\n  {dataset_name.upper()}:")
        print(f"    Total files checked: {dataset_result['total_files']}")
        print(f"    Total missing: {dataset_result['total_missing']}")
        
        for split_name, split_result in dataset_result['splits'].items():
            print(f"\n    {split_name.upper()}:")
            
            if split_result['csv']:
                csv = split_result['csv']
                print(f"      CSV Manifest:")
                print(f"        Total: {csv['total']}")
                print(f"        ✅ Exists: {csv['exists']}")
                print(f"        ❌ Missing: {csv['missing']}")
                if csv['missing'] > 0 and csv['missing'] <= 10:
                    print(f"        Missing files:")
                    for item in csv['missing_files']:
                        print(f"          - Row {item.get('row', '?')}: {item.get('path', '?')}")
                elif csv['missing'] > 10:
                    print(f"        Missing files: {csv['missing']} (showing first 5)")
                    for item in csv['missing_files'][:5]:
                        print(f"          - Row {item.get('row', '?')}: {item.get('path', '?')}")
            
            if split_result['json']:
                json_data = split_result['json']
                print(f"      JSON Timestamps:")
                print(f"        Total: {json_data['total']}")
                print(f"        ✅ Exists: {json_data['exists']}")
                print(f"        ❌ Missing: {json_data['missing']}")
    
    # Database results
    if db_results:
        print(f"\n💾 DATABASE:")
        print(f"    Total audio files: {db_results['total']}")
        print(f"    ✅ Exists: {db_results['exists']}")
        print(f"    ❌ Missing: {db_results['missing']}")
        print(f"    Absolute paths: {db_results['absolute_paths']}")
        print(f"    Relative paths: {db_results['relative_paths']}")
        
        if db_results['missing'] > 0:
            print(f"\n    Missing files (first 10):")
            for item in db_results['missing_files'][:10]:
                print(f"      - ID {item['id']}: {item['path']}")
    
    # Summary
    total_missing = sum(d['total_missing'] for d in datasets_results.values())
    if db_results:
        total_missing += db_results['missing']
    
    print(f"\n{'=' * 80}")
    print(f"📈 SUMMARY:")
    print(f"    Total missing links: {total_missing}")
    
    if total_missing == 0:
        print(f"\n    ✅ ALL LINKS ARE VALID!")
    else:
        print(f"\n    ⚠️  FOUND {total_missing} MISSING/BROKEN LINKS")
        print(f"    Please review and fix the issues above.")
    
    print("=" * 80)
    
    # Save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE LINK CHECK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for dataset_name, dataset_result in datasets_results.items():
                f.write(f"{dataset_name.upper()}:\n")
                f.write(f"  Total missing: {dataset_result['total_missing']}\n\n")
                
                for split_name, split_result in dataset_result['splits'].items():
                    f.write(f"  {split_name.upper()}:\n")
                    
                    if split_result['csv']:
                        csv = split_result['csv']
                        f.write(f"    CSV: {csv['missing']} missing\n")
                        if csv['missing'] > 0:
                            for item in csv['missing_files']:
                                f.write(f"      Row {item.get('row', '?')}: {item.get('path', '?')}\n")
                    
                    if split_result['json']:
                        json_data = split_result['json']
                        f.write(f"    JSON: {json_data['missing']} missing\n")
            
            if db_results:
                f.write(f"\nDATABASE:\n")
                f.write(f"  Missing: {db_results['missing']}\n")
                if db_results['missing'] > 0:
                    for item in db_results['missing_files']:
                        f.write(f"    ID {item['id']}: {item['path']}\n")
        
        print(f"\n💾 Full report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive link checking for all datasets and database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comprehensive_links_report.txt',
        help='Output file for report'
    )
    parser.add_argument(
        '--skip-db',
        action='store_true',
        help='Skip database checking'
    )
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    
    print("🔍 Starting comprehensive link check...")
    print(f"📁 Project root: {project_root}")
    
    datasets_results = {}
    
    # Check full_merged_dataset
    full_merged_path = project_root / 'data' / 'processed' / 'full_merged_dataset'
    if full_merged_path.exists():
        print(f"\n📂 Checking full_merged_dataset...")
        datasets_results['full_merged_dataset'] = check_dataset(full_merged_path, 'full_merged_dataset')
    else:
        print(f"\n⚠️  full_merged_dataset not found")
    
    # Check merged_dataset
    merged_path = project_root / 'data' / 'processed' / 'merged_dataset'
    if merged_path.exists():
        print(f"\n📂 Checking merged_dataset...")
        datasets_results['merged_dataset'] = check_dataset(merged_path, 'merged_dataset')
    else:
        print(f"\n⚠️  merged_dataset not found")
    
    # Check database
    db_results = None
    if not args.skip_db:
        db_path = project_root / 'database' / 'asr_training.db'
        if db_path.exists():
            print(f"\n💾 Checking database...")
            db_results = check_database_audio_files(db_path)
        else:
            print(f"\n⚠️  Database not found")
    
    # Generate report
    generate_comprehensive_report(datasets_results, db_results, args.output)


if __name__ == "__main__":
    main()

