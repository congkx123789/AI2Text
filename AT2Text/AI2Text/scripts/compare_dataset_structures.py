#!/usr/bin/env python3
"""
Compare folder structure and file formats between VietSpeech and librispeech_alignments.
Ensures both datasets have the same structure and format.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict


def check_directory_structure(base_dir, dataset_name):
    """Check directory structure of a dataset."""
    print(f"\n{'='*70}")
    print(f"Checking {dataset_name} directory structure")
    print(f"{'='*70}")
    
    structure = {}
    issues = []
    
    for split in ['test', 'train', 'val']:
        split_dir = base_dir / split
        structure[split] = {}
        
        if not split_dir.exists():
            issues.append(f"  ✗ {split}/ directory missing")
            continue
        
        # Check required files
        required_files = ['timestamps.json', 'manifest.csv']
        for file in required_files:
            file_path = split_dir / file
            if file_path.exists():
                structure[split][file] = True
            else:
                structure[split][file] = False
                issues.append(f"  ✗ {split}/{file} missing")
        
        # Check audio directory
        audio_dir = split_dir / 'audio'
        if audio_dir.exists() and audio_dir.is_dir():
            structure[split]['audio'] = True
        else:
            structure[split]['audio'] = False
            issues.append(f"  ✗ {split}/audio/ directory missing")
    
    # Print structure
    for split in ['test', 'train', 'val']:
        print(f"\n{split.upper()} split:")
        for item, exists in structure[split].items():
            status = "✓" if exists else "✗"
            print(f"  {status} {item}")
    
    if issues:
        print(f"\n⚠ Issues found:")
        for issue in issues:
            print(issue)
    else:
        print(f"\n✓ Directory structure is correct")
    
    return structure, issues


def check_timestamps_json_format(json_path, dataset_name, split_name):
    """Check timestamps.json format."""
    print(f"\n  Checking timestamps.json format for {dataset_name}/{split_name}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    issues = []
    required_fields = ['duration', 'text', 'segments', 'audio_filepath']
    
    # Check first few entries
    sample_count = min(5, len(data))
    sample_keys = list(data.keys())[:sample_count]
    
    for key in sample_keys:
        entry = data[key]
        
        # Check required fields
        for field in required_fields:
            if field not in entry:
                issues.append(f"    ✗ Entry '{key}' missing field: {field}")
        
        # Check segments structure
        if 'segments' in entry:
            if not isinstance(entry['segments'], list):
                issues.append(f"    ✗ Entry '{key}': segments is not a list")
            elif len(entry['segments']) > 0:
                segment = entry['segments'][0]
                segment_fields = ['word', 'start', 'end', 'score']
                for field in segment_fields:
                    if field not in segment:
                        issues.append(f"    ✗ Entry '{key}': segment missing field: {field}")
    
    if issues:
        print(f"    ⚠ Found {len(issues)} issues in sample entries")
        for issue in issues[:5]:
            print(issue)
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print(f"    ✓ Format is correct (checked {sample_count} sample entries)")
    
    return {
        'total_entries': len(data),
        'sample_checked': sample_count,
        'issues': issues
    }


def check_manifest_csv_format(csv_path, dataset_name, split_name):
    """Check manifest.csv format."""
    print(f"\n  Checking manifest.csv format for {dataset_name}/{split_name}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {'total_rows': 0, 'issues': ['CSV file is empty']}
    
    required_columns = ['id', 'transcript', 'audio_path', 'words_json', 'sex', 'subset']
    issues = []
    
    # Check header
    header = rows[0].keys() if rows else []
    for col in required_columns:
        if col not in header:
            issues.append(f"    ✗ Missing column: {col}")
    
    # Check sample rows
    sample_count = min(5, len(rows))
    for i, row in enumerate(rows[:sample_count]):
        for col in required_columns:
            if col not in row or not row[col]:
                issues.append(f"    ✗ Row {i+2}: missing or empty {col}")
    
    if issues:
        print(f"    ⚠ Found {len(issues)} issues")
        for issue in issues[:5]:
            print(issue)
    else:
        print(f"    ✓ Format is correct (checked {sample_count} sample rows)")
    
    return {
        'total_rows': len(rows),
        'sample_checked': sample_count,
        'issues': issues
    }


def compare_formats(vietspeech_base, librispeech_base):
    """Compare formats between the two datasets."""
    print(f"\n{'='*70}")
    print("COMPARING FORMATS")
    print(f"{'='*70}")
    
    differences = []
    
    for split in ['test', 'train', 'val']:
        print(f"\n{split.upper()} split comparison:")
        
        vs_json = vietspeech_base / split / 'timestamps.json'
        ls_json = librispeech_base / split / 'timestamps.json'
        
        vs_csv = vietspeech_base / split / 'manifest.csv'
        ls_csv = librispeech_base / split / 'manifest.csv'
        
        # Compare JSON structure
        if vs_json.exists() and ls_json.exists():
            with open(vs_json, 'r') as f:
                vs_data = json.load(f)
            with open(ls_json, 'r') as f:
                ls_data = json.load(f)
            
            # Get sample entries
            vs_key = list(vs_data.keys())[0] if vs_data else None
            ls_key = list(ls_data.keys())[0] if ls_data else None
            
            if vs_key and ls_key:
                vs_entry = vs_data[vs_key]
                ls_entry = ls_data[ls_key]
                
                # Check fields
                vs_fields = set(vs_entry.keys())
                ls_fields = set(ls_entry.keys())
                
                if vs_fields != ls_fields:
                    missing_in_vs = ls_fields - vs_fields
                    missing_in_ls = vs_fields - ls_fields
                    if missing_in_vs:
                        differences.append(f"  {split}: VietSpeech missing fields: {missing_in_vs}")
                    if missing_in_ls:
                        differences.append(f"  {split}: librispeech missing fields: {missing_in_ls}")
                else:
                    print(f"    ✓ JSON fields match: {vs_fields}")
        
        # Compare CSV structure
        if vs_csv.exists() and ls_csv.exists():
            with open(vs_csv, 'r') as f:
                vs_reader = csv.DictReader(f)
                vs_headers = set(vs_reader.fieldnames or [])
            
            with open(ls_csv, 'r') as f:
                ls_reader = csv.DictReader(f)
                ls_headers = set(ls_reader.fieldnames or [])
            
            if vs_headers != ls_headers:
                missing_in_vs = ls_headers - vs_headers
                missing_in_ls = vs_headers - ls_headers
                if missing_in_vs:
                    differences.append(f"  {split}: VietSpeech CSV missing columns: {missing_in_vs}")
                if missing_in_ls:
                    differences.append(f"  {split}: librispeech CSV missing columns: {missing_in_ls}")
            else:
                print(f"    ✓ CSV columns match: {vs_headers}")
    
    if differences:
        print(f"\n⚠ Format differences found:")
        for diff in differences:
            print(diff)
    else:
        print(f"\n✓ Formats match between datasets")
    
    return differences


def main():
    """Main comparison function."""
    base_data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    vietspeech_dir = base_data_dir / "VietSpeech"
    librispeech_dir = base_data_dir / "librispeech_alignments"
    
    print("="*70)
    print("DATASET STRUCTURE COMPARISON")
    print("="*70)
    
    # Check directory structures
    vs_structure, vs_issues = check_directory_structure(vietspeech_dir, "VietSpeech")
    ls_structure, ls_issues = check_directory_structure(librispeech_dir, "librispeech_alignments")
    
    # Check formats
    print(f"\n{'='*70}")
    print("CHECKING FILE FORMATS")
    print(f"{'='*70}")
    
    for split in ['test', 'train', 'val']:
        print(f"\n{split.upper()} split:")
        
        # VietSpeech
        vs_json = vietspeech_dir / split / 'timestamps.json'
        vs_csv = vietspeech_dir / split / 'manifest.csv'
        
        if vs_json.exists():
            check_timestamps_json_format(vs_json, "VietSpeech", split)
        if vs_csv.exists():
            check_manifest_csv_format(vs_csv, "VietSpeech", split)
        
        # librispeech_alignments
        ls_json = librispeech_dir / split / 'timestamps.json'
        ls_csv = librispeech_dir / split / 'manifest.csv'
        
        if ls_json.exists():
            check_timestamps_json_format(ls_json, "librispeech_alignments", split)
        if ls_csv.exists():
            check_manifest_csv_format(ls_csv, "librispeech_alignments", split)
    
    # Compare formats
    differences = compare_formats(vietspeech_dir, librispeech_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    all_issues = len(vs_issues) + len(ls_issues) + len(differences)
    
    if all_issues == 0:
        print("✓ Both datasets have identical structure and format!")
    else:
        print(f"⚠ Found {all_issues} issues/differences:")
        if vs_issues:
            print(f"  - VietSpeech structure issues: {len(vs_issues)}")
        if ls_issues:
            print(f"  - librispeech_alignments structure issues: {len(ls_issues)}")
        if differences:
            print(f"  - Format differences: {len(differences)}")
    
    print("="*70 + "\n")
    
    return all_issues == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


