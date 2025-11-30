#!/usr/bin/env python3
"""
Verify that all audio_filepath entries in VietSpeech timestamps.json files
match existing audio files.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def verify_audio_filepaths(timestamps_json_path, split_name):
    """
    Verify that all audio_filepath entries point to existing files.
    
    Args:
        timestamps_json_path: Path to timestamps.json file
        split_name: Name of the split (test/train/val) for reporting
        
    Returns:
        tuple: (total_entries, existing_files, missing_files, missing_list)
    """
    print(f"\n{'='*60}")
    print(f"Verifying {split_name} split")
    print(f"{'='*60}")
    
    # Get the base directory (parent of the split directory)
    split_dir = timestamps_json_path.parent
    base_dir = split_dir.parent
    
    # Load timestamps.json
    print(f"  Loading {timestamps_json_path.name}...")
    with open(timestamps_json_path, 'r', encoding='utf-8') as f:
        timestamps_data = json.load(f)
    
    total_entries = len(timestamps_data)
    existing_files = 0
    missing_files = 0
    missing_list = []
    
    print(f"  Checking {total_entries} entries...")
    
    # Check each entry
    for filename, entry_data in timestamps_data.items():
        if 'audio_filepath' not in entry_data:
            print(f"    ⚠ Warning: Entry {filename} missing audio_filepath field")
            missing_files += 1
            missing_list.append((filename, "Missing audio_filepath field"))
            continue
        
        audio_filepath = entry_data['audio_filepath']
        
        # Resolve path: audio_filepath is relative to the split directory
        # e.g., "audio/189_000001959.wav" -> split_dir/audio/189_000001959.wav
        full_path = split_dir / audio_filepath
        
        if full_path.exists() and full_path.is_file():
            existing_files += 1
        else:
            missing_files += 1
            missing_list.append((filename, str(full_path)))
    
    # Print results
    print(f"\n  Results for {split_name}:")
    print(f"    Total entries: {total_entries}")
    print(f"    ✓ Existing files: {existing_files}")
    print(f"    ✗ Missing files: {missing_files}")
    
    if missing_files > 0:
        print(f"\n  Missing files (showing first 10):")
        for filename, path in missing_list[:10]:
            print(f"    - {filename}: {path}")
        if len(missing_list) > 10:
            print(f"    ... and {len(missing_list) - 10} more")
    
    return total_entries, existing_files, missing_files, missing_list


def main():
    """Verify all VietSpeech splits."""
    base_dir = Path(__file__).parent.parent / "data" / "processed" / "VietSpeech"
    
    splits = ['test', 'train', 'val']
    total_stats = {
        'total_entries': 0,
        'existing_files': 0,
        'missing_files': 0,
        'all_missing': []
    }
    
    for split in splits:
        timestamps_json_path = base_dir / split / "timestamps.json"
        
        if not timestamps_json_path.exists():
            print(f"⚠ Skipping {split}: timestamps.json not found")
            continue
        
        total, existing, missing, missing_list = verify_audio_filepaths(
            timestamps_json_path,
            split
        )
        
        total_stats['total_entries'] += total
        total_stats['existing_files'] += existing
        total_stats['missing_files'] += missing
        total_stats['all_missing'].extend([(split, f, p) for f, p in missing_list])
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries checked: {total_stats['total_entries']}")
    print(f"✓ Existing files: {total_stats['existing_files']} ({100*total_stats['existing_files']/total_stats['total_entries']:.2f}%)")
    print(f"✗ Missing files: {total_stats['missing_files']} ({100*total_stats['missing_files']/total_stats['total_entries']:.2f}%)")
    
    if total_stats['missing_files'] > 0:
        print(f"\n⚠ WARNING: {total_stats['missing_files']} audio files are missing!")
        print(f"\nAll missing files:")
        for split, filename, path in total_stats['all_missing'][:50]:
            print(f"  [{split}] {filename}: {path}")
        if len(total_stats['all_missing']) > 50:
            print(f"  ... and {len(total_stats['all_missing']) - 50} more")
    else:
        print(f"\n✓ SUCCESS: All audio_filepath entries match existing files!")
    
    print(f"{'='*60}\n")
    
    return total_stats['missing_files'] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

