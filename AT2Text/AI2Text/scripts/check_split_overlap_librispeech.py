#!/usr/bin/env python3
"""
Check for file overlap between train/val/test splits in librispeech_alignments.
Ensures no data leakage between splits.
"""

import json
from pathlib import Path
from collections import defaultdict


def load_split_filenames(timestamps_json_path):
    """
    Load all filenames (keys) from a timestamps.json file.
    
    Args:
        timestamps_json_path: Path to timestamps.json file
        
    Returns:
        set: Set of filenames (keys) in the JSON
    """
    with open(timestamps_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return set(data.keys())


def check_overlaps(test_files, train_files, val_files):
    """
    Check for overlaps between splits.
    
    Args:
        test_files: Set of test filenames
        train_files: Set of train filenames
        val_files: Set of val filenames
        
    Returns:
        dict: Dictionary with overlap information
    """
    overlaps = {
        'test_train': test_files & train_files,
        'test_val': test_files & val_files,
        'train_val': train_files & val_files,
        'all_three': test_files & train_files & val_files
    }
    return overlaps


def main():
    """Check for overlaps in librispeech_alignments splits."""
    base_dir = Path(__file__).parent.parent / "data" / "processed" / "librispeech_alignments"
    
    print("="*70)
    print("Checking librispeech_alignments for split overlaps")
    print("="*70)
    
    # Load filenames from each split
    print("\nLoading split files...")
    
    test_path = base_dir / "test" / "timestamps.json"
    train_path = base_dir / "train" / "timestamps.json"
    val_path = base_dir / "val" / "timestamps.json"
    
    if not test_path.exists():
        print(f"✗ ERROR: {test_path} not found")
        return False
    
    if not train_path.exists():
        print(f"✗ ERROR: {train_path} not found")
        return False
    
    if not val_path.exists():
        print(f"✗ ERROR: {val_path} not found")
        return False
    
    print("  Loading test split...")
    test_files = load_split_filenames(test_path)
    print(f"    Test files: {len(test_files)}")
    
    print("  Loading train split...")
    train_files = load_split_filenames(train_path)
    print(f"    Train files: {len(train_files)}")
    
    print("  Loading val split...")
    val_files = load_split_filenames(val_path)
    print(f"    Val files: {len(val_files)}")
    
    # Check for overlaps
    print("\n" + "="*70)
    print("Checking for overlaps...")
    print("="*70)
    
    overlaps = check_overlaps(test_files, train_files, val_files)
    
    # Report results
    test_train_overlap = overlaps['test_train']
    test_val_overlap = overlaps['test_val']
    train_val_overlap = overlaps['train_val']
    all_three_overlap = overlaps['all_three']
    
    print(f"\nTest ∩ Train: {len(test_train_overlap)} files")
    if test_train_overlap:
        print("  ⚠ WARNING: Files found in both test and train!")
        print("  First 10 overlapping files:")
        for filename in list(test_train_overlap)[:10]:
            print(f"    - {filename}")
        if len(test_train_overlap) > 10:
            print(f"    ... and {len(test_train_overlap) - 10} more")
    
    print(f"\nTest ∩ Val: {len(test_val_overlap)} files")
    if test_val_overlap:
        print("  ⚠ WARNING: Files found in both test and val!")
        print("  First 10 overlapping files:")
        for filename in list(test_val_overlap)[:10]:
            print(f"    - {filename}")
        if len(test_val_overlap) > 10:
            print(f"    ... and {len(test_val_overlap) - 10} more")
    
    print(f"\nTrain ∩ Val: {len(train_val_overlap)} files")
    if train_val_overlap:
        print("  ⚠ WARNING: Files found in both train and val!")
        print("  First 10 overlapping files:")
        for filename in list(train_val_overlap)[:10]:
            print(f"    - {filename}")
        if len(train_val_overlap) > 10:
            print(f"    ... and {len(train_val_overlap) - 10} more")
    
    print(f"\nTest ∩ Train ∩ Val: {len(all_three_overlap)} files")
    if all_three_overlap:
        print("  ⚠ CRITICAL: Files found in all three splits!")
        print("  Overlapping files:")
        for filename in all_three_overlap:
            print(f"    - {filename}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_unique_files = len(test_files | train_files | val_files)
    total_files_with_duplicates = len(test_files) + len(train_files) + len(val_files)
    duplicate_count = total_files_with_duplicates - total_unique_files
    
    print(f"Total unique files across all splits: {total_unique_files}")
    print(f"Total files (with duplicates): {total_files_with_duplicates}")
    print(f"Duplicate files: {duplicate_count}")
    
    has_overlaps = (len(test_train_overlap) > 0 or 
                   len(test_val_overlap) > 0 or 
                   len(train_val_overlap) > 0)
    
    if not has_overlaps:
        print("\n✓ SUCCESS: No overlaps found between splits!")
        print("  All splits are properly separated.")
    else:
        print("\n✗ WARNING: Overlaps detected between splits!")
        print("  This indicates potential data leakage.")
    
    print("="*70 + "\n")
    
    return not has_overlaps


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

