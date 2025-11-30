#!/usr/bin/env python3
"""
Remove 'sex' and 'subset' columns from manifest.csv files in merged dataset.
"""

import pandas as pd
from pathlib import Path
import argparse


def remove_columns_from_split(split_dir: Path, split_name: str):
    """Remove sex and subset columns from a split's manifest.csv."""
    manifest_path = split_dir / 'manifest.csv'
    
    if not manifest_path.exists():
        print(f"  ⚠ manifest.csv not found in {split_dir}")
        return False
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    original_columns = list(df.columns)
        
        # Check which columns exist
    columns_to_remove = []
    if 'sex' in df.columns:
        columns_to_remove.append('sex')
    if 'subset' in df.columns:
        columns_to_remove.append('subset')
    
    if not columns_to_remove:
        print(f"  ✓ {split_name}: No 'sex' or 'subset' columns found (already removed)")
            return True
        
    # Remove columns
    df = df.drop(columns=columns_to_remove)
    
    # Save back
    df.to_csv(manifest_path, index=False)
    
    print(f"  ✅ {split_name}: Removed columns: {', '.join(columns_to_remove)}")
    print(f"     Remaining columns: {list(df.columns)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove 'sex' and 'subset' columns from merged dataset manifests"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/processed/merged_dataset',
        help='Path to merged dataset directory'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print("="*70)
    print("REMOVING 'sex' AND 'subset' COLUMNS")
    print("="*70)
    print(f"Dataset directory: {dataset_dir}\n")
    
    # Process all splits
    splits = ['train', 'val', 'test']
    success_count = 0
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  ⚠ {split} directory not found, skipping...")
            continue
        
        if remove_columns_from_split(split_dir, split):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Processed {success_count}/{len(splits)} splits")
    print(f"✅ All 'sex' and 'subset' columns removed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
