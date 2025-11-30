#!/usr/bin/env python3
"""
Split LibriSpeech alignments dataset into train/val/test splits.
Can work with either the original splits or a clean manifest.
"""

import pandas as pd
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def load_all_data(base_dir, use_clean_manifest=None):
    """
    Load all data from splits or clean manifest.
    
    Args:
        base_dir: Base directory containing splits
        use_clean_manifest: Path to clean manifest CSV (if None, loads from splits)
        
    Returns:
        DataFrame with all samples
    """
    if use_clean_manifest and Path(use_clean_manifest).exists():
        print(f"Loading from clean manifest: {use_clean_manifest}")
        df = pd.read_csv(use_clean_manifest)
        return df
    
    # Load from all splits
    print(f"Loading from splits in: {base_dir}")
    base_dir = Path(base_dir)
    splits = sorted([d for d in base_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("split_")])
    
    all_rows = []
    
    for split_dir in tqdm(splits, desc="Loading splits"):
        csv_path = split_dir / "train.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        # Add split name and absolute paths
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['split_name'] = split_dir.name
            row_dict['abs_audio_path'] = str(split_dir / row['audio_path'])
            all_rows.append(row_dict)
    
    return pd.DataFrame(all_rows)


def split_dataframe(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                   random_state=42, stratify_by=None):
    """
    Split dataframe into train/val/test.
    
    Args:
        df: Input dataframe
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        stratify_by: Column name to stratify by (e.g., 'sex', 'subset')
        
    Returns:
        (train_df, val_df, test_df): Three dataframes
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # First split: train vs (val + test)
    train_size = train_ratio
    val_test_size = val_ratio + test_ratio
    
    if stratify_by and stratify_by in df.columns:
        stratify = df[stratify_by]
    else:
        stratify = None
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Second split: val vs test
    val_size_in_temp = val_ratio / val_test_size
    
    if stratify_by and stratify_by in temp_df.columns:
        stratify = temp_df[stratify_by]
    else:
        stratify = None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_in_temp),
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df, val_df, test_df


def copy_files_to_split(df, output_base, split_name, copy_audio=True):
    """
    Copy files to train/val/test directories.
    
    Args:
        df: DataFrame with samples for this split
        output_base: Base output directory
        split_name: 'train', 'val', or 'test'
        copy_audio: Whether to copy audio files (if False, just creates manifest)
    """
    split_dir = output_base / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV manifest
    manifest_path = split_dir / "manifest.csv"
    
    # Prepare manifest (use relative paths)
    manifest_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        row_dict = row.to_dict()
        
        # Get audio path
        if 'abs_audio_path' in row_dict:
            abs_audio_path = Path(row_dict['abs_audio_path'])
        else:
            # Reconstruct from split_name and audio_path
            split_name_orig = row_dict.get('split_name', 'split_000')
            base_dir = Path(row_dict.get('base_dir', 'data/raw/librispeech_alignments'))
            abs_audio_path = base_dir / split_name_orig / row_dict['audio_path']
        
        if abs_audio_path.exists():
            # Copy audio file
            if copy_audio:
                audio_filename = abs_audio_path.name
                dest_audio = audio_dir / audio_filename
                if not dest_audio.exists():
                    shutil.copy2(abs_audio_path, dest_audio)
                row_dict['audio_path'] = f"audio/{audio_filename}"
            else:
                row_dict['audio_path'] = str(abs_audio_path)
        else:
            row_dict['audio_path'] = row_dict.get('audio_path', '')
        
        # Remove internal columns
        row_dict.pop('abs_audio_path', None)
        row_dict.pop('split_name', None)
        row_dict.pop('base_dir', None)
        
        manifest_data.append(row_dict)
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"  Saved {len(manifest_df)} samples to {manifest_path}")
    if copy_audio:
        print(f"  Copied {len([f for f in audio_dir.iterdir() if f.is_file()])} audio files to {audio_dir}")


def copy_timestamps(df, output_base, split_name, base_dir):
    """
    Copy and filter timestamps.json for this split.
    
    Args:
        df: DataFrame with samples for this split
        output_base: Base output directory
        split_name: 'train', 'val', or 'test'
        base_dir: Original base directory with splits
    """
    split_dir = output_base / split_name
    timestamps_data = []
    
    # Get unique split names in this dataframe
    if 'split_name' in df.columns:
        split_names = df['split_name'].unique()
    else:
        # Try to infer from paths
        split_names = set()
        for path in df.get('abs_audio_path', df.get('audio_path', [])):
            if isinstance(path, str):
                parts = Path(path).parts
                for part in parts:
                    if part.startswith('split_'):
                        split_names.add(part)
                        break
    
    # Load timestamps from original splits
    base_dir = Path(base_dir)
    all_timestamps = {}
    
    for split_name_orig in split_names:
        timestamps_path = base_dir / split_name_orig / "timestamps.json"
        if timestamps_path.exists():
            with open(timestamps_path, 'r', encoding='utf-8') as f:
                split_timestamps = json.load(f)
                for entry in split_timestamps:
                    all_timestamps[entry['id']] = entry
    
    # Filter timestamps for this split
    valid_ids = set(df['id'].values)
    for entry_id, entry in all_timestamps.items():
        if entry_id in valid_ids:
            timestamps_data.append(entry)
    
    # Save filtered timestamps
    if timestamps_data:
        timestamps_path = split_dir / "timestamps.json"
        with open(timestamps_path, 'w', encoding='utf-8') as f:
            json.dump(timestamps_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(timestamps_data)} timestamp entries to {timestamps_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Split LibriSpeech alignments dataset into train/val/test'
    )
    parser.add_argument(
        '--input', type=str, default='data/raw/librispeech_alignments',
        help='Input directory containing splits'
    )
    parser.add_argument(
        '--clean-manifest', type=str, default=None,
        help='Path to clean manifest CSV (if provided, uses this instead of loading from splits)'
    )
    parser.add_argument(
        '--output', type=str, default='data/processed/librispeech_alignments',
        help='Output directory for train/val/test splits'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--stratify-by', type=str, default=None,
        help='Column to stratify by (e.g., "sex", "subset")'
    )
    parser.add_argument(
        '--no-copy-audio', action='store_true',
        help='Do not copy audio files, only create manifests'
    )
    parser.add_argument(
        '--no-timestamps', action='store_true',
        help='Do not copy timestamps.json files'
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_all_data(args.input, args.clean_manifest)
    print(f"\nTotal samples loaded: {len(df):,}")
    
    if len(df) == 0:
        print("Error: No data loaded!")
        return
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(df):,}")
    if 'sex' in df.columns:
        print(f"  By sex: {df['sex'].value_counts().to_dict()}")
    if 'subset' in df.columns:
        print(f"  By subset: {df['subset'].value_counts().to_dict()}")
    
    # Split data
    print(f"\nSplitting data:")
    print(f"  Train: {args.train_ratio*100:.1f}%")
    print(f"  Val:   {args.val_ratio*100:.1f}%")
    print(f"  Test:  {args.test_ratio*100:.1f}%")
    if args.stratify_by:
        print(f"  Stratify by: {args.stratify_by}")
    
    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        stratify_by=args.stratify_by
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Create output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Copy files to splits
    print(f"\nSaving splits to: {output_base}")
    
    copy_files_to_split(train_df, output_base, 'train', copy_audio=not args.no_copy_audio)
    copy_files_to_split(val_df, output_base, 'val', copy_audio=not args.no_copy_audio)
    copy_files_to_split(test_df, output_base, 'test', copy_audio=not args.no_copy_audio)
    
    # Copy timestamps
    if not args.no_timestamps:
        print("\nCopying timestamps...")
        copy_timestamps(train_df, output_base, 'train', args.input)
        copy_timestamps(val_df, output_base, 'val', args.input)
        copy_timestamps(test_df, output_base, 'test', args.input)
    
    # Save split summary
    summary = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'random_state': args.random_state,
        'stratify_by': args.stratify_by,
        'copy_audio': not args.no_copy_audio
    }
    
    summary_path = output_base / 'split_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Split complete!")
    print(f"   Summary saved to: {summary_path}")
    print(f"\nOutput structure:")
    print(f"   {output_base}/")
    print(f"   ├── train/")
    print(f"   │   ├── manifest.csv")
    print(f"   │   ├── timestamps.json")
    print(f"   │   └── audio/")
    print(f"   ├── val/")
    print(f"   │   ├── manifest.csv")
    print(f"   │   ├── timestamps.json")
    print(f"   │   └── audio/")
    print(f"   ├── test/")
    print(f"   │   ├── manifest.csv")
    print(f"   │   ├── timestamps.json")
    print(f"   │   └── audio/")
    print(f"   └── split_summary.json")


if __name__ == '__main__':
    main()

