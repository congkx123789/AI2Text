#!/usr/bin/env python3
"""
Add <|en|> language token to LibriSpeech manifest.csv files.
Makes the format consistent with VietSpeech which uses <|vi|> tokens.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

def add_en_token_to_manifest(manifest_path: Path, backup: bool = True):
    """Add <|en|> token to transcript column if not already present."""
    print(f"Processing: {manifest_path}")
    
    # Read the manifest
    df = pd.read_csv(manifest_path)
    original_count = len(df)
    
    # Check if transcript column exists
    if 'transcript' not in df.columns:
        print(f"ERROR: 'transcript' column not found in {manifest_path}")
        return False
    
    # Backup original file if requested
    if backup:
        backup_path = manifest_path.with_suffix('.csv.backup')
        if not backup_path.exists():
            print(f"Creating backup: {backup_path}")
            df.to_csv(backup_path, index=False)
        else:
            print(f"Backup already exists: {backup_path}")
    
    # Count how many already have the token
    has_token = df['transcript'].str.startswith('<|en|>', na=False).sum()
    print(f"Found {has_token}/{original_count} transcripts already with <|en|> token")
    
    # Add <|en|> token to transcripts that don't have it
    mask = ~df['transcript'].str.startswith('<|en|>', na=False)
    df.loc[mask, 'transcript'] = '<|en|> ' + df.loc[mask, 'transcript'].astype(str)
    
    # Count changes
    updated_count = mask.sum()
    print(f"Updated {updated_count} transcripts with <|en|> token")
    
    # Save the updated manifest
    df.to_csv(manifest_path, index=False)
    print(f"Saved updated manifest: {manifest_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Add <|en|> language token to LibriSpeech manifest.csv files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/librispeech_alignments',
        help='Path to librispeech_alignments directory'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Which splits to process (default: train val test)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)
    
    backup = not args.no_backup
    
    # Process each split
    for split in args.splits:
        split_dir = data_dir / split
        manifest_path = split_dir / 'manifest.csv'
        
        if not manifest_path.exists():
            print(f"WARNING: Manifest not found: {manifest_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        success = add_en_token_to_manifest(manifest_path, backup=backup)
        
        if not success:
            print(f"ERROR: Failed to process {manifest_path}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All manifests processed successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

