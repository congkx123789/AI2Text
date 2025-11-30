#!/usr/bin/env python3
"""
Reorder words_json in librispeech_alignments manifests to have "start" before "end".
This matches the format used in VietSpeech manifests.
"""

import pandas as pd
import json
import shutil
from pathlib import Path
import argparse


def reorder_words_json(words_json_str):
    """Reorder JSON objects to have 'start' before 'end'."""
    if pd.isna(words_json_str) or not words_json_str:
        return words_json_str
    
    try:
        words = json.loads(words_json_str)
        # Reorder each word object to have start, end, word order
        reordered = []
        for word_obj in words:
            reordered.append({
                'start': word_obj.get('start'),
                'end': word_obj.get('end'),
                'word': word_obj.get('word')
            })
        return json.dumps(reordered, separators=(',', ':'))
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  Warning: Could not parse words_json: {e}")
        return words_json_str


def process_manifest(manifest_path: Path):
    """Process a manifest CSV to reorder words_json."""
    print(f"Processing {manifest_path}...")
    
    # Create backup if it doesn't exist
    backup_path = manifest_path.with_suffix('.csv.backup')
    if not backup_path.exists():
        print(f"  Creating backup: {backup_path}")
        shutil.copy2(manifest_path, backup_path)
    else:
        print(f"  Backup already exists: {backup_path}")
    
    # Read the manifest
    print(f"  Reading manifest...")
    df = pd.read_csv(manifest_path)
    print(f"  Loaded {len(df)} rows")
    
    # Check if words_json column exists
    if 'words_json' not in df.columns:
        print(f"  ⚠️  No 'words_json' column found, skipping...")
        return
    
    # Process words_json column
    print(f"  Reordering words_json (start before end)...")
    df['words_json'] = df['words_json'].apply(reorder_words_json)
    
    # Save the updated manifest
    print(f"  Saving updated manifest...")
    df.to_csv(manifest_path, index=False)
    print(f"  ✅ Updated manifest saved: {manifest_path}")
    
    # Verify a sample
    if len(df) > 0:
        sample_json = df.iloc[0]['words_json']
        try:
            sample_words = json.loads(sample_json)
            if sample_words:
                first_word = sample_words[0]
                keys = list(first_word.keys())
                print(f"  Sample word object keys: {keys}")
                if keys[0] == 'start' and keys[1] == 'end':
                    print(f"  ✅ Verification: start comes before end")
                else:
                    print(f"  ⚠️  Warning: Unexpected key order: {keys}")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Reorder words_json in librispeech_alignments manifests to have start before end"
    )
    parser.add_argument(
        '--librispeech-dir',
        type=str,
        default='data/processed/librispeech_alignments',
        help='Path to librispeech_alignments processed directory'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to process (default: train val test)'
    )
    
    args = parser.parse_args()
    
    librispeech_dir = Path(args.librispeech_dir)
    if not librispeech_dir.exists():
        raise SystemExit(f"Error: librispeech_alignments directory not found: {librispeech_dir}")
    
    print(f"Reordering words_json in librispeech_alignments manifests")
    print(f"Directory: {librispeech_dir}")
    print(f"Processing splits: {args.splits}\n")
    
    # Process each split
    for split in args.splits:
        split_dir = librispeech_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}, skipping...")
            continue
        
        manifest_path = split_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"⚠️  Manifest not found: {manifest_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        process_manifest(manifest_path)
    
    print(f"\n✅ Done! All manifests updated to have 'start' before 'end' in words_json.")


if __name__ == "__main__":
    main()

