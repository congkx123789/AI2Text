#!/usr/bin/env python3
"""
Verify that IDs in manifest.csv match with timestamps.json and audio files.
Checks consistency across all splits.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def verify_split(split_dir: Path, split_name: str, verbose: bool = False) -> dict:
    """Verify a single split (train/val/test)."""
    results = {
        'split': split_name,
        'manifest_count': 0,
        'timestamps_count': 0,
        'audio_count': 0,
        'id_matches': 0,
        'id_mismatches': [],
        'missing_audio': [],
        'invalid_audio_paths': [],
        'timestamp_mismatches': []
    }
    
    manifest_path = split_dir / 'manifest.csv'
    timestamps_path = split_dir / 'timestamps.json'
    audio_dir = split_dir / 'audio'
    
    # Check files exist
    if not manifest_path.exists():
        print(f"  ✗ manifest.csv not found")
        return results
    if not timestamps_path.exists():
        print(f"  ✗ timestamps.json not found")
        return results
    if not audio_dir.exists():
        print(f"  ✗ audio/ directory not found")
        return results
    
    # Load data
    try:
        manifest = pd.read_csv(manifest_path)
        results['manifest_count'] = len(manifest)
    except Exception as e:
        print(f"  ✗ Error loading manifest.csv: {e}")
        return results
    
    try:
        with open(timestamps_path, 'r', encoding='utf-8') as f:
            timestamps = json.load(f)
        results['timestamps_count'] = len(timestamps)
    except Exception as e:
        print(f"  ✗ Error loading timestamps.json: {e}")
        return results
    
    # Count audio files
    audio_files = list(audio_dir.glob('*.wav'))
    results['audio_count'] = len(audio_files)
    
    # Check ID matching between manifest and timestamps
    manifest_ids = set(manifest['id'].values)
    timestamp_keys = set(timestamps.keys())
    
    # Extract basenames from timestamp keys (remove .wav)
    timestamp_basenames = {k.replace('.wav', '') for k in timestamp_keys}
    
    # Find matches and mismatches
    id_matches = manifest_ids & timestamp_basenames
    results['id_matches'] = len(id_matches)
    
    # Find mismatches
    missing_in_timestamps = manifest_ids - timestamp_basenames
    missing_in_manifest = timestamp_basenames - manifest_ids
    
    if missing_in_timestamps:
        results['id_mismatches'].extend(list(missing_in_timestamps)[:10])  # Limit to 10
    if missing_in_manifest:
        results['timestamp_mismatches'].extend(list(missing_in_manifest)[:10])
    
    # Check audio files exist for each manifest entry
    for _, row in manifest.iterrows():
        audio_path_str = row['audio_path']
        
        # Validate audio_path format
        if not audio_path_str.startswith('audio/'):
            results['invalid_audio_paths'].append(row['id'])
            continue
        
        # Extract filename
        filename = audio_path_str.replace('audio/', '')
        audio_file = audio_dir / filename
        
        if not audio_file.exists():
            results['missing_audio'].append(row['id'])
    
    # Check that timestamps have corresponding audio files
    for timestamp_key in list(timestamp_keys)[:100]:  # Sample check
        filename = timestamp_key if timestamp_key.endswith('.wav') else f"{timestamp_key}.wav"
        audio_file = audio_dir / filename
        if not audio_file.exists() and len(results['missing_audio']) < 20:
            # Only add if not already in missing_audio
            base_id = timestamp_key.replace('.wav', '')
            if base_id not in results['missing_audio']:
                results['missing_audio'].append(base_id)
    
    return results


def print_results(results: dict, verbose: bool = False):
    """Print verification results."""
    split = results['split']
    print(f"\n{'='*70}")
    print(f"{split.upper()} SPLIT VERIFICATION")
    print(f"{'='*70}")
    
    print(f"\nFile Counts:")
    print(f"  manifest.csv entries: {results['manifest_count']:,}")
    print(f"  timestamps.json entries: {results['timestamps_count']:,}")
    print(f"  audio files: {results['audio_count']:,}")
    
    # Check consistency
    all_ok = True
    
    print(f"\nID Matching:")
    if results['manifest_count'] == results['timestamps_count'] == results['id_matches']:
        print(f"  ✅ All IDs match: {results['id_matches']:,}")
    else:
        print(f"  ⚠ ID matches: {results['id_matches']:,}")
        print(f"     Expected: {results['manifest_count']:,}")
        all_ok = False
    
    if results['id_mismatches']:
        print(f"  ⚠ IDs in manifest but not in timestamps ({len(results['id_mismatches'])}):")
        for mid in results['id_mismatches'][:5]:
            print(f"     - {mid}")
        if len(results['id_mismatches']) > 5:
            print(f"     ... and {len(results['id_mismatches']) - 5} more")
    
    if results['timestamp_mismatches']:
        print(f"  ⚠ IDs in timestamps but not in manifest ({len(results['timestamp_mismatches'])}):")
        for tid in results['timestamp_mismatches'][:5]:
            print(f"     - {tid}")
        if len(results['timestamp_mismatches']) > 5:
            print(f"     ... and {len(results['timestamp_mismatches']) - 5} more")
    
    print(f"\nAudio Files:")
    if results['missing_audio']:
        print(f"  ⚠ Missing audio files: {len(results['missing_audio'])}")
        if verbose:
            for aid in results['missing_audio'][:10]:
                print(f"     - {aid}")
        all_ok = False
    else:
        print(f"  ✅ All audio files exist: {results['audio_count']:,}")
    
    if results['invalid_audio_paths']:
        print(f"  ⚠ Invalid audio_path format: {len(results['invalid_audio_paths'])}")
        if verbose:
            for aid in results['invalid_audio_paths'][:10]:
                print(f"     - {aid}")
        all_ok = False
    
    # Final status
    if all_ok:
        print(f"\n✅ {split.upper()} SPLIT: All checks passed!")
    else:
        print(f"\n⚠️  {split.upper()} SPLIT: Issues found")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify merged dataset consistency"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/processed/merged_dataset',
        help='Path to merged dataset directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed error information'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print("="*70)
    print("VERIFYING MERGED DATASET")
    print("="*70)
    print(f"Dataset directory: {dataset_dir}")
    
    # Verify all splits
    splits = ['train', 'val', 'test']
    all_results = {}
    all_ok = True
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"\n⚠ {split} directory not found, skipping...")
            continue
        
        results = verify_split(split_dir, split, args.verbose)
        all_results[split] = results
        ok = print_results(results, args.verbose)
        if not ok:
            all_ok = False
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for split, results in all_results.items():
        status = "✅" if (
            results['id_matches'] == results['manifest_count'] and
            len(results['missing_audio']) == 0 and
            len(results['invalid_audio_paths']) == 0
        ) else "⚠️"
        print(f"{status} {split}: "
              f"{results['manifest_count']:,} entries, "
              f"{results['id_matches']:,} ID matches, "
              f"{len(results['missing_audio'])} missing audio")
    
    if all_ok:
        print(f"\n✅ All splits verified successfully!")
    else:
        print(f"\n⚠️  Some issues found. Use --verbose for details.")
    
    print("="*70)


if __name__ == '__main__':
    main()

