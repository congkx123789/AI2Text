#!/usr/bin/env python3
"""
Create merged dataset from VietSpeech and LibriSpeech with specified durations.
Samples audio files randomly and creates train/val/test splits in data/processed.
Optimized for multi-core CPUs (Ryzen 9 9900X).
"""

import json
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import pandas as pd
# Removed parallel processing imports - using optimized single-threaded approach


def load_timestamps(json_path: Path) -> Dict:
    """Load timestamps.json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_manifest(csv_path: Path) -> pd.DataFrame:
    """Load manifest.csv file."""
    return pd.read_csv(csv_path)


def build_manifest_index(manifest: pd.DataFrame) -> Dict[str, dict]:
    """Build fast lookup index from manifest."""
    index = {}
    for _, row in manifest.iterrows():
        audio_path = str(row.get('audio_path', ''))
        file_id = str(row.get('id', ''))
        
        # Extract filename from audio_path
        if '/' in audio_path:
            filename = audio_path.split('/')[-1]
        else:
            filename = audio_path
        
        # Index by filename
        if filename:
            index[filename] = row.to_dict()
        
        # Also index by base name (without extension)
        base_name = filename.replace('.wav', '')
        if base_name and base_name not in index:
            index[base_name] = row.to_dict()
        
        # Index by ID
        if file_id and file_id not in index:
            index[file_id] = row.to_dict()
    
    return index


def match_filename_to_manifest_fast(filename: str, manifest_index: Dict) -> dict:
    """Fast filename to manifest matching using pre-built index."""
    # Try exact filename match
    if filename in manifest_index:
        return manifest_index[filename]
    
    # Try without extension
    base_name = filename.replace('.wav', '')
    if base_name in manifest_index:
        return manifest_index[base_name]
    
    return {}


def sample_files_by_duration(
    timestamps: Dict,
    manifest: pd.DataFrame,
    target_hours: float,
    dataset_name: str,
    exclude_filenames: set = None,
    num_workers: int = None
) -> Tuple[List[Dict], float]:
    """
    Sample files until we reach target duration (in hours).
    Optimized for Ryzen 9 9900X with fast indexing.
    Returns list of file entries and actual duration achieved.
    """
    target_seconds = target_hours * 3600
    if exclude_filenames is None:
        exclude_filenames = set()
    
    print(f"  {dataset_name}: Processing {len(timestamps):,} files...")
    
    # Build fast lookup index (much faster than searching DataFrame each time)
    manifest_index = build_manifest_index(manifest)
    print(f"  {dataset_name}: Manifest index built with {len(manifest_index):,} entries")
    
    # Create list of files with their durations (optimized single-pass)
    file_list = []
    matched_count = 0
    
    for filename, entry in timestamps.items():
        # Skip if excluded
        if filename in exclude_filenames:
            continue
            
        duration = entry.get('duration', 0.0)
        if duration > 0:
            # Fast lookup using index
            manifest_row = match_filename_to_manifest_fast(filename, manifest_index)
            if manifest_row:
                file_list.append({
                    'filename': filename,
                    'duration': duration,
                    'entry': entry,
                    'manifest_row': manifest_row
                })
                matched_count += 1
    
    print(f"  {dataset_name}: Matched {matched_count:,} files with manifest entries")
    
    # Shuffle randomly
    random.shuffle(file_list)
    
    # Sample until we reach target duration
    selected_files = []
    total_duration = 0.0
    
    for file_info in file_list:
        if total_duration >= target_seconds:
            break
        
        selected_files.append(file_info)
        total_duration += file_info['duration']
    
    actual_hours = total_duration / 3600
    print(f"  {dataset_name}: Selected {len(selected_files):,} files, "
          f"{actual_hours:.2f} hours ({total_duration:.2f} seconds)")
    
    if actual_hours < target_hours * 0.95:  # Warn if significantly under target
        print(f"  ⚠ WARNING: Only achieved {actual_hours:.2f}h out of {target_hours:.2f}h target")
        print(f"     This might indicate insufficient data or matching issues")
    
    return selected_files, actual_hours


def copy_audio_file(
    source_path: Path,
    dest_audio_dir: Path
) -> Path:
    """Copy audio file to destination directory."""
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_audio_dir / source_path.name
    
    if not dest_file.exists():
        shutil.copy2(source_path, dest_file)
    
    # Return relative path for manifest
    return Path('audio') / source_path.name


def create_manifest_entry(
    file_info: Dict,
    audio_rel_path: Path,
    dataset_source: str
) -> Dict:
    """Create a manifest entry from file info."""
    manifest_row = file_info['manifest_row']
    
    # Extract words_json from timestamps entry
    segments = file_info['entry'].get('segments', [])
    words_json = json.dumps([
        {
            'word': seg.get('word', ''),
            'start': seg.get('start', 0.0),
            'end': seg.get('end', 0.0)
        }
        for seg in segments
    ])
    
    # Create manifest entry
    entry = {
        'id': manifest_row.get('id', file_info['filename'].replace('.wav', '')),
        'transcript': manifest_row.get('transcript', file_info['entry'].get('text', '')),
        'audio_path': str(audio_rel_path),
        'words_json': words_json,
        'sex': manifest_row.get('sex', 'U'),
        'subset': dataset_source
    }
    
    return entry


def create_timestamps_entry(file_info: Dict) -> Tuple[str, Dict]:
    """Create timestamps.json entry from file info."""
    filename = file_info['filename']
    entry = file_info['entry']
    
    # Return filename and entry (already in correct format)
    return filename, entry


def create_split(
    split_name: str,
    vietspeech_files: List[Dict],
    librispeech_files: List[Dict],
    vietspeech_base: Path,
    librispeech_base: Path,
    output_dir: Path
):
    """Create a split (train/val/test) with merged files."""
    print(f"\n{'='*70}")
    print(f"Creating {split_name.upper()} split")
    print(f"{'='*70}")
    
    split_dir = output_dir / split_name
    audio_dir = split_dir / 'audio'
    split_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge and shuffle files
    all_files = []
    
    # Add VietSpeech files
    vs_found = 0
    for file_info in vietspeech_files:
        source_audio = vietspeech_base / 'audio' / file_info['filename']
        if source_audio.exists():
            all_files.append(('vietspeech', file_info, source_audio))
            vs_found += 1
        else:
            # Try alternative path (might be in subdirectory)
            alt_path = vietspeech_base / file_info['filename']
            if alt_path.exists():
                all_files.append(('vietspeech', file_info, alt_path))
                vs_found += 1
    
    # Add LibriSpeech files
    ls_found = 0
    for file_info in librispeech_files:
        source_audio = librispeech_base / 'audio' / file_info['filename']
        if source_audio.exists():
            all_files.append(('librispeech', file_info, source_audio))
            ls_found += 1
        else:
            # Try alternative path
            alt_path = librispeech_base / file_info['filename']
            if alt_path.exists():
                all_files.append(('librispeech', file_info, alt_path))
                ls_found += 1
    
    print(f"  Found {vs_found}/{len(vietspeech_files)} VietSpeech files")
    print(f"  Found {ls_found}/{len(librispeech_files)} LibriSpeech files")
    
    # Randomly shuffle merged list
    random.shuffle(all_files)
    
    print(f"  Total files to process: {len(all_files)}")
    
    # Create manifest and timestamps
    manifest_entries = []
    timestamps_dict = {}
    
    for idx, (dataset_source, file_info, source_audio) in enumerate(all_files):
        if (idx + 1) % 1000 == 0:
            print(f"    Processing file {idx + 1}/{len(all_files)}...")
        
        # Copy audio file
        audio_rel_path = copy_audio_file(source_audio, audio_dir)
        
        # Create manifest entry
        manifest_entry = create_manifest_entry(file_info, audio_rel_path, dataset_source)
        manifest_entries.append(manifest_entry)
        
        # Create timestamps entry
        filename, timestamp_entry = create_timestamps_entry(file_info)
        timestamps_dict[filename] = timestamp_entry
    
    # Save manifest.csv
    manifest_df = pd.DataFrame(manifest_entries)
    manifest_csv = split_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"  ✅ Created manifest.csv with {len(manifest_entries)} entries")
    
    # Save timestamps.json
    timestamps_json = split_dir / 'timestamps.json'
    with open(timestamps_json, 'w', encoding='utf-8') as f:
        json.dump(timestamps_dict, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Created timestamps.json with {len(timestamps_dict)} entries")
    
    # Calculate total duration
    total_duration = sum(f['duration'] for _, f, _ in all_files)
    print(f"  ✅ Total duration: {total_duration/3600:.2f} hours ({total_duration:.2f} seconds)")
    
    return len(manifest_entries), total_duration


def main():
    parser = argparse.ArgumentParser(
        description="Create merged dataset from VietSpeech and LibriSpeech"
    )
    parser.add_argument(
        '--vietspeech-dir',
        type=str,
        default='data/raw/VietSpeech',
        help='Path to VietSpeech directory'
    )
    parser.add_argument(
        '--librispeech-dir',
        type=str,
        default='data/raw/librispeech_alignments',
        help='Path to LibriSpeech alignments directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/merged_dataset',
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--train-vs-hours',
        type=float,
        default=50.0,
        help='VietSpeech hours for train split'
    )
    parser.add_argument(
        '--train-ls-hours',
        type=float,
        default=25.0,
        help='LibriSpeech hours for train split'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect, max 16)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    vietspeech_dir = Path(args.vietspeech_dir)
    librispeech_dir = Path(args.librispeech_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("CREATING MERGED DATASET")
    print("="*70)
    print(f"VietSpeech source: {vietspeech_dir}")
    print(f"LibriSpeech source: {librispeech_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load train data for sampling
    print("Loading source data...")
    vs_train_timestamps = load_timestamps(vietspeech_dir / 'train' / 'timestamps.json')
    vs_train_manifest = load_manifest(vietspeech_dir / 'train' / 'manifest.csv')
    
    ls_train_timestamps = load_timestamps(librispeech_dir / 'train' / 'timestamps.json')
    ls_train_manifest = load_manifest(librispeech_dir / 'train' / 'manifest.csv')
    
    # Sample files for train split
    print(f"\nSampling files for TRAIN split:")
    print(f"  Target: {args.train_vs_hours}h VietSpeech + {args.train_ls_hours}h LibriSpeech = {args.train_vs_hours + args.train_ls_hours}h total")
    
    vs_train_files, vs_train_actual = sample_files_by_duration(
        vs_train_timestamps, vs_train_manifest,
        args.train_vs_hours, 'VietSpeech',
        num_workers=args.num_workers
    )
    
    ls_train_files, ls_train_actual = sample_files_by_duration(
        ls_train_timestamps, ls_train_manifest,
        args.train_ls_hours, 'LibriSpeech',
        num_workers=args.num_workers
    )
    
    train_total = vs_train_actual + ls_train_actual
    print(f"  Train total: {train_total:.2f} hours")
    
    # Calculate val/test sizes (1/8 of train)
    val_test_size = train_total / 8
    vs_val_test_hours = (vs_train_actual / train_total) * val_test_size if train_total > 0 else args.train_vs_hours / 8
    ls_val_test_hours = (ls_train_actual / train_total) * val_test_size if train_total > 0 else args.train_ls_hours / 8
    
    print(f"\nSampling files for VAL split:")
    print(f"  Target: {vs_val_test_hours:.2f}h VietSpeech + {ls_val_test_hours:.2f}h LibriSpeech = {val_test_size:.2f}h total")
    
    # Remove train files from available pool for val/test
    vs_train_filenames = {f['filename'] for f in vs_train_files}
    ls_train_filenames = {f['filename'] for f in ls_train_files}
    
    vs_val_files, vs_val_actual = sample_files_by_duration(
        vs_train_timestamps, vs_train_manifest,
        vs_val_test_hours, 'VietSpeech',
        exclude_filenames=vs_train_filenames,
        num_workers=args.num_workers
    )
    
    ls_val_files, ls_val_actual = sample_files_by_duration(
        ls_train_timestamps, ls_train_manifest,
        ls_val_test_hours, 'LibriSpeech',
        exclude_filenames=ls_train_filenames,
        num_workers=args.num_workers
    )
    
    val_total = vs_val_actual + ls_val_actual
    print(f"  Val total: {val_total:.2f} hours")
    
    # Sample test files (exclude train and val)
    print(f"\nSampling files for TEST split:")
    print(f"  Target: {vs_val_test_hours:.2f}h VietSpeech + {ls_val_test_hours:.2f}h LibriSpeech = {val_test_size:.2f}h total")
    
    vs_val_filenames = {f['filename'] for f in vs_val_files}
    ls_val_filenames = {f['filename'] for f in ls_val_files}
    
    vs_test_files, vs_test_actual = sample_files_by_duration(
        vs_train_timestamps, vs_train_manifest,
        vs_val_test_hours, 'VietSpeech',
        exclude_filenames=vs_train_filenames | vs_val_filenames,
        num_workers=args.num_workers
    )
    
    ls_test_files, ls_test_actual = sample_files_by_duration(
        ls_train_timestamps, ls_train_manifest,
        ls_val_test_hours, 'LibriSpeech',
        exclude_filenames=ls_train_filenames | ls_val_filenames,
        num_workers=args.num_workers
    )
    
    test_total = vs_test_actual + ls_test_actual
    print(f"  Test total: {test_total:.2f} hours")
    
    # Create splits
    print(f"\n{'='*70}")
    print("CREATING SPLITS")
    print(f"{'='*70}")
    
    # Create train split
    train_count, train_duration = create_split(
        'train',
        vs_train_files, ls_train_files,
        vietspeech_dir / 'train',
        librispeech_dir / 'train',
        output_dir
    )
    
    # Create val split
    val_count, val_duration = create_split(
        'val',
        vs_val_files, ls_val_files,
        vietspeech_dir / 'train',  # Still use train source for audio files
        librispeech_dir / 'train',
        output_dir
    )
    
    # Create test split
    test_count, test_duration = create_split(
        'test',
        vs_test_files, ls_test_files,
        vietspeech_dir / 'train',  # Still use train source for audio files
        librispeech_dir / 'train',
        output_dir
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Train: {train_count:,} files, {train_duration/3600:.2f} hours")
    print(f"Val:   {val_count:,} files, {val_duration/3600:.2f} hours")
    print(f"Test:  {test_count:,} files, {test_duration/3600:.2f} hours")
    print(f"Total: {train_count + val_count + test_count:,} files, {(train_duration + val_duration + test_duration)/3600:.2f} hours")
    print(f"\n✅ Dataset created successfully at: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

