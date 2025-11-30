#!/usr/bin/env python3
"""
Merge ALL data from VietSpeech and LibriSpeech alignments into processed directory.
Creates the same structure as merged_dataset but with all available data.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import pandas as pd


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


def process_all_files(
    timestamps: Dict,
    manifest: pd.DataFrame,
    dataset_name: str
) -> List[Dict]:
    """Process all files from timestamps and match with manifest."""
    print(f"  {dataset_name}: Processing {len(timestamps):,} files...")
    
    # Build fast lookup index
    manifest_index = build_manifest_index(manifest)
    print(f"  {dataset_name}: Manifest index built with {len(manifest_index):,} entries")
    
    # Create list of files
    file_list = []
    matched_count = 0
    
    for filename, entry in timestamps.items():
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
    return file_list


def copy_audio_file(source_path: Path, dest_audio_dir: Path) -> Path:
    """Copy audio file to destination directory."""
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_audio_dir / source_path.name
    
    if not dest_file.exists():
        shutil.copy2(source_path, dest_file)
    
    # Return relative path for manifest
    return Path('audio') / source_path.name


def create_manifest_entry(file_info: Dict, audio_rel_path: Path, dataset_source: str) -> Dict:
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
    
    # Create manifest entry (without sex and subset)
    entry = {
        'id': manifest_row.get('id', file_info['filename'].replace('.wav', '')),
        'transcript': manifest_row.get('transcript', file_info['entry'].get('text', '')),
        'audio_path': str(audio_rel_path),
        'words_json': words_json
    }
    
    return entry


def create_timestamps_entry(file_info: Dict) -> Tuple[str, Dict]:
    """Create timestamps.json entry from file info."""
    filename = file_info['filename']
    entry = file_info['entry']
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
            # Try alternative path
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
    print(f"  Total files to process: {len(all_files)}")
    
    # Randomly shuffle merged list
    random.shuffle(all_files)
    
    # Create manifest and timestamps
    manifest_entries = []
    timestamps_dict = {}
    
    for idx, (dataset_source, file_info, source_audio) in enumerate(all_files):
        if (idx + 1) % 5000 == 0:
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
        description="Merge ALL data from VietSpeech and LibriSpeech"
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
        default='data/processed/full_merged_dataset',
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    vietspeech_dir = Path(args.vietspeech_dir)
    librispeech_dir = Path(args.librispeech_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("MERGING ALL DATA FROM BOTH DATASETS")
    print("="*70)
    print(f"VietSpeech source: {vietspeech_dir}")
    print(f"LibriSpeech source: {librispeech_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Process all splits
    splits = ['train', 'val', 'test']
    results = {}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        # Load data
        print("\nLoading source data...")
        vs_split_dir = vietspeech_dir / split
        ls_split_dir = librispeech_dir / split
        
        if not vs_split_dir.exists():
            print(f"  ⚠ VietSpeech {split} directory not found, skipping...")
            continue
        if not ls_split_dir.exists():
            print(f"  ⚠ LibriSpeech {split} directory not found, skipping...")
            continue
        
        vs_timestamps = load_timestamps(vs_split_dir / 'timestamps.json')
        vs_manifest = load_manifest(vs_split_dir / 'manifest.csv')
        
        ls_timestamps = load_timestamps(ls_split_dir / 'timestamps.json')
        ls_manifest = load_manifest(ls_split_dir / 'manifest.csv')
        
        # Process all files
        print(f"\nProcessing files:")
        vs_files = process_all_files(vs_timestamps, vs_manifest, 'VietSpeech')
        ls_files = process_all_files(ls_timestamps, ls_manifest, 'LibriSpeech')
        
        # Create split
        count, duration = create_split(
            split,
            vs_files, ls_files,
            vs_split_dir,
            ls_split_dir,
            output_dir
        )
        
        results[split] = {'count': count, 'duration': duration}
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_count = 0
    total_duration = 0.0
    
    for split, result in results.items():
        count = result['count']
        duration = result['duration']
        total_count += count
        total_duration += duration
        print(f"{split}: {count:,} files, {duration/3600:.2f} hours")
    
    print(f"\nTotal: {total_count:,} files, {total_duration/3600:.2f} hours")
    print(f"\n✅ Full merged dataset created successfully at: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

