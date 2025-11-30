#!/usr/bin/env python3
"""
Add audio_filepath field to VietSpeech timestamps.json files,
matching the structure of librispeech_alignments.
"""

import json
import csv
import os
from pathlib import Path


def add_audio_filepath_to_json(timestamps_json_path, manifest_csv_path):
    """
    Add audio_filepath field to each entry in timestamps.json
    by matching with manifest.csv.
    
    Args:
        timestamps_json_path: Path to timestamps.json file
        manifest_csv_path: Path to manifest.csv file
    """
    print(f"Processing {timestamps_json_path}...")
    
    # Load timestamps.json
    print("  Loading timestamps.json...")
    with open(timestamps_json_path, 'r', encoding='utf-8') as f:
        timestamps_data = json.load(f)
    
    # Load manifest.csv and create a mapping from filename to audio_path
    print("  Loading manifest.csv...")
    filename_to_audio_path = {}
    with open(manifest_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract filename from audio_path (e.g., "audio/189_000001959.wav" -> "189_000001959.wav")
            audio_path = row['audio_path']
            filename = os.path.basename(audio_path)
            filename_to_audio_path[filename] = audio_path
    
    # Add audio_filepath to each entry
    print("  Adding audio_filepath to entries...")
    updated_count = 0
    missing_count = 0
    
    for filename, entry_data in timestamps_data.items():
        if filename in filename_to_audio_path:
            entry_data['audio_filepath'] = filename_to_audio_path[filename]
            updated_count += 1
        else:
            print(f"    Warning: No matching entry in manifest.csv for {filename}")
            missing_count += 1
    
    # Save updated JSON
    print(f"  Saving updated timestamps.json (updated: {updated_count}, missing: {missing_count})...")
    with open(timestamps_json_path, 'w', encoding='utf-8') as f:
        json.dump(timestamps_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Completed: {updated_count} entries updated")
    if missing_count > 0:
        print(f"  ⚠ Warning: {missing_count} entries not found in manifest.csv")
    
    return updated_count, missing_count


def main():
    """Process all VietSpeech splits (test, train, val)."""
    base_dir = Path(__file__).parent.parent / "data" / "processed" / "VietSpeech"
    
    splits = ['test', 'train', 'val']
    total_updated = 0
    total_missing = 0
    
    for split in splits:
        timestamps_json_path = base_dir / split / "timestamps.json"
        manifest_csv_path = base_dir / split / "manifest.csv"
        
        if not timestamps_json_path.exists():
            print(f"⚠ Skipping {split}: timestamps.json not found")
            continue
        
        if not manifest_csv_path.exists():
            print(f"⚠ Skipping {split}: manifest.csv not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        updated, missing = add_audio_filepath_to_json(
            timestamps_json_path,
            manifest_csv_path
        )
        
        total_updated += updated
        total_missing += missing
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total entries updated: {total_updated}")
    print(f"  Total entries missing: {total_missing}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

