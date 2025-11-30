#!/usr/bin/env python3
"""
Create a CSV manifest file with timestamps (words_json column) from timestamps.json,
similar to librispeech_alignments format.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def convert_segments_to_words_json(segments: List[Dict]) -> str:
    """Convert segments from timestamps.json to words_json format."""
    words = []
    for segment in segments:
        words.append({
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "word": segment.get("word", "")
        })
    return json.dumps(words, ensure_ascii=False)


def create_manifest_with_timestamps(
    base_dir: Path,
    output_suffix: str = "_with_timestamps",
    dry_run: bool = False
) -> bool:
    """
    Create CSV manifest files with timestamps from timestamps.json.
    
    Args:
        base_dir: Base directory containing train/ val/ test/ split folders
        output_suffix: Suffix to add to output filename (default: "_with_timestamps")
        dry_run: If True, only show what would be created without writing files
    """
    print("=" * 70)
    print("CREATING CSV MANIFEST WITH TIMESTAMPS")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be created")
    else:
        print("\n⚠ LIVE MODE - Files will be created")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        print("-" * 70)
        
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        manifest_csv = split_dir / "manifest.csv"
        output_csv = split_dir / f"manifest{output_suffix}.csv"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found, skipping")
            continue
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found, skipping")
            continue
        
        # Load timestamps.json
        print("  Loading timestamps.json...")
        with timestamps_json.open("r", encoding="utf-8") as f:
            timestamps_data = json.load(f)
        
        if not isinstance(timestamps_data, dict):
            print(f"  ⚠ ERROR: timestamps.json is not in dictionary format")
            continue
        
        print(f"  Found {len(timestamps_data):,} entries in timestamps.json")
        
        # Load manifest.csv
        print("  Loading manifest.csv...")
        file_to_transcript = {}
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name", "").strip()
                transcript = row.get("transcription", "").strip()
                if file_name:
                    file_to_transcript[file_name] = transcript
        
        print(f"  Found {len(file_to_transcript):,} entries in manifest.csv")
        
        # Create output CSV
        print("  Creating manifest with timestamps...")
        output_rows = []
        matched_count = 0
        missing_timestamps = 0
        missing_transcript = 0
        
        for file_name, timestamp_entry in timestamps_data.items():
            # Get transcript from manifest
            transcript = file_to_transcript.get(file_name, "")
            if not transcript:
                # Try to get from timestamps.json
                transcript = timestamp_entry.get("text", "")
                if not transcript:
                    missing_transcript += 1
                    continue
            
            # Get segments
            segments = timestamp_entry.get("segments", [])
            if not segments:
                missing_timestamps += 1
                continue
            
            # Convert segments to words_json format
            words_json = convert_segments_to_words_json(segments)
            
            # Create id from file_name (remove extension)
            entry_id = Path(file_name).stem
            
            # Get audio_path (relative to split directory)
            audio_filepath = timestamp_entry.get("audio_filepath", f"audio/{file_name}")
            # Make it relative if it's absolute
            if Path(audio_filepath).is_absolute():
                # Try to make it relative to split_dir
                try:
                    audio_path = Path(audio_filepath).relative_to(split_dir)
                except ValueError:
                    audio_path = f"audio/{file_name}"
            else:
                audio_path = audio_filepath
            
            # Create row
            row = {
                "id": entry_id,
                "transcript": transcript,
                "audio_path": str(audio_path),
                "words_json": words_json
            }
            output_rows.append(row)
            matched_count += 1
        
        print(f"  Matched entries: {matched_count:,}")
        if missing_timestamps > 0:
            print(f"  ⚠ Entries without timestamps: {missing_timestamps:,}")
        if missing_transcript > 0:
            print(f"  ⚠ Entries without transcript: {missing_transcript:,}")
        
        # Write output CSV
        if not dry_run and output_rows:
            print(f"  Writing to {output_csv}...")
            with output_csv.open("w", encoding="utf-8", newline="") as f:
                fieldnames = ["id", "transcript", "audio_path", "words_json"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)
            print(f"  ✓ Created {output_csv} with {len(output_rows):,} entries")
        elif dry_run and output_rows:
            # Show sample
            sample = output_rows[0]
            print(f"\n  Sample entry:")
            print(f"    ID: {sample['id']}")
            print(f"    Transcript: {sample['transcript'][:50]}...")
            print(f"    Audio path: {sample['audio_path']}")
            words_sample = json.loads(sample['words_json'])
            print(f"    Words count: {len(words_sample)}")
            if words_sample:
                print(f"    First word: {words_sample[0]}")
    
    print("\n" + "=" * 70)
    if dry_run:
        print("⚠ This was a DRY RUN - no files were actually created")
        print("  Run without --dry-run to create files")
    else:
        print("✓ Manifest creation complete!")
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CSV manifest files with timestamps from timestamps.json, "
                    "similar to librispeech_alignments format."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/raw/VietSpeech/processed_dataset_structured",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_with_timestamps",
        help="Suffix to add to output filename (default: '_with_timestamps')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating files",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    create_manifest_with_timestamps(
        base_dir,
        output_suffix=args.output_suffix,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()










