import argparse
import csv
import json
from pathlib import Path
from typing import Set, Dict, List


def calculate_duration(entry: dict) -> float:
    """Calculate duration from word timestamps."""
    words = entry.get("words", [])
    if not words:
        return 0.0
    return words[-1]["end"] - words[0]["start"]


def filter_librispeech_by_duration(base_dir: Path, min_duration: float = 5.0, max_duration: float = 13.0, dry_run: bool = False) -> bool:
    """
    Keep only audio files with duration between min_duration and max_duration seconds.
    Remove all other files from audio directories, CSV manifests, and JSON timestamps.
    Works with librispeech_alignments format where timestamps.json is a list.
    """
    print("=" * 70)
    print("FILTERING LIBRISPEECH AUDIO BY DURATION")
    print("=" * 70)
    
    print(f"\nDuration filter: {min_duration}s - {max_duration}s")
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be deleted")
    else:
        print("\n⚠ LIVE MODE - Files will be removed")
    
    splits = ["train", "val", "test"]
    total_removed = 0
    ids_to_keep: Dict[str, Set[str]] = {split: set() for split in splits}
    ids_to_remove: Dict[str, Set[str]] = {split: set() for split in splits}
    
    # ===== Step 1: Identify files to keep/remove based on duration =====
    print("\n[1] ANALYZING DURATIONS")
    print("-" * 70)
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        with timestamps_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            entries = {entry["id"]: entry for entry in data}
        else:
            entries = data
        
        kept_count = 0
        removed_count = 0
        
        for entry_id, entry in entries.items():
            duration = calculate_duration(entry)
            
            if min_duration <= duration <= max_duration:
                ids_to_keep[split].add(entry_id)
                kept_count += 1
            else:
                ids_to_remove[split].add(entry_id)
                removed_count += 1
        
        print(f"  Total files: {len(entries):,}")
        print(f"  Files to keep ({min_duration}s-{max_duration}s): {kept_count:,}")
        print(f"  Files to remove: {removed_count:,}")
        
        total_removed += removed_count
    
    # ===== Step 2: Remove files from CSV manifests =====
    print("\n[2] UPDATING CSV MANIFESTS")
    print("-" * 70)
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        manifest_csv = split_dir / "manifest.csv"
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found")
            continue
        
        rows_to_keep = []
        removed_rows = 0
        
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                # Try both 'id' and 'file_name' columns
                entry_id = row.get("id", "").strip() or row.get("file_name", "").strip()
                if entry_id in ids_to_keep[split]:
                    rows_to_keep.append(row)
                else:
                    removed_rows += 1
        
        print(f"  Rows to keep: {len(rows_to_keep):,}")
        print(f"  Rows to remove: {removed_rows:,}")
        
        if removed_rows > 0 and not dry_run:
            with manifest_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_keep)
            print(f"  ✓ Updated {manifest_csv}")
        
        # Also update prepared_manifest.csv if it exists
        prepared_manifest_csv = split_dir / "prepared_manifest.csv"
        if prepared_manifest_csv.is_file():
            rows_to_keep_prepared = []
            removed_rows_prepared = 0
            
            with prepared_manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames_prepared = reader.fieldnames
                
                for row in reader:
                    # Extract id from file_path
                    file_path = row.get("file_path", "")
                    # Try to extract id from path (e.g., .../audio/1225-129527-0027.wav)
                    entry_id = Path(file_path).stem if file_path else ""
                    if entry_id in ids_to_keep[split]:
                        rows_to_keep_prepared.append(row)
                    else:
                        removed_rows_prepared += 1
            
            print(f"  Prepared manifest - Rows to keep: {len(rows_to_keep_prepared):,}")
            print(f"  Prepared manifest - Rows to remove: {removed_rows_prepared:,}")
            
            if removed_rows_prepared > 0 and not dry_run:
                with prepared_manifest_csv.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_prepared)
                    writer.writeheader()
                    writer.writerows(rows_to_keep_prepared)
                print(f"  ✓ Updated {prepared_manifest_csv}")
    
    # ===== Step 3: Remove entries from JSON timestamps =====
    print("\n[3] UPDATING JSON TIMESTAMPS")
    print("-" * 70)
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        with timestamps_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            cleaned_data = [entry for entry in data if entry.get("id") in ids_to_keep[split]]
            removed_entries = len(data) - len(cleaned_data)
        else:
            cleaned_data = {k: v for k, v in data.items() if k in ids_to_keep[split]}
            removed_entries = len(data) - len(cleaned_data)
        
        print(f"  Entries to keep: {len(cleaned_data):,}")
        print(f"  Entries to remove: {removed_entries:,}")
        
        if removed_entries > 0 and not dry_run:
            with timestamps_json.open("w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Updated {timestamps_json}")
    
    # ===== Step 4: Remove audio files =====
    print("\n[4] REMOVING AUDIO FILES")
    print("-" * 70)
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        audio_dir = split_dir / "audio"
        
        if not audio_dir.is_dir():
            print(f"  ⚠ WARNING: {audio_dir} not found")
            continue
        
        removed_audio = 0
        for entry_id in ids_to_remove[split]:
            # Try common audio extensions
            for ext in [".wav", ".flac", ".mp3"]:
                audio_file = audio_dir / f"{entry_id}{ext}"
                if audio_file.exists():
                    removed_audio += 1
                    if not dry_run:
                        audio_file.unlink()
                    break
        
        print(f"  Audio files to remove: {removed_audio:,}")
        if not dry_run and removed_audio > 0:
            print(f"  ✓ Removed {removed_audio:,} audio files")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_kept = sum(len(ids_to_keep[split]) for split in splits)
    
    print(f"\nDuration filter: {min_duration}s - {max_duration}s")
    print(f"\nFiles kept: {total_kept:,}")
    print(f"Files removed: {total_removed:,}")
    
    for split in splits:
        print(f"  {split}: {len(ids_to_keep[split]):,} kept, {len(ids_to_remove[split]):,} removed")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Filtering complete!")
        print(f"  Only audio files with duration {min_duration}s-{max_duration}s remain")
    
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter LibriSpeech audio files by duration, keeping only files between min and max duration."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/processed/librispeech_alignments",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=13.0,
        help="Maximum duration in seconds (default: 13.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing files",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    filter_librispeech_by_duration(
        base_dir, 
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()




