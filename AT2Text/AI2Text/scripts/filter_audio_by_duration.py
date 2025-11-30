import argparse
import csv
import json
from pathlib import Path
from typing import Set, Dict


def filter_audio_by_duration(base_dir: Path, min_duration: float = 5.0, max_duration: float = 13.0, dry_run: bool = False) -> bool:
    """
    Keep only audio files with duration between min_duration and max_duration seconds.
    Remove all other files from audio directories, CSV manifests, and JSON timestamps.
    """
    print("=" * 70)
    print("FILTERING AUDIO BY DURATION")
    print("=" * 70)
    
    print(f"\nDuration filter: {min_duration}s - {max_duration}s")
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be deleted")
    else:
        print("\n⚠ LIVE MODE - Files will be removed")
    
    splits = ["train", "val", "test"]
    total_removed = 0
    files_to_keep: Dict[str, Set[str]] = {split: set() for split in splits}
    files_to_remove: Dict[str, Set[str]] = {split: set() for split in splits}
    
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
        
        kept_count = 0
        removed_count = 0
        
        for file_name, entry in data.items():
            if isinstance(entry, dict):
                duration = entry.get("duration", 0.0)
                try:
                    duration = float(duration)
                except (ValueError, TypeError):
                    duration = 0.0
                
                if min_duration <= duration <= max_duration:
                    files_to_keep[split].add(file_name)
                    kept_count += 1
                else:
                    files_to_remove[split].add(file_name)
                    removed_count += 1
        
        print(f"  Total files: {len(data):,}")
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
                file_name = row.get("file_name", "").strip()
                if file_name in files_to_keep[split]:
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
        
        cleaned_data = {k: v for k, v in data.items() if k in files_to_keep[split]}
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
        for file_name in files_to_remove[split]:
            audio_file = audio_dir / file_name
            if audio_file.exists():
                removed_audio += 1
                if not dry_run:
                    audio_file.unlink()
        
        print(f"  Audio files to remove: {removed_audio:,}")
        if not dry_run and removed_audio > 0:
            print(f"  ✓ Removed {removed_audio:,} audio files")
    
    # ===== Step 5: Clean shard directories =====
    print("\n[5] CLEANING SHARD DIRECTORIES")
    print("-" * 70)
    
    shard_removed_total = 0
    
    for split in splits:
        split_dir = base_dir / split
        shards = sorted([d for d in split_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("train-") and "-of-" in d.name])
        
        if not shards:
            continue
        
        print(f"\n{split.upper()} shards:")
        shard_removed = 0
        
        for shard in shards:
            # Clean shard CSV
            shard_csv = shard / "metadata.csv"
            if shard_csv.is_file():
                rows = []
                with shard_csv.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    for row in reader:
                        file_name = row.get("file_name", "").strip()
                        if file_name in files_to_keep[split]:
                            rows.append(row)
                        else:
                            shard_removed += 1
                
                if shard_removed > 0 and not dry_run:
                    with shard_csv.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
            
            # Clean shard JSON
            shard_json = next(shard.glob("*.json"), None)
            if shard_json and shard_json.is_file():
                with shard_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                cleaned_data = {k: v for k, v in data.items() if k in files_to_keep[split]}
                removed = len(data) - len(cleaned_data)
                if removed > 0:
                    shard_removed += removed
                    if not dry_run:
                        with shard_json.open("w", encoding="utf-8") as f:
                            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
            # Clean shard audio
            shard_audio = shard / "audio"
            if shard_audio.is_dir():
                for file_name in files_to_remove[split]:
                    audio_file = shard_audio / file_name
                    if audio_file.exists() and not dry_run:
                        audio_file.unlink()
        
        if shard_removed > 0:
            print(f"  Removed {shard_removed:,} entries from {split} shards")
            shard_removed_total += shard_removed
    
    # ===== Step 6: Re-merge files =====
    if not dry_run:
        print("\n[6] RE-MERGING FILES")
        print("-" * 70)
        
        import subprocess
        import sys
        
        # Re-merge shards
        merge_shards_script = Path(__file__).parent / "merge_vietspeech_shards.py"
        if merge_shards_script.exists():
            print("  Re-merging shards...")
            subprocess.run([
                sys.executable, str(merge_shards_script),
                "--base-dir", str(base_dir)
            ], check=False)
        
        # Re-merge all splits
        merge_all_script = Path(__file__).parent / "merge_all_splits.py"
        if merge_all_script.exists():
            print("  Re-merging all splits...")
            subprocess.run([
                sys.executable, str(merge_all_script),
                "--base-dir", str(base_dir)
            ], check=False)
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_kept = sum(len(files_to_keep[split]) for split in splits)
    
    print(f"\nDuration filter: {min_duration}s - {max_duration}s")
    print(f"\nFiles kept: {total_kept:,}")
    print(f"Files removed: {total_removed:,}")
    
    for split in splits:
        print(f"  {split}: {len(files_to_keep[split]):,} kept, {len(files_to_remove[split]):,} removed")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Filtering complete!")
        print("  Only audio files with duration 5s-13s remain")
        print("  Run verify_audio_csv_json_counts.py to verify counts")
    
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter audio files by duration, keeping only files between min and max duration."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/raw/VietSpeech/processed_dataset_structured",
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
    
    filter_audio_by_duration(
        base_dir, 
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

