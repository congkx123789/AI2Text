import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def remove_phonemes_from_timestamps(timestamps_path: Path) -> tuple[int, bool]:
    """
    Remove phonemes from timestamps.json file.
    Returns (entries_processed, was_modified).
    """
    with timestamps_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries_processed = 0
    was_modified = False
    
    if isinstance(data, list):
        # List format
        for entry in data:
            if "phonemes" in entry:
                del entry["phonemes"]
                was_modified = True
            entries_processed += 1
    else:
        # Dict format
        for entry_id, entry in data.items():
            if isinstance(entry, dict) and "phonemes" in entry:
                del entry["phonemes"]
                was_modified = True
            entries_processed += 1
    
    if was_modified:
        with timestamps_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    return entries_processed, was_modified


def remove_phonemes_from_manifest(manifest_path: Path) -> tuple[int, bool]:
    """
    Remove phonemes_json column from manifest.csv file.
    Returns (rows_processed, was_modified).
    """
    rows = []
    fieldnames = None
    rows_processed = 0
    was_modified = False
    
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if "phonemes_json" in fieldnames:
            was_modified = True
            # Remove phonemes_json from fieldnames
            fieldnames = [f for f in fieldnames if f != "phonemes_json"]
        
        for row in reader:
            if "phonemes_json" in row:
                del row["phonemes_json"]
            rows.append(row)
            rows_processed += 1
    
    if was_modified:
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return rows_processed, was_modified


def remove_phonemes_from_librispeech(base_dir: Path, dry_run: bool = False) -> bool:
    """
    Remove all phonemes data from librispeech_alignments dataset.
    Removes 'phonemes' from timestamps.json and 'phonemes_json' from manifest.csv.
    """
    print("=" * 70)
    print("REMOVING PHONEMES FROM LIBRISPEECH ALIGNMENTS")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified")
    else:
        print("\n⚠ LIVE MODE - Files will be modified")
    
    splits = ["train", "val", "test"]
    
    # ===== Step 1: Remove phonemes from timestamps.json =====
    print("\n[1] REMOVING PHONEMES FROM TIMESTAMPS.JSON")
    print("-" * 70)
    
    total_entries = 0
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        if dry_run:
            # Just count entries
            with timestamps_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else len(data)
            print(f"  Would process {count:,} entries")
            total_entries += count
        else:
            entries_processed, was_modified = remove_phonemes_from_timestamps(timestamps_json)
            total_entries += entries_processed
            if was_modified:
                print(f"  ✓ Removed phonemes from {entries_processed:,} entries")
            else:
                print(f"  ℹ No phonemes found in {entries_processed:,} entries")
    
    # ===== Step 2: Remove phonemes_json from manifest.csv =====
    print("\n[2] REMOVING PHONEMES_JSON FROM MANIFEST.CSV")
    print("-" * 70)
    
    total_rows = 0
    for split in splits:
        print(f"\n{split.upper()} split:")
        split_dir = base_dir / split
        manifest_csv = split_dir / "manifest.csv"
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found")
            continue
        
        if dry_run:
            # Just check if column exists
            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                has_phonemes = "phonemes_json" in fieldnames
                row_count = sum(1 for _ in reader)
            
            total_rows += row_count
            if has_phonemes:
                print(f"  Would remove 'phonemes_json' column from {row_count:,} rows")
            else:
                print(f"  ℹ 'phonemes_json' column not found in {row_count:,} rows")
        else:
            rows_processed, was_modified = remove_phonemes_from_manifest(manifest_csv)
            total_rows += rows_processed
            if was_modified:
                print(f"  ✓ Removed 'phonemes_json' column from {rows_processed:,} rows")
            else:
                print(f"  ℹ 'phonemes_json' column not found in {rows_processed:,} rows")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal entries processed in timestamps.json: {total_entries:,}")
    print(f"Total rows processed in manifest.csv: {total_rows:,}")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Phoneme removal complete!")
        print("  All 'phonemes' data removed from timestamps.json")
        print("  All 'phonemes_json' columns removed from manifest.csv")
    
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove all phonemes data from librispeech_alignments dataset."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/processed/librispeech_alignments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually modifying files",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    remove_phonemes_from_librispeech(
        base_dir, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()




