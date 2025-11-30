import argparse
import csv
from pathlib import Path
from typing import Dict, List


def remove_duplicates(base_dir: Path, dry_run: bool = False) -> bool:
    """
    Remove duplicate file_name entries from CSV files, keeping only the first occurrence.
    This ensures CSV row count matches audio file count and JSON entry count.
    """
    print("=" * 70)
    print("REMOVING CSV DUPLICATES")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified")
    else:
        print("\n⚠ LIVE MODE - CSV files will be updated")
    
    splits = ["train", "val", "test"]
    total_removed = 0
    
    for split in splits:
        print(f"\n[{split.upper()}] Processing {split} split...")
        print("-" * 70)
        
        split_dir = base_dir / split
        manifest_csv = split_dir / "manifest.csv"
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found")
            continue
        
        # Read all rows and track duplicates
        rows = []
        seen_files = set()
        duplicates = []
        
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                file_name = row.get("file_name", "").strip()
                if file_name:
                    if file_name not in seen_files:
                        seen_files.add(file_name)
                        rows.append(row)
                    else:
                        duplicates.append(file_name)
        
        removed_count = len(duplicates)
        total_removed += removed_count
        
        print(f"  Total rows: {len(rows) + removed_count:,}")
        print(f"  Unique files: {len(rows):,}")
        print(f"  Duplicates to remove: {removed_count:,}")
        
        if removed_count > 0:
            # Show some examples
            from collections import Counter
            dup_counter = Counter(duplicates)
            print(f"  Most duplicated files:")
            for file_name, count in dup_counter.most_common(5):
                print(f"    {file_name}: {count} times")
            
            if not dry_run:
                # Write cleaned CSV
                with manifest_csv.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  ✓ Updated {manifest_csv} ({len(rows):,} rows)")
            else:
                print(f"  [DRY RUN] Would update {manifest_csv} ({len(rows):,} rows)")
        else:
            print(f"  ✓ No duplicates found")
    
    # ===== Re-merge files =====
    if not dry_run and total_removed > 0:
        print("\n[RE-MERGING] Updating merged files...")
        print("-" * 70)
        
        import subprocess
        import sys
        
        merge_all_script = Path(__file__).parent / "merge_all_splits.py"
        if merge_all_script.exists():
            print("  Re-merging all splits...")
            subprocess.run([
                sys.executable, str(merge_all_script),
                "--base-dir", str(base_dir)
            ], check=False)
            print("  ✓ Merged files updated")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal duplicate rows removed: {total_removed:,}")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Duplicates removed!")
        print("  CSV row counts should now match audio file counts")
        print("  Run verify_audio_csv_json_counts.py to verify")
    
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate file_name entries from CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/raw/VietSpeech/processed_dataset_structured",
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
    
    remove_duplicates(base_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

