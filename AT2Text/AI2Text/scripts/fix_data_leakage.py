import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Set, Dict


def fix_data_leakage(base_dir: Path, dry_run: bool = False) -> bool:
    """
    Remove test files from train and val splits to prevent data leakage.
    This ensures test data is completely separate from training data.
    """
    print("=" * 60)
    print("FIXING DATA LEAKAGE")
    print("=" * 60)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be deleted")
    else:
        print("\n⚠ LIVE MODE - Files will be removed from train and val")
    
    # ===== Step 1: Identify test files =====
    print("\n[1] IDENTIFYING TEST FILES")
    print("-" * 60)
    
    test_dir = base_dir / "test"
    test_files: Set[str] = set()
    
    # Get test files from CSV
    test_manifest = test_dir / "manifest.csv"
    if test_manifest.is_file():
        with test_manifest.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name", "").strip()
                if file_name:
                    test_files.add(file_name)
        print(f"  Test files from CSV: {len(test_files):,}")
    
    # Get test files from JSON
    test_timestamps = test_dir / "timestamps.json"
    if test_timestamps.is_file():
        with test_timestamps.open("r", encoding="utf-8") as f:
            data = json.load(f)
            test_files.update(data.keys())
        print(f"  Test files from JSON: {len(test_files):,}")
    
    print(f"  Total unique test files: {len(test_files):,}")
    
    # ===== Step 2: Remove test files from train =====
    print("\n[2] REMOVING TEST FILES FROM TRAIN")
    print("-" * 60)
    
    train_dir = base_dir / "train"
    train_removed = 0
    
    # Remove from train CSV
    train_manifest = train_dir / "manifest.csv"
    if train_manifest.is_file():
        train_rows = []
        with train_manifest.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                file_name = row.get("file_name", "").strip()
                if file_name not in test_files:
                    train_rows.append(row)
                else:
                    train_removed += 1
        
        if train_removed > 0:
            print(f"  Removing {train_removed:,} rows from train/manifest.csv")
            if not dry_run:
                with train_manifest.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(train_rows)
                print(f"  ✓ Updated train/manifest.csv ({len(train_rows):,} rows remaining)")
        else:
            print(f"  ✓ No test files found in train/manifest.csv")
    
    # Remove from train JSON
    train_timestamps = train_dir / "timestamps.json"
    if train_timestamps.is_file():
        with train_timestamps.open("r", encoding="utf-8") as f:
            train_json_data = json.load(f)
        
        train_json_removed = 0
        train_json_cleaned = {}
        for key, value in train_json_data.items():
            if key not in test_files:
                train_json_cleaned[key] = value
            else:
                train_json_removed += 1
        
        if train_json_removed > 0:
            print(f"  Removing {train_json_removed:,} entries from train/timestamps.json")
            if not dry_run:
                with train_timestamps.open("w", encoding="utf-8") as f:
                    json.dump(train_json_cleaned, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Updated train/timestamps.json ({len(train_json_cleaned):,} entries remaining)")
        else:
            print(f"  ✓ No test files found in train/timestamps.json")
    
    # Remove audio files from train
    train_audio_dir = train_dir / "audio"
    if train_audio_dir.is_dir():
        train_audio_removed = 0
        for file_name in test_files:
            audio_file = train_audio_dir / file_name
            if audio_file.exists():
                train_audio_removed += 1
                if not dry_run:
                    audio_file.unlink()
        
        if train_audio_removed > 0:
            print(f"  Removing {train_audio_removed:,} audio files from train/audio/")
            if not dry_run:
                print(f"  ✓ Removed {train_audio_removed:,} audio files from train/audio/")
        else:
            print(f"  ✓ No test audio files found in train/audio/")
    
    # ===== Step 3: Remove test files from val =====
    print("\n[3] REMOVING TEST FILES FROM VAL")
    print("-" * 60)
    
    val_dir = base_dir / "val"
    val_removed = 0
    
    # Remove from val CSV
    val_manifest = val_dir / "manifest.csv"
    if val_manifest.is_file():
        val_rows = []
        with val_manifest.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                file_name = row.get("file_name", "").strip()
                if file_name not in test_files:
                    val_rows.append(row)
                else:
                    val_removed += 1
        
        if val_removed > 0:
            print(f"  Removing {val_removed:,} rows from val/manifest.csv")
            if not dry_run:
                with val_manifest.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(val_rows)
                print(f"  ✓ Updated val/manifest.csv ({len(val_rows):,} rows remaining)")
        else:
            print(f"  ✓ No test files found in val/manifest.csv")
    
    # Remove from val JSON
    val_timestamps = val_dir / "timestamps.json"
    if val_timestamps.is_file():
        with val_timestamps.open("r", encoding="utf-8") as f:
            val_json_data = json.load(f)
        
        val_json_removed = 0
        val_json_cleaned = {}
        for key, value in val_json_data.items():
            if key not in test_files:
                val_json_cleaned[key] = value
            else:
                val_json_removed += 1
        
        if val_json_removed > 0:
            print(f"  Removing {val_json_removed:,} entries from val/timestamps.json")
            if not dry_run:
                with val_timestamps.open("w", encoding="utf-8") as f:
                    json.dump(val_json_cleaned, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Updated val/timestamps.json ({len(val_json_cleaned):,} entries remaining)")
        else:
            print(f"  ✓ No test files found in val/timestamps.json")
    
    # Remove audio files from val
    val_audio_dir = val_dir / "audio"
    if val_audio_dir.is_dir():
        val_audio_removed = 0
        for file_name in test_files:
            audio_file = val_audio_dir / file_name
            if audio_file.exists():
                val_audio_removed += 1
                if not dry_run:
                    audio_file.unlink()
        
        if val_audio_removed > 0:
            print(f"  Removing {val_audio_removed:,} audio files from val/audio/")
            if not dry_run:
                print(f"  ✓ Removed {val_audio_removed:,} audio files from val/audio/")
        else:
            print(f"  ✓ No test audio files found in val/audio/")
    
    # ===== Step 4: Clean shard directories =====
    print("\n[4] CLEANING SHARD DIRECTORIES")
    print("-" * 60)
    
    shard_removed_train = 0
    shard_removed_val = 0
    
    # Clean train shards
    train_dir = base_dir / "train"
    train_shards = sorted([d for d in train_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("train-") and "-of-" in d.name])
    
    for shard in train_shards:
        shard_removed_this_shard = 0
        shard_csv = shard / "metadata.csv"
        if shard_csv.is_file():
            rows = []
            with shard_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    file_name = row.get("file_name", "").strip()
                    if file_name not in test_files:
                        rows.append(row)
                    else:
                        shard_removed_this_shard += 1
            
            if shard_removed_this_shard > 0:
                shard_removed_train += shard_removed_this_shard
                if not dry_run:
                    with shard_csv.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
        
        # Clean shard JSON
        shard_json = next(shard.glob("*.json"), None)
        if shard_json and shard_json.is_file():
            with shard_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            cleaned_data = {k: v for k, v in data.items() if k not in test_files}
            removed_json = len(data) - len(cleaned_data)
            if removed_json > 0:
                shard_removed_train += removed_json
                if not dry_run:
                    with shard_json.open("w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        # Clean shard audio
        shard_audio = shard / "audio"
        if shard_audio.is_dir():
            for file_name in test_files:
                audio_file = shard_audio / file_name
                if audio_file.exists() and not dry_run:
                    audio_file.unlink()
    
    if shard_removed_train > 0:
        print(f"  Removed {shard_removed_train:,} test entries from train shards")
    
    # Clean val shards
    val_dir = base_dir / "val"
    val_shards = sorted([d for d in val_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("train-") and "-of-" in d.name])
    
    for shard in val_shards:
        shard_removed_this_shard = 0
        shard_csv = shard / "metadata.csv"
        if shard_csv.is_file():
            rows = []
            with shard_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    file_name = row.get("file_name", "").strip()
                    if file_name not in test_files:
                        rows.append(row)
                    else:
                        shard_removed_this_shard += 1
            
            if shard_removed_this_shard > 0:
                shard_removed_val += shard_removed_this_shard
                if not dry_run:
                    with shard_csv.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
        
        # Clean shard JSON
        shard_json = next(shard.glob("*.json"), None)
        if shard_json and shard_json.is_file():
            with shard_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            cleaned_data = {k: v for k, v in data.items() if k not in test_files}
            removed_json = len(data) - len(cleaned_data)
            if removed_json > 0:
                shard_removed_val += removed_json
                if not dry_run:
                    with shard_json.open("w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        # Clean shard audio
        shard_audio = shard / "audio"
        if shard_audio.is_dir():
            for file_name in test_files:
                audio_file = shard_audio / file_name
                if audio_file.exists() and not dry_run:
                    audio_file.unlink()
    
    if shard_removed_val > 0:
        print(f"  Removed {shard_removed_val:,} test entries from val shards")
    
    # ===== Step 5: Re-merge files =====
    if not dry_run:
        print("\n[5] RE-MERGING FILES")
        print("-" * 60)
        print("  Re-running merge scripts to update merged files...")
        
        # Re-merge split-level files
        import subprocess
        import sys
        
        # Re-merge shards (if needed)
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
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_removed = train_removed + val_removed
    print(f"\nTotal files removed from train: {train_removed:,}")
    print(f"Total files removed from val: {val_removed:,}")
    print(f"Total files removed: {total_removed:,}")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Data leakage fixed!")
        print("  Test files have been removed from train and val splits")
        print("  Run check_data_leakage.py to verify")
    
    print("=" * 60)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove test files from train and val splits to prevent data leakage."
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
    
    fix_data_leakage(base_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
