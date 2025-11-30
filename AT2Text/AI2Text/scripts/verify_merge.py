import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Set


def verify_merge(base_dir: Path) -> bool:
    """
    Verify that all CSV and JSON data from train/val/test splits
    are correctly merged into all_manifest.csv and all_timestamps.json.
    """
    splits = ["train", "val", "test"]
    all_ok = True
    
    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    
    # ===== CSV Verification =====
    print("\n[1] CSV FILE VERIFICATION")
    print("-" * 60)
    
    # Collect all rows from individual split CSVs
    split_csv_rows: Dict[str, Set[str]] = {split: set() for split in splits}
    split_csv_counts: Dict[str, int] = {split: 0 for split in splits}
    
    for split in splits:
        manifest_csv = base_dir / split / "manifest.csv"
        if not manifest_csv.is_file():
            print(f"⚠ WARNING: {manifest_csv} not found")
            continue
        
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a unique key from file_name and transcription
                key = f"{row.get('file_name', '')}|{row.get('transcription', '')}"
                split_csv_rows[split].add(key)
                split_csv_counts[split] += 1
        
        print(f"  {split:5s}: {split_csv_counts[split]:,} rows")
    
    total_expected = sum(split_csv_counts.values())
    print(f"  Total expected: {total_expected:,} rows")
    
    # Check merged CSV
    merged_csv = base_dir / "all_manifest.csv"
    if not merged_csv.is_file():
        print(f"❌ ERROR: {merged_csv} not found")
        all_ok = False
    else:
        merged_rows: Set[str] = set()
        merged_by_split: Dict[str, int] = {split: 0 for split in splits}
        merged_count = 0
        
        with merged_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('file_name', '')}|{row.get('transcription', '')}"
                merged_rows.add(key)
                split_name = row.get('split', '').strip()
                if split_name in merged_by_split:
                    merged_by_split[split_name] += 1
                merged_count += 1
        
        print(f"\n  Merged CSV: {merged_count:,} rows")
        print(f"  By split:")
        for split in splits:
            print(f"    {split:5s}: {merged_by_split[split]:,} rows")
        
        # Verify counts match
        if merged_count != total_expected:
            print(f"❌ ERROR: Row count mismatch! Expected {total_expected:,}, got {merged_count:,}")
            all_ok = False
        else:
            print(f"✓ Row count matches: {merged_count:,}")
        
        # Verify all rows from each split are present
        for split in splits:
            missing = split_csv_rows[split] - merged_rows
            if missing:
                print(f"❌ ERROR: {len(missing)} rows from {split} are missing in merged CSV")
                all_ok = False
                if len(missing) <= 5:
                    print(f"   Missing examples: {list(missing)[:3]}")
            else:
                print(f"✓ All {split_csv_counts[split]:,} rows from {split} are present")
        
        # Check for unexpected rows
        all_expected_keys = set()
        for split_rows in split_csv_rows.values():
            all_expected_keys.update(split_rows)
        
        unexpected = merged_rows - all_expected_keys
        if unexpected:
            print(f"⚠ WARNING: {len(unexpected)} unexpected rows in merged CSV")
            if len(unexpected) <= 5:
                print(f"   Examples: {list(unexpected)[:3]}")
    
    # ===== JSON Verification =====
    print("\n[2] JSON FILE VERIFICATION")
    print("-" * 60)
    
    # Collect all entries from individual split JSONs
    split_json_keys: Dict[str, Set[str]] = {split: set() for split in splits}
    split_json_counts: Dict[str, int] = {split: 0 for split in splits}
    
    for split in splits:
        timestamps_json = base_dir / split / "timestamps.json"
        if not timestamps_json.is_file():
            print(f"⚠ WARNING: {timestamps_json} not found")
            continue
        
        with timestamps_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for key in data.keys():
                split_json_keys[split].add(key)
                split_json_counts[split] += 1
        
        print(f"  {split:5s}: {split_json_counts[split]:,} entries")
    
    total_expected_json = sum(split_json_counts.values())
    print(f"  Total expected: {total_expected_json:,} entries")
    
    # Check merged JSON
    merged_json_file = base_dir / "all_timestamps.json"
    if not merged_json_file.is_file():
        print(f"❌ ERROR: {merged_json_file} not found")
        all_ok = False
    else:
        print(f"\n  Loading merged JSON (this may take a moment)...")
        with merged_json_file.open("r", encoding="utf-8") as f:
            merged_json_data = json.load(f)
        
        merged_json_keys = set(merged_json_data.keys())
        merged_json_count = len(merged_json_keys)
        
        print(f"  Merged JSON: {merged_json_count:,} entries")
        
        # Verify counts match
        if merged_json_count != total_expected_json:
            print(f"❌ ERROR: Entry count mismatch! Expected {total_expected_json:,}, got {merged_json_count:,}")
            all_ok = False
        else:
            print(f"✓ Entry count matches: {merged_json_count:,}")
        
        # Verify all entries from each split are present
        # Note: duplicates may have split prefix in merged JSON
        for split in splits:
            # Check for both original keys and split-prefixed keys
            found_count = 0
            missing_keys = []
            for key in split_json_keys[split]:
                # Check if key exists as-is or with split prefix
                if key in merged_json_keys:
                    found_count += 1
                elif f"{split}_{key}" in merged_json_keys:
                    found_count += 1
                else:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"❌ ERROR: {len(missing_keys)} entries from {split} are missing in merged JSON")
                all_ok = False
                if len(missing_keys) <= 5:
                    print(f"   Missing examples: {list(missing_keys)[:3]}")
            else:
                print(f"✓ All {split_json_counts[split]:,} entries from {split} are present")
        
        # Check for unexpected entries
        # Note: "unexpected" entries are actually duplicate keys that were made unique with split prefixes
        all_expected_json_keys = set()
        for split_keys in split_json_keys.values():
            all_expected_json_keys.update(split_keys)
        
        # Build set of expected keys including split-prefixed versions for duplicates
        all_expected_with_prefixes = set(all_expected_json_keys)
        for split in splits:
            for key in split_json_keys[split]:
                # Add split-prefixed version (for duplicates)
                all_expected_with_prefixes.add(f"{split}_{key}")
        
        unexpected = merged_json_keys - all_expected_with_prefixes
        if unexpected:
            print(f"⚠ WARNING: {len(unexpected)} truly unexpected entries in merged JSON")
            if len(unexpected) <= 5:
                print(f"   Examples: {list(unexpected)[:3]}")
        else:
            # Count how many are split-prefixed duplicates
            split_prefixed = {k for k in merged_json_keys if any(k.startswith(f"{s}_") for s in splits)}
            if split_prefixed:
                print(f"ℹ INFO: {len(split_prefixed):,} entries are split-prefixed duplicates (expected behavior)")
        
        # Verify split information is present in JSON entries
        print(f"\n  Checking split information in JSON entries...")
        entries_with_split = 0
        entries_without_split = 0
        for key, value in list(merged_json_data.items())[:1000]:  # Sample check
            if isinstance(value, dict) and "split" in value:
                entries_with_split += 1
            else:
                entries_without_split += 1
        
        if entries_without_split > 0:
            print(f"⚠ WARNING: Some entries may be missing 'split' field (sampled {entries_without_split} out of 1000)")
        else:
            print(f"✓ Split information present in sampled entries")
    
    # ===== Cross-reference check =====
    print("\n[3] CROSS-REFERENCE CHECK")
    print("-" * 60)
    
    # Check if CSV file_names match JSON keys
    if merged_csv.is_file() and merged_json_file.is_file():
        csv_file_names = set()
        csv_by_split: Dict[str, Set[str]] = {split: set() for split in splits}
        with merged_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get('file_name', '')
                split_name = row.get('split', '').strip()
                csv_file_names.add(file_name)
                if split_name in csv_by_split:
                    csv_by_split[split_name].add(file_name)
        
        # Extract original file names from JSON (remove split prefixes)
        json_original_keys = set()
        for key in merged_json_data.keys():
            # Remove split prefix if present (e.g., "train_file.wav" -> "file.wav")
            original_key = key
            for split in splits:
                if key.startswith(f"{split}_"):
                    original_key = key[len(split) + 1:]
                    break
            json_original_keys.add(original_key)
        
        csv_only = csv_file_names - json_original_keys
        json_only = json_original_keys - csv_file_names
        
        if csv_only:
            print(f"⚠ WARNING: {len(csv_only)} file names in CSV but not in JSON")
            if len(csv_only) <= 5:
                print(f"   Examples: {list(csv_only)[:3]}")
        
        if json_only:
            print(f"⚠ WARNING: {len(json_only)} keys in JSON but not in CSV")
            if len(json_only) <= 5:
                print(f"   Examples: {list(json_only)[:3]}")
        
        if not csv_only and not json_only:
            print(f"✓ CSV file names and JSON keys match perfectly")
        else:
            overlap = len(csv_file_names & json_original_keys)
            print(f"  Overlap: {overlap:,} entries present in both")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ VERIFICATION PASSED: All data correctly merged!")
    else:
        print("❌ VERIFICATION FAILED: Issues found (see above)")
    print("=" * 60)
    
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that all CSV and JSON files are correctly merged."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/raw/VietSpeech/processed_dataset_structured",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    verify_merge(base_dir)


if __name__ == "__main__":
    main()

