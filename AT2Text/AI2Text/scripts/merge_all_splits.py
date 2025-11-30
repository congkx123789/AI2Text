import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def merge_all_csv_and_json(base_dir: Path) -> None:
    """
    Merge all CSV and JSON files from train/, val/, and test/ splits
    into single unified files at the base directory level.
    
    Input:
        base_dir/
          train/manifest.csv, train/timestamps.json
          val/manifest.csv, val/timestamps.json
          test/manifest.csv, test/timestamps.json
    
    Output:
        base_dir/
          all_manifest.csv          # All CSV rows from all splits
          all_timestamps.json       # All JSON entries from all splits
    """
    splits = ["train", "val", "test"]
    
    # Collect all CSV rows
    all_csv_rows: List[Dict[str, str]] = []
    fieldnames = None
    
    # Collect all JSON entries
    # Handle duplicates by making keys unique with split prefix
    all_json: Dict[str, dict] = {}
    duplicate_count = 0
    
    for split in splits:
        split_dir = base_dir / split
        
        # Merge CSV
        manifest_csv = split_dir / "manifest.csv"
        if manifest_csv.is_file():
            print(f"Reading {manifest_csv}...")
            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    # Add split information
                    row = {k.strip(): v for k, v in row.items()}
                    row["split"] = split
                    all_csv_rows.append(row)
        
        # Merge JSON
        timestamps_json = split_dir / "timestamps.json"
        if timestamps_json.is_file():
            print(f"Reading {timestamps_json}...")
            with timestamps_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Add split info to each entry and handle duplicates
                for key, value in data.items():
                    # Make key unique by prefixing with split if it already exists
                    unique_key = key
                    if key in all_json:
                        unique_key = f"{split}_{key}"
                        duplicate_count += 1
                    
                    if isinstance(value, dict):
                        value["split"] = split
                        value["original_file_name"] = key  # Preserve original filename
                    all_json[unique_key] = value
    
    if duplicate_count > 0:
        print(f"⚠ Found {duplicate_count} duplicate keys across splits (made unique with split prefix)")
    
    # Write merged CSV
    if all_csv_rows and fieldnames:
        # Add 'split' to fieldnames if not already there
        if "split" not in fieldnames:
            fieldnames = list(fieldnames) + ["split"]
        
        output_csv = base_dir / "all_manifest.csv"
        print(f"Writing {output_csv} with {len(all_csv_rows)} rows...")
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"✓ Created {output_csv}")
    
    # Write merged JSON
    if all_json:
        output_json = base_dir / "all_timestamps.json"
        print(f"Writing {output_json} with {len(all_json)} entries...")
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        print(f"✓ Created {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all CSV and JSON files from train/val/test splits into unified files."
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
    
    merge_all_csv_and_json(base_dir)


if __name__ == "__main__":
    main()

