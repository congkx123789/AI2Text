import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict


def verify_audio_files(base_dir: Path) -> bool:
    """
    Verify that all file names in CSV and JSON have corresponding audio files
    in the audio directories.
    """
    splits = ["train", "val", "test"]
    all_ok = True
    
    print("=" * 60)
    print("AUDIO FILE VERIFICATION")
    print("=" * 60)
    
    # ===== Check individual splits =====
    print("\n[1] CHECKING INDIVIDUAL SPLITS")
    print("-" * 60)
    
    split_results = {}
    
    for split in splits:
        split_dir = base_dir / split
        audio_dir = split_dir / "audio"
        manifest_csv = split_dir / "manifest.csv"
        timestamps_json = split_dir / "timestamps.json"
        
        print(f"\n{split.upper()} split:")
        
        # Get all audio files in the directory
        audio_files: Set[str] = set()
        if audio_dir.is_dir():
            audio_files = {f.name for f in audio_dir.glob("*.wav")}
            print(f"  Audio files found: {len(audio_files):,}")
        else:
            print(f"  ⚠ WARNING: {audio_dir} not found")
        
        # Check CSV files
        csv_file_names: Set[str] = set()
        csv_missing: List[str] = []
        if manifest_csv.is_file():
            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = row.get("file_name", "").strip()
                    if file_name:
                        csv_file_names.add(file_name)
                        if file_name not in audio_files:
                            csv_missing.append(file_name)
            
            print(f"  CSV file names: {len(csv_file_names):,}")
            if csv_missing:
                print(f"  ❌ ERROR: {len(csv_missing)} CSV entries missing audio files")
                if len(csv_missing) <= 10:
                    print(f"     Examples: {csv_missing[:5]}")
                all_ok = False
            else:
                print(f"  ✓ All CSV entries have audio files")
        else:
            print(f"  ⚠ WARNING: {manifest_csv} not found")
        
        # Check JSON files
        json_keys: Set[str] = set()
        json_missing: List[str] = []
        if timestamps_json.is_file():
            with timestamps_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for key in data.keys():
                    json_keys.add(key)
                    if key not in audio_files:
                        json_missing.append(key)
            
            print(f"  JSON keys: {len(json_keys):,}")
            if json_missing:
                print(f"  ❌ ERROR: {len(json_missing)} JSON entries missing audio files")
                if len(json_missing) <= 10:
                    print(f"     Examples: {json_missing[:5]}")
                all_ok = False
            else:
                print(f"  ✓ All JSON entries have audio files")
        else:
            print(f"  ⚠ WARNING: {timestamps_json} not found")
        
        # Check for audio files not in CSV/JSON
        orphan_audio = audio_files - csv_file_names - json_keys
        if orphan_audio:
            print(f"  ⚠ WARNING: {len(orphan_audio)} audio files not in CSV or JSON")
            if len(orphan_audio) <= 10:
                print(f"     Examples: {list(orphan_audio)[:5]}")
        
        # Check CSV/JSON consistency
        csv_only = csv_file_names - json_keys
        json_only = json_keys - csv_file_names
        if csv_only:
            print(f"  ⚠ WARNING: {len(csv_only)} file names in CSV but not in JSON")
        if json_only:
            print(f"  ⚠ WARNING: {len(json_only)} keys in JSON but not in CSV")
        
        split_results[split] = {
            "audio_count": len(audio_files),
            "csv_count": len(csv_file_names),
            "json_count": len(json_keys),
            "csv_missing": len(csv_missing),
            "json_missing": len(json_missing),
        }
    
    # ===== Check merged files =====
    print("\n[2] CHECKING MERGED FILES")
    print("-" * 60)
    
    merged_csv = base_dir / "all_manifest.csv"
    merged_json = base_dir / "all_timestamps.json"
    
    if merged_csv.is_file():
        print("\nMerged CSV:")
        merged_csv_files: Set[str] = set()
        merged_csv_by_split: Dict[str, Set[str]] = defaultdict(set)
        
        with merged_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name", "").strip()
                split_name = row.get("split", "").strip()
                if file_name:
                    merged_csv_files.add(file_name)
                    merged_csv_by_split[split_name].add(file_name)
        
        print(f"  Total unique file names: {len(merged_csv_files):,}")
        for split in splits:
            print(f"    {split}: {len(merged_csv_by_split[split]):,} files")
        
        # Check if all merged CSV files have audio in their respective splits
        merged_csv_missing = []
        for split in splits:
            audio_dir = base_dir / split / "audio"
            if audio_dir.is_dir():
                audio_files = {f.name for f in audio_dir.glob("*.wav")}
                for file_name in merged_csv_by_split[split]:
                    if file_name not in audio_files:
                        merged_csv_missing.append((split, file_name))
        
        if merged_csv_missing:
            print(f"  ❌ ERROR: {len(merged_csv_missing)} merged CSV entries missing audio files")
            if len(merged_csv_missing) <= 10:
                print(f"     Examples: {merged_csv_missing[:5]}")
            all_ok = False
        else:
            print(f"  ✓ All merged CSV entries have audio files in their splits")
    
    if merged_json.is_file():
        print("\nMerged JSON:")
        print("  Loading merged JSON (this may take a moment)...")
        with merged_json.open("r", encoding="utf-8") as f:
            merged_json_data = json.load(f)
        
        merged_json_keys = set(merged_json_data.keys())
        print(f"  Total keys: {len(merged_json_keys):,}")
        
        # Extract original file names (remove split prefixes)
        merged_json_original: Set[str] = set()
        merged_json_by_split: Dict[str, Set[str]] = defaultdict(set)
        
        for key in merged_json_keys:
            entry = merged_json_data[key]
            if isinstance(entry, dict):
                # Get original file name
                original_name = entry.get("original_file_name", key)
                # Remove split prefix if present
                for split in splits:
                    if original_name.startswith(f"{split}_"):
                        original_name = original_name[len(split) + 1:]
                        break
                
                split_name = entry.get("split", "")
                merged_json_original.add(original_name)
                merged_json_by_split[split_name].add(original_name)
        
        print(f"  Total unique original file names: {len(merged_json_original):,}")
        for split in splits:
            print(f"    {split}: {len(merged_json_by_split[split]):,} files")
        
        # Check if all merged JSON files have audio in their respective splits
        merged_json_missing = []
        for split in splits:
            audio_dir = base_dir / split / "audio"
            if audio_dir.is_dir():
                audio_files = {f.name for f in audio_dir.glob("*.wav")}
                for file_name in merged_json_by_split[split]:
                    if file_name not in audio_files:
                        merged_json_missing.append((split, file_name))
        
        if merged_json_missing:
            print(f"  ❌ ERROR: {len(merged_json_missing)} merged JSON entries missing audio files")
            if len(merged_json_missing) <= 10:
                print(f"     Examples: {merged_json_missing[:5]}")
            all_ok = False
        else:
            print(f"  ✓ All merged JSON entries have audio files in their splits")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_audio = sum(r["audio_count"] for r in split_results.values())
    total_csv = sum(r["csv_count"] for r in split_results.values())
    total_json = sum(r["json_count"] for r in split_results.values())
    total_csv_missing = sum(r["csv_missing"] for r in split_results.values())
    total_json_missing = sum(r["json_missing"] for r in split_results.values())
    
    print(f"\nTotal across all splits:")
    print(f"  Audio files: {total_audio:,}")
    print(f"  CSV entries: {total_csv:,}")
    print(f"  JSON entries: {total_json:,}")
    print(f"  CSV missing audio: {total_csv_missing:,}")
    print(f"  JSON missing audio: {total_json_missing:,}")
    
    if all_ok and total_csv_missing == 0 and total_json_missing == 0:
        print("\n✓ VERIFICATION PASSED: All CSV and JSON entries have corresponding audio files!")
    else:
        print("\n❌ VERIFICATION FAILED: Some entries are missing audio files")
        all_ok = False
    
    print("=" * 60)
    
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that all CSV and JSON file names have corresponding audio files."
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
    
    verify_audio_files(base_dir)


if __name__ == "__main__":
    main()

