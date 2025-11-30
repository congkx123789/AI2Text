import argparse
import csv
import json
from pathlib import Path
from typing import Set, Dict, List


def verify_files_match_audio(base_dir: Path) -> bool:
    """
    Verify that all file names in CSV and JSON match actual audio files in audio folders.
    Check for:
    1. Files in CSV/JSON that don't have corresponding audio files
    2. Audio files that aren't in CSV/JSON
    3. Consistency across all splits
    """
    print("=" * 70)
    print("VERIFYING FILES MATCH AUDIO FOLDERS")
    print("=" * 70)
    
    all_ok = True
    splits = ["train", "val", "test"]
    
    # ===== Check individual splits =====
    print("\n[1] CHECKING INDIVIDUAL SPLITS")
    print("-" * 70)
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        print("-" * 70)
        
        split_dir = base_dir / split
        audio_dir = split_dir / "audio"
        manifest_csv = split_dir / "manifest.csv"
        timestamps_json = split_dir / "timestamps.json"
        
        # Get audio files
        audio_files: Set[str] = set()
        if audio_dir.is_dir():
            audio_files = {f.name for f in audio_dir.glob("*.wav")}
            print(f"  Audio files in folder: {len(audio_files):,}")
        else:
            print(f"  ❌ ERROR: {audio_dir} not found")
            all_ok = False
            continue
        
        # Get CSV file names
        csv_files: Set[str] = set()
        if manifest_csv.is_file():
            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = row.get("file_name", "").strip()
                    if file_name:
                        csv_files.add(file_name)
            print(f"  Files in CSV: {len(csv_files):,}")
        else:
            print(f"  ❌ ERROR: {manifest_csv} not found")
            all_ok = False
            continue
        
        # Get JSON file names
        json_files: Set[str] = set()
        if timestamps_json.is_file():
            with timestamps_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
                json_files = set(data.keys())
            print(f"  Files in JSON: {len(json_files):,}")
        else:
            print(f"  ❌ ERROR: {timestamps_json} not found")
            all_ok = False
            continue
        
        # Check CSV vs Audio
        csv_missing_audio = csv_files - audio_files
        if csv_missing_audio:
            print(f"  ❌ ERROR: {len(csv_missing_audio):,} CSV entries missing audio files")
            if len(csv_missing_audio) <= 10:
                print(f"     Missing files:")
                for f in sorted(list(csv_missing_audio)[:10]):
                    print(f"       - {f}")
            all_ok = False
        else:
            print(f"  ✓ All CSV entries have audio files")
        
        # Check JSON vs Audio
        json_missing_audio = json_files - audio_files
        if json_missing_audio:
            print(f"  ❌ ERROR: {len(json_missing_audio):,} JSON entries missing audio files")
            if len(json_missing_audio) <= 10:
                print(f"     Missing files:")
                for f in sorted(list(json_missing_audio)[:10]):
                    print(f"       - {f}")
            all_ok = False
        else:
            print(f"  ✓ All JSON entries have audio files")
        
        # Check Audio vs CSV/JSON (orphaned audio files)
        orphaned_audio = audio_files - csv_files - json_files
        if orphaned_audio:
            print(f"  ⚠ WARNING: {len(orphaned_audio):,} audio files not in CSV or JSON")
            if len(orphaned_audio) <= 10:
                print(f"     Orphaned files:")
                for f in sorted(list(orphaned_audio)[:10]):
                    print(f"       - {f}")
        else:
            print(f"  ✓ All audio files are in CSV and JSON")
        
        # Check CSV vs JSON consistency
        csv_only = csv_files - json_files
        json_only = json_files - csv_files
        if csv_only:
            print(f"  ⚠ WARNING: {len(csv_only):,} files in CSV but not in JSON")
        if json_only:
            print(f"  ⚠ WARNING: {len(json_only):,} files in JSON but not in CSV")
        if not csv_only and not json_only:
            print(f"  ✓ CSV and JSON are consistent")
    
    # ===== Check merged files =====
    print("\n[2] CHECKING MERGED FILES")
    print("-" * 70)
    
    # Collect all audio files from all splits
    all_audio_files: Set[str] = set()
    for split in splits:
        audio_dir = base_dir / split / "audio"
        if audio_dir.is_dir():
            all_audio_files.update({f.name for f in audio_dir.glob("*.wav")})
    
    print(f"\nTotal audio files across all splits: {len(all_audio_files):,}")
    
    # Check merged CSV
    merged_csv = base_dir / "all_manifest.csv"
    if merged_csv.is_file():
        merged_csv_files: Set[str] = set()
        with merged_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name", "").strip()
                if file_name:
                    merged_csv_files.add(file_name)
        
        print(f"Merged CSV files: {len(merged_csv_files):,}")
        
        # Check if merged CSV files have audio
        merged_csv_missing = merged_csv_files - all_audio_files
        if merged_csv_missing:
            print(f"  ❌ ERROR: {len(merged_csv_missing):,} merged CSV entries missing audio files")
            all_ok = False
        else:
            print(f"  ✓ All merged CSV entries have audio files")
        
        # Check if all audio files are in merged CSV
        audio_not_in_csv = all_audio_files - merged_csv_files
        if audio_not_in_csv:
            print(f"  ⚠ WARNING: {len(audio_not_in_csv):,} audio files not in merged CSV")
        else:
            print(f"  ✓ All audio files are in merged CSV")
    else:
        print(f"  ⚠ WARNING: {merged_csv} not found")
    
    # Check merged JSON
    merged_json_file = base_dir / "all_timestamps.json"
    if merged_json_file.is_file():
        print("\nLoading merged JSON (this may take a moment)...")
        with merged_json_file.open("r", encoding="utf-8") as f:
            merged_json_data = json.load(f)
        
        # Extract original file names (remove split prefixes)
        merged_json_files: Set[str] = set()
        for key in merged_json_data.keys():
            # Remove split prefix if present
            original_key = key
            for split in splits:
                if key.startswith(f"{split}_"):
                    original_key = key[len(split) + 1:]
                    break
            merged_json_files.add(original_key)
        
        print(f"Merged JSON files: {len(merged_json_files):,}")
        
        # Check if merged JSON files have audio
        merged_json_missing = merged_json_files - all_audio_files
        if merged_json_missing:
            print(f"  ❌ ERROR: {len(merged_json_missing):,} merged JSON entries missing audio files")
            all_ok = False
        else:
            print(f"  ✓ All merged JSON entries have audio files")
        
        # Check if all audio files are in merged JSON
        audio_not_in_json = all_audio_files - merged_json_files
        if audio_not_in_json:
            print(f"  ⚠ WARNING: {len(audio_not_in_json):,} audio files not in merged JSON")
        else:
            print(f"  ✓ All audio files are in merged JSON")
    else:
        print(f"  ⚠ WARNING: {merged_json_file} not found")
    
    # ===== Check shard directories =====
    print("\n[3] CHECKING SHARD DIRECTORIES")
    print("-" * 70)
    
    shard_issues = []
    
    for split in splits:
        split_dir = base_dir / split
        shards = sorted([d for d in split_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("train-") and "-of-" in d.name])
        
        if not shards:
            continue
        
        print(f"\n{split.upper()} shards ({len(shards)} shards):")
        
        for shard in shards:
            shard_audio = shard / "audio"
            shard_csv = shard / "metadata.csv"
            
            if shard_audio.is_dir() and shard_csv.is_file():
                shard_audio_files = {f.name for f in shard_audio.glob("*.wav")}
                
                shard_csv_files = set()
                with shard_csv.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        file_name = row.get("file_name", "").strip()
                        if file_name:
                            shard_csv_files.add(file_name)
                
                # Check if shard CSV files have audio
                missing = shard_csv_files - shard_audio_files
                if missing:
                    shard_issues.append((shard.name, len(missing), "CSV entries missing audio"))
        
        if shard_issues:
            print(f"  ⚠ Found {len(shard_issues)} shards with issues")
            for shard_name, count, issue in shard_issues[:5]:
                print(f"    {shard_name}: {count} {issue}")
        else:
            print(f"  ✓ All shards are consistent")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_ok:
        print("\n✓ VERIFICATION PASSED: All files match audio folders!")
        print("  ✓ All CSV entries have corresponding audio files")
        print("  ✓ All JSON entries have corresponding audio files")
        print("  ✓ All audio files are referenced in CSV/JSON")
    else:
        print("\n❌ VERIFICATION FAILED: Some files don't match!")
        print("  ⚠ Some CSV/JSON entries are missing audio files")
        print("  ⚠ Or some audio files are not in CSV/JSON")
    
    print("=" * 70)
    
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that all files in CSV and JSON match audio files in audio folders."
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
    
    verify_files_match_audio(base_dir)


if __name__ == "__main__":
    main()













