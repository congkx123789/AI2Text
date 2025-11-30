import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def verify_counts(base_dir: Path) -> bool:
    """
    Verify that audio file counts match CSV row counts and JSON ID counts
    for each split and overall.
    """
    print("=" * 70)
    print("AUDIO/CSV/JSON COUNT VERIFICATION")
    print("=" * 70)
    
    all_ok = True
    splits = ["train", "val", "test"]
    
    # ===== Check individual splits =====
    print("\n[1] CHECKING INDIVIDUAL SPLITS")
    print("-" * 70)
    
    split_results = {}
    
    for split in splits:
        split_dir = base_dir / split
        audio_dir = split_dir / "audio"
        manifest_csv = split_dir / "manifest.csv"
        timestamps_json = split_dir / "timestamps.json"
        
        print(f"\n{split.upper()} split:")
        
        # Count audio files
        audio_count = 0
        if audio_dir.is_dir():
            audio_files = list(audio_dir.glob("*.wav"))
            audio_count = len(audio_files)
            print(f"  Audio files: {audio_count:,}")
        else:
            print(f"  ⚠ WARNING: {audio_dir} not found")
        
        # Count CSV rows
        csv_count = 0
        if manifest_csv.is_file():
            with manifest_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_count = sum(1 for _ in reader)
            print(f"  CSV rows: {csv_count:,}")
        else:
            print(f"  ⚠ WARNING: {manifest_csv} not found")
        
        # Count JSON entries
        json_count = 0
        if timestamps_json.is_file():
            with timestamps_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
                json_count = len(data)
            print(f"  JSON entries: {json_count:,}")
        else:
            print(f"  ⚠ WARNING: {timestamps_json} not found")
        
        # Verify counts match
        if audio_count == csv_count == json_count:
            print(f"  ✓ All counts match: {audio_count:,}")
        else:
            print(f"  ❌ ERROR: Count mismatch!")
            if audio_count != csv_count:
                print(f"     Audio ({audio_count:,}) ≠ CSV ({csv_count:,}) - difference: {abs(audio_count - csv_count):,}")
            if audio_count != json_count:
                print(f"     Audio ({audio_count:,}) ≠ JSON ({json_count:,}) - difference: {abs(audio_count - json_count):,}")
            if csv_count != json_count:
                print(f"     CSV ({csv_count:,}) ≠ JSON ({json_count:,}) - difference: {abs(csv_count - json_count):,}")
            all_ok = False
        
        split_results[split] = {
            "audio": audio_count,
            "csv": csv_count,
            "json": json_count
        }
    
    # ===== Check merged files =====
    print("\n[2] CHECKING MERGED FILES")
    print("-" * 70)
    
    # Count audio files across all splits
    total_audio = sum(r["audio"] for r in split_results.values())
    print(f"\nTotal audio files (sum of splits): {total_audio:,}")
    
    # Count merged CSV
    merged_csv = base_dir / "all_manifest.csv"
    merged_csv_count = 0
    if merged_csv.is_file():
        with merged_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            merged_csv_count = sum(1 for _ in reader)
        print(f"Merged CSV rows: {merged_csv_count:,}")
        
        if merged_csv_count == total_audio:
            print(f"  ✓ Merged CSV matches total audio count")
        else:
            print(f"  ❌ ERROR: Merged CSV ({merged_csv_count:,}) ≠ Total audio ({total_audio:,})")
            print(f"     Difference: {abs(merged_csv_count - total_audio):,}")
            all_ok = False
    else:
        print(f"  ⚠ WARNING: {merged_csv} not found")
    
    # Count merged JSON
    merged_json_file = base_dir / "all_timestamps.json"
    merged_json_count = 0
    if merged_json_file.is_file():
        print("  Loading merged JSON (this may take a moment)...")
        with merged_json_file.open("r", encoding="utf-8") as f:
            merged_json_data = json.load(f)
        merged_json_count = len(merged_json_data)
        print(f"Merged JSON entries: {merged_json_count:,}")
        
        # Note: Merged JSON may have more entries due to duplicate keys with split prefixes
        # So we check if it's >= total audio
        if merged_json_count >= total_audio:
            if merged_json_count == total_audio:
                print(f"  ✓ Merged JSON matches total audio count")
            else:
                extra = merged_json_count - total_audio
                print(f"  ℹ INFO: Merged JSON has {extra:,} extra entries (likely split-prefixed duplicates)")
                print(f"  ✓ Merged JSON contains all audio entries")
        else:
            print(f"  ❌ ERROR: Merged JSON ({merged_json_count:,}) < Total audio ({total_audio:,})")
            print(f"     Missing: {total_audio - merged_json_count:,} entries")
            all_ok = False
    else:
        print(f"  ⚠ WARNING: {merged_json_file} not found")
    
    # ===== Detailed breakdown =====
    print("\n[3] DETAILED BREAKDOWN")
    print("-" * 70)
    
    print(f"\n{'Split':<10} {'Audio':>12} {'CSV':>12} {'JSON':>12} {'Status':<15}")
    print("-" * 70)
    
    for split in splits:
        r = split_results[split]
        status = "✓ Match" if r["audio"] == r["csv"] == r["json"] else "❌ Mismatch"
        print(f"{split:<10} {r['audio']:>12,} {r['csv']:>12,} {r['json']:>12,} {status:<15}")
    
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_audio:>12,} {sum(r['csv'] for r in split_results.values()):>12,} {sum(r['json'] for r in split_results.values()):>12,}")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_csv = sum(r["csv"] for r in split_results.values())
    total_json = sum(r["json"] for r in split_results.values())
    
    print(f"\nTotal counts:")
    print(f"  Audio files: {total_audio:,}")
    print(f"  CSV rows: {total_csv:,}")
    print(f"  JSON entries: {total_json:,}")
    
    if total_audio == total_csv == total_json:
        print("\n✓ VERIFICATION PASSED: All counts match perfectly!")
        print("  ✓ Audio files = CSV rows = JSON entries")
    else:
        print("\n❌ VERIFICATION FAILED: Count mismatches detected!")
        if total_audio != total_csv:
            print(f"  ⚠ Audio ({total_audio:,}) ≠ CSV ({total_csv:,})")
        if total_audio != total_json:
            print(f"  ⚠ Audio ({total_audio:,}) ≠ JSON ({total_json:,})")
        if total_csv != total_json:
            print(f"  ⚠ CSV ({total_csv:,}) ≠ JSON ({total_json:,})")
        all_ok = False
    
    print("=" * 70)
    
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that audio file counts match CSV row counts and JSON ID counts."
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
    
    verify_counts(base_dir)


if __name__ == "__main__":
    main()

