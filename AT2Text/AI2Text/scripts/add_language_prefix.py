import argparse
import json
from pathlib import Path


def add_language_prefix(base_dir: Path, language_prefix: str = "<|en|>", dry_run: bool = False) -> bool:
    """
    Add language prefix to text fields in timestamps.json files.
    Matches the format used in VietSpeech (which has <|vi|>).
    """
    print("=" * 70)
    print("ADDING LANGUAGE PREFIX TO TIMESTAMPS.JSON")
    print("=" * 70)
    
    print(f"\nLanguage prefix: {language_prefix}")
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified")
    else:
        print("\n⚠ LIVE MODE - Files will be modified")
    
    splits = ["train", "val", "test"]
    total_updated = 0
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        print("-" * 70)
        
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        # Load timestamps
        print("  Loading timestamps.json...")
        with timestamps_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print(f"  ⚠ WARNING: timestamps.json is not in dictionary format")
            continue
        
        print(f"  Found {len(data):,} entries")
        
        # Update text fields
        updated_count = 0
        for file_name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            
            text = entry.get("text", "").strip()
            if not text:
                continue
            
            # Check if prefix already exists
            if text.startswith(language_prefix):
                continue
            
            # Add prefix
            entry["text"] = f"{language_prefix} {text}"
            updated_count += 1
        
        total_updated += updated_count
        print(f"  Updated {updated_count:,} entries")
        
        # Write back
        if updated_count > 0 and not dry_run:
            print("  Writing updated data to timestamps.json...")
            with timestamps_json.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Updated {timestamps_json}")
        elif updated_count == 0:
            print("  ℹ All entries already have the prefix")
        
        # Show sample in dry run
        if dry_run and updated_count > 0:
            for file_name, entry in data.items():
                if isinstance(entry, dict) and entry.get("text", "").startswith(language_prefix):
                    print(f"\n  Sample updated entry:")
                    print(f"    Key: {file_name}")
                    print(f"    Text: {entry.get('text', '')[:80]}...")
                    break
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal entries updated: {total_updated:,}")
    
    if dry_run:
        print("\n⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("\n✓ Language prefix added!")
        print(f"  All text fields now have '{language_prefix}' prefix")
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add language prefix to text fields in timestamps.json files."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/processed/librispeech_alignments",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="<|en|>",
        help="Language prefix to add (default: <|en|>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually modifying files",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    add_language_prefix(
        base_dir, 
        language_prefix=args.prefix,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

