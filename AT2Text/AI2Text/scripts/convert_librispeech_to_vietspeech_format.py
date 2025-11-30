import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def calculate_duration(words: List[Dict]) -> float:
    """Calculate duration from word timestamps."""
    if not words:
        return 0.0
    return words[-1]["end"] - words[0]["start"]


def convert_librispeech_to_vietspeech_format(base_dir: Path, dry_run: bool = False) -> bool:
    """
    Convert librispeech_alignments timestamps.json from list format to dictionary format
    matching VietSpeech processed_dataset_structured structure.
    """
    print("=" * 70)
    print("CONVERTING LIBRISPEECH TO VIETSPEECH FORMAT")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified")
    else:
        print("\n⚠ LIVE MODE - Files will be modified")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n{split.upper()} split:")
        print("-" * 70)
        
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        manifest_csv = split_dir / "manifest.csv"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found")
            continue
        
        # Load timestamps (list format)
        print("  Loading timestamps.json...")
        with timestamps_json.open("r", encoding="utf-8") as f:
            timestamps_data = json.load(f)
        
        # Create mapping from id to entry
        id_to_entry = {}
        if isinstance(timestamps_data, list):
            for entry in timestamps_data:
                entry_id = entry.get("id", "")
                if entry_id:
                    id_to_entry[entry_id] = entry
        else:
            id_to_entry = timestamps_data
        
        print(f"  Found {len(id_to_entry):,} entries in timestamps.json")
        
        # Load manifest to get transcripts and audio paths
        print("  Loading manifest.csv...")
        id_to_transcript = {}
        id_to_audio_path = {}
        
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry_id = row.get("id", "").strip()
                transcript = row.get("transcript", "").strip()
                audio_path = row.get("audio_path", "").strip()
                
                if entry_id:
                    id_to_transcript[entry_id] = transcript
                    id_to_audio_path[entry_id] = audio_path
        
        print(f"  Found {len(id_to_transcript):,} entries in manifest.csv")
        
        # Convert to VietSpeech format
        print("  Converting to VietSpeech format...")
        converted_data = {}
        
        for entry_id, entry in id_to_entry.items():
            words = entry.get("words", [])
            if not words:
                continue
            
            # Get file name from audio_path or construct from id
            audio_path = id_to_audio_path.get(entry_id, f"audio/{entry_id}.wav")
            file_name = Path(audio_path).name
            
            # Get transcript
            transcript = id_to_transcript.get(entry_id, "")
            
            # Calculate duration
            duration = calculate_duration(words)
            
            # Convert words to segments (add score field)
            segments = []
            for word_entry in words:
                segments.append({
                    "word": word_entry.get("word", ""),
                    "start": word_entry.get("start", 0.0),
                    "end": word_entry.get("end", 0.0),
                    "score": 0.0  # Default score, LibriSpeech doesn't have confidence scores
                })
            
            # Create VietSpeech format entry
            converted_data[file_name] = {
                "duration": duration,
                "text": transcript,
                "segments": segments,
                "audio_filepath": audio_path
            }
        
        print(f"  Converted {len(converted_data):,} entries")
        
        # Write back to timestamps.json
        if not dry_run:
            print("  Writing converted data to timestamps.json...")
            with timestamps_json.open("w", encoding="utf-8") as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Updated {timestamps_json}")
        else:
            # Show sample in dry run
            if converted_data:
                sample_key = list(converted_data.keys())[0]
                sample = converted_data[sample_key]
                print(f"\n  Sample converted entry:")
                print(f"    Key: {sample_key}")
                print(f"    Duration: {sample['duration']:.2f}s")
                print(f"    Text: {sample['text'][:50]}...")
                print(f"    Segments: {len(sample['segments'])} words")
                print(f"    First segment: {sample['segments'][0] if sample['segments'] else 'None'}")
    
    print("\n" + "=" * 70)
    if dry_run:
        print("⚠ This was a DRY RUN - no files were actually modified")
        print("  Run without --dry-run to apply changes")
    else:
        print("✓ Conversion complete!")
        print("  All timestamps.json files now match VietSpeech format")
    print("=" * 70)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert librispeech_alignments timestamps.json to match VietSpeech format."
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
        help="Show what would be converted without actually modifying files",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        return
    
    convert_librispeech_to_vietspeech_format(
        base_dir, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

