import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def calculate_duration_from_timestamps(entry: dict) -> float:
    """Calculate duration from word timestamps."""
    words = entry.get("words", [])
    if not words:
        return 0.0
    return words[-1]["end"] - words[0]["start"]


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using soundfile (fast) or ffprobe (fallback)."""
    # Try soundfile first (much faster)
    if HAS_SOUNDFILE:
        try:
            info = sf.info(audio_path)
            return float(info.duration)
        except Exception:
            pass
    
    # Fallback to ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def calculate_librispeech_duration(base_dir: Path) -> Dict[str, float]:
    """Calculate total duration for librispeech_alignments dataset."""
    print("\n" + "=" * 70)
    print("LIBRISPEECH ALIGNMENTS DATASET")
    print("=" * 70)
    
    splits = ["train", "val", "test"]
    total_duration = 0.0
    split_durations = {}
    
    for split in splits:
        split_dir = base_dir / split
        timestamps_json = split_dir / "timestamps.json"
        
        if not timestamps_json.is_file():
            print(f"  ⚠ WARNING: {timestamps_json} not found")
            continue
        
        print(f"\n{split.upper()} split:")
        with timestamps_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            entries = data
        else:
            entries = list(data.values())
        
        split_duration = 0.0
        for entry in entries:
            duration = calculate_duration_from_timestamps(entry)
            split_duration += duration
        
        split_durations[split] = split_duration
        total_duration += split_duration
        
        print(f"  Files: {len(entries):,}")
        print(f"  Total duration: {split_duration:,.2f} seconds")
        print(f"  Total duration: {split_duration/60:,.2f} minutes")
        print(f"  Total duration: {split_duration/3600:,.2f} hours")
    
    split_durations["total"] = total_duration
    return split_durations


def calculate_vietspeech_duration(base_dir: Path, use_ffprobe: bool = False) -> Dict[str, float]:
    """Calculate total duration for VietSpeech dataset."""
    print("\n" + "=" * 70)
    print("VIETSPEECH DATASET")
    print("=" * 70)
    
    splits = ["train", "val", "test"]
    total_duration = 0.0
    split_durations = {}
    
    for split in splits:
        split_dir = base_dir / split
        manifest_csv = split_dir / "manifest.csv"
        
        if not manifest_csv.is_file():
            print(f"  ⚠ WARNING: {manifest_csv} not found")
            continue
        
        print(f"\n{split.upper()} split:")
        
        audio_dir = split_dir / "audio"
        rows_processed = 0
        split_duration = 0.0
        
        with manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if duration column exists
            has_duration_col = "duration" in reader.fieldnames
            
            for row in reader:
                rows_processed += 1
                
                if has_duration_col:
                    try:
                        duration = float(row.get("duration", 0))
                        split_duration += duration
                    except (ValueError, TypeError):
                        pass
                elif use_ffprobe or not has_duration_col:
                    # Try to get duration from audio file
                    file_name = row.get("file_name", "").strip()
                    if file_name:
                        audio_path = audio_dir / file_name
                        if audio_path.exists():
                            duration = get_audio_duration(audio_path)
                            split_duration += duration
                
                # Progress indicator for large files
                if rows_processed % 10000 == 0:
                    print(f"    Processed {rows_processed:,} files...")
        
        split_durations[split] = split_duration
        total_duration += split_duration
        
        print(f"  Files: {rows_processed:,}")
        if split_duration > 0:
            print(f"  Total duration: {split_duration:,.2f} seconds")
            print(f"  Total duration: {split_duration/60:,.2f} minutes")
            print(f"  Total duration: {split_duration/3600:,.2f} hours")
        else:
            if not use_ffprobe:
                print(f"  ⚠ No duration data found. Use --use-ffprobe to calculate from audio files.")
    
    split_durations["total"] = total_duration
    return split_durations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate total audio duration for datasets."
    )
    parser.add_argument(
        "--librispeech-dir",
        type=str,
        help="LibriSpeech alignments directory (e.g. data/processed/librispeech_alignments)",
    )
    parser.add_argument(
        "--vietspeech-dir",
        type=str,
        help="VietSpeech directory (e.g. data/raw/VietSpeech/processed_dataset_structured)",
    )
    parser.add_argument(
        "--use-ffprobe",
        action="store_true",
        help="Calculate duration from audio files (uses soundfile if available, otherwise ffprobe)",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALCULATING TOTAL AUDIO DURATION")
    print("=" * 70)
    
    all_results = {}
    
    if args.librispeech_dir:
        librispeech_dir = Path(args.librispeech_dir)
        if librispeech_dir.is_dir():
            all_results["librispeech"] = calculate_librispeech_duration(librispeech_dir)
        else:
            print(f"\n⚠ ERROR: {librispeech_dir} is not a directory")
    
    if args.vietspeech_dir:
        vietspeech_dir = Path(args.vietspeech_dir)
        if vietspeech_dir.is_dir():
            all_results["vietspeech"] = calculate_vietspeech_duration(vietspeech_dir, args.use_ffprobe)
        else:
            print(f"\n⚠ ERROR: {vietspeech_dir} is not a directory")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    grand_total = 0.0
    for dataset_name, results in all_results.items():
        dataset_total = results.get("total", 0.0)
        grand_total += dataset_total
        
        print(f"\n{dataset_name.upper()}:")
        for split in ["train", "val", "test"]:
            if split in results:
                duration = results[split]
                print(f"  {split}: {duration/3600:,.2f} hours ({duration/60:,.2f} minutes)")
        print(f"  TOTAL: {dataset_total/3600:,.2f} hours ({dataset_total/60:,.2f} minutes)")
    
    if len(all_results) > 1:
        print(f"\nGRAND TOTAL (all datasets): {grand_total/3600:,.2f} hours ({grand_total/60:,.2f} minutes)")
        print(f"  ({grand_total:,.2f} seconds)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

