import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, List


def merge_split(split_dir: Path) -> None:
    """
    Merge all shard subdirectories within a split (train/val/test) into a
    single `audio/` folder plus consolidated `manifest.csv` and `timestamps.json`.

    Expected input layout for each shard:
        split_dir/
          train-00000-of-00027/
            audio/*.wav
            metadata.csv          # columns: file_name,transcription
            train-00000-of-00027.json  # mapping file_name -> {duration, text, segments}
          train-00001-of-00027/
          ...

    Output layout (per split):
        split_dir/
          audio/                  # all WAV files from shards
          manifest.csv            # concatenated metadata.csv files
          timestamps.json         # merged JSON objects from all shard JSON files
    """
    if not split_dir.is_dir():
        return

    shard_dirs: List[Path] = sorted(
        d for d in split_dir.iterdir()
        if d.is_dir() and d.name.startswith("train-") and "-of-" in d.name
    )
    if not shard_dirs:
        return

    audio_out = split_dir / "audio"
    audio_out.mkdir(exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    merged_json: Dict[str, dict] = {}

    for shard in shard_dirs:
        # 1) Copy audio
        shard_audio = shard / "audio"
        if shard_audio.is_dir():
            for wav in shard_audio.glob("*.wav"):
                dest = audio_out / wav.name
                # If a file already exists with the same name, keep the first one
                if not dest.exists():
                    shutil.copy2(wav, dest)

        # 2) Append CSV rows
        metadata_csv = shard / "metadata.csv"
        if metadata_csv.is_file():
            with metadata_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Normalize keys in case of minor differences
                    row = {k.strip(): v for k, v in row.items()}
                    manifest_rows.append(row)

        # 3) Merge JSON
        shard_json = next(shard.glob("*.json"), None)
        if shard_json and shard_json.is_file():
            with shard_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Later shards win on key collision, but this should be rare
            merged_json.update(data)

    # Write manifest.csv
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        manifest_path = split_dir / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    # Write timestamps.json
    if merged_json:
        timestamps_path = split_dir / "timestamps.json"
        with timestamps_path.open("w", encoding="utf-8") as f:
            json.dump(merged_json, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge VietSpeech shard directories into split-level audio/ + manifest + timestamps."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing train/ val/ test/ split folders, "
             "e.g. data/raw/VietSpeech/processed_dataset_structured",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to merge (default: train val test).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    for split in args.splits:
        split_dir = base_dir / split
        merge_split(split_dir)


if __name__ == "__main__":
    main()


