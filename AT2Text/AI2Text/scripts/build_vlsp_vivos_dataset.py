"""
Utility script to build a combined VLSP (train/val) + VIVOS (test) dataset
manifests at 16 kHz with standardized transcript columns.
"""

from __future__ import annotations

import csv
import itertools
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download


@dataclass
class SamplePreview:
    """Quick preview of a sample for logging/inspection."""

    text: str
    sampling_rate: int
    num_samples: int


def take(dataset: IterableDataset, count: int) -> Iterator[dict]:
    """Take the first `count` samples from a streaming dataset."""
    return itertools.islice(dataset, count)


def skip(dataset: IterableDataset, count: int) -> Iterator[dict]:
    """Skip the first `count` samples from a streaming dataset."""
    return itertools.islice(dataset, count, None)


def standardize_columns(sample: dict) -> dict:
    """Ensure all datasets expose a `transcription` field."""
    if "sentence" in sample and "transcription" not in sample:
        sample["transcription"] = sample["sentence"]
        del sample["sentence"]
    return sample


def cast_to_16khz(dataset: IterableDataset) -> IterableDataset:
    """Force audio column to 16 kHz for downstream consistency."""
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def preview_sample(iterable: Iterable[dict]) -> SamplePreview:
    """Grab a single sample for inspection."""
    iterator = iter(iterable)
    sample = next(iterator)
    audio_arr = np.asarray(sample["audio"]["array"])
    text = sample.get("transcription") or sample.get("sentence", "")
    return SamplePreview(
        text=text,
        sampling_rate=sample["audio"]["sampling_rate"],
        num_samples=audio_arr.shape[0],
    )


def save_audio(array: np.ndarray, sr: int, path: Path) -> None:
    """Write audio samples to disk in WAV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sr)


def normalize_transcript(text: Optional[str]) -> str:
    """Lowercase and strip transcripts; fallback to empty string."""
    if not text:
        return ""
    return text.strip().lower()


def ensure_vivos_local_copy(project_root: Path) -> Path:
    """Download and extract the VIVOS dataset archive if needed."""
    target_dir = project_root / "data" / "external" / "vivos_raw"
    extracted_root = target_dir / "vivos"
    if extracted_root.exists():
        return extracted_root

    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(repo_id="vivos", repo_type="dataset", filename="data/vivos.tar.gz")

    def is_within_directory(directory: Path, target: Path) -> bool:
        try:
            directory = directory.resolve()
            target = target.resolve()
            return str(target).startswith(str(directory))
        except FileNotFoundError:
            # Target may not exist yet; fallback to string comparison
            return str(target).startswith(str(directory))

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = target_dir / member.name
            if not is_within_directory(target_dir, member_path):
                raise RuntimeError("Attempted Path Traversal in Tar File")
        tar.extractall(target_dir)

    return extracted_root


def iter_vivos_test_samples(vivos_root: Path) -> Iterator[dict]:
    """Yield VIVOS test samples with audio arrays and transcripts."""
    test_dir = vivos_root / "test"
    prompts_file = test_dir / "prompts.txt"
    waves_dir = test_dir / "waves"

    with prompts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            speaker = utt_id.split("_")[0]
            audio_path = waves_dir / speaker / f"{utt_id}.wav"
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            yield {
                "audio": {"array": audio, "sampling_rate": sr},
                "transcription": transcript,
                "speaker": speaker,
            }


def export_split(
    iterator: Iterable[dict],
    dataset_name: str,
    split: str,
    output_root: Path,
    manifest_rows: List[Dict[str, str]],
) -> None:
    """Materialize a streaming iterator to disk and append manifest rows."""
    for idx, sample in enumerate(iterator):
        audio_info = sample["audio"]
        transcript = normalize_transcript(sample.get("transcription"))
        speaker = sample.get("speaker") or sample.get("speaker_id") or dataset_name

        filename = f"{dataset_name}_{split}_{idx:06d}.wav"
        file_path = output_root / split / dataset_name / filename

        save_audio(np.asarray(audio_info["array"], dtype=np.float32), audio_info["sampling_rate"], file_path)

        manifest_rows.append(
            {
                "file_path": str(file_path.resolve()),
                "transcript": transcript,
                "split": split,
                "speaker_id": speaker,
            }
        )


def write_manifest(rows: List[Dict[str, str]], destination: Path) -> None:
    """Write manifest CSV in the format expected by prepare_data.py."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_path", "transcript", "split", "speaker_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    audio_output_dir = project_root / "data" / "raw" / "vlsp_vivos"
    manifest_path = project_root / "data" / "manifests" / "vlsp_vivos_manifest.csv"
    vivos_root = ensure_vivos_local_copy(project_root)

    print("=== Step 1: Install dependencies (ensure datasets, soundfile, librosa) ===")
    print("Already assumed installed inside virtual environment.\n")

    print("=== Step 2.1: Loading VLSP 100h (Streaming) ===")
    vlsp_stream = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train", streaming=True)
    vlsp_stream = cast_to_16khz(vlsp_stream)

    train_iter = take(vlsp_stream, 10_000)
    val_iter = take(skip(vlsp_stream, 10_000), 1_000)
    print("VLSP splits ready: 10,000 train / 1,000 val\n")

    print("=== Step 2.2: Loading VIVOS Official Test ===")
    vivos_preview_iter = iter_vivos_test_samples(vivos_root)
    preview = preview_sample(vivos_preview_iter)
    print("VIVOS test loaded from local archive.\n")

    print("=== Step 3: Preview sample from VIVOS ===")
    print(f"Text: {preview.text}")
    print(f"Sampling Rate: {preview.sampling_rate} Hz")
    print(f"Audio Samples: {preview.num_samples}\n")

    vivos_stream = iter_vivos_test_samples(vivos_root)

    print("=== Step 4: Materializing audio + manifest ===")
    manifest_rows: List[Dict[str, str]] = []

    export_split(train_iter, "vlsp", "train", audio_output_dir, manifest_rows)
    export_split(val_iter, "vlsp", "val", audio_output_dir, manifest_rows)
    export_split(vivos_stream, "vivos", "test", audio_output_dir, manifest_rows)

    write_manifest(manifest_rows, manifest_path)
    print(f"\nManifest written to {manifest_path}")
    print(f"Total rows: {len(manifest_rows)}")
    print("Splits summary:")
    splits = {}
    for row in manifest_rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    for split, count in splits.items():
        print(f"  - {split}: {count}")

    print("\n=== Step 5: Pipeline Ready ===")
    print("1) Train: VLSP first 10k samples")
    print("2) Val:   VLSP next 1k samples")
    print("3) Test:  VIVOS official benchmark\n")
    print("Use the CSV with scripts/prepare_data.py to import into the database.")


if __name__ == "__main__":
    main()

