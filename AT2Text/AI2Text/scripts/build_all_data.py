"""
Build comprehensive dataset manifest including ALL available data:
- All VIVOS train data (11,660 files)
- All VIVOS test data (760 files)  
- Extended VLSP data (50k train, 5k val)
"""

import csv
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
import itertools


def normalize_transcript(text: str) -> str:
    """Normalize transcript text."""
    if not text:
        return ""
    return text.strip().lower()


def process_vivos_train(vivos_root: Path, output_dir: Path, manifest_rows: List[Dict]) -> None:
    """Process all VIVOS training data."""
    print("\n=== Processing VIVOS Training Data ===")
    prompts_file = vivos_root / "train" / "prompts.txt"
    waves_dir = vivos_root / "train" / "waves"
    output_train_dir = output_dir / "train" / "vivos"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    
    # Read prompts
    prompts = {}
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, transcript = parts
                prompts[file_id] = transcript
    
    print(f"Found {len(prompts)} prompts")
    
    # Process all speaker directories
    speaker_dirs = sorted(waves_dir.glob("VIVOSSPK*"))
    print(f"Processing {len(speaker_dirs)} speakers...")
    
    idx = 0
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = speaker_dir.name
        wav_files = sorted(speaker_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            file_id = f"{speaker_id}_{wav_file.stem}"
            
            if file_id not in prompts:
                continue
            
            transcript = normalize_transcript(prompts[file_id])
            
            # Load and resample to 16kHz if needed
            try:
                audio, sr = librosa.load(str(wav_file), sr=16000, mono=True)
                
                # Save to output directory
                output_file = output_train_dir / f"vivos_train_{idx:06d}.wav"
                sf.write(str(output_file), audio, 16000)
                
                manifest_rows.append({
                    "file_path": str(output_file.resolve()),
                    "transcript": transcript,
                    "split": "train",
                    "speaker_id": speaker_id
                })
                idx += 1
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
    
    print(f"Processed {idx} VIVOS training files")


def process_vivos_test(vivos_root: Path, output_dir: Path, manifest_rows: List[Dict]) -> None:
    """Process all VIVOS test data."""
    print("\n=== Processing VIVOS Test Data ===")
    prompts_file = vivos_root / "test" / "prompts.txt"
    waves_dir = vivos_root / "test" / "waves"
    output_test_dir = output_dir / "test" / "vivos"
    output_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Read prompts
    prompts = {}
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, transcript = parts
                prompts[file_id] = transcript
    
    print(f"Found {len(prompts)} test prompts")
    
    # Process all speaker directories
    speaker_dirs = sorted(waves_dir.glob("VIVOSDEV*"))
    print(f"Processing {len(speaker_dirs)} test speakers...")
    
    idx = 0
    for speaker_dir in tqdm(speaker_dirs, desc="Processing test speakers"):
        speaker_id = speaker_dir.name
        wav_files = sorted(speaker_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            file_id = f"{speaker_id}_{wav_file.stem}"
            
            if file_id not in prompts:
                continue
            
            transcript = normalize_transcript(prompts[file_id])
            
            # Load and resample to 16kHz if needed
            try:
                audio, sr = librosa.load(str(wav_file), sr=16000, mono=True)
                
                # Save to output directory
                output_file = output_test_dir / f"vivos_test_{idx:06d}.wav"
                sf.write(str(output_file), audio, 16000)
                
                manifest_rows.append({
                    "file_path": str(output_file.resolve()),
                    "transcript": transcript,
                    "split": "test",
                    "speaker_id": speaker_id
                })
                idx += 1
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
    
    print(f"Processed {idx} VIVOS test files")


def process_vlsp_extended(output_dir: Path, manifest_rows: List[Dict], train_count: int = 50000, val_count: int = 5000) -> None:
    """Process extended VLSP dataset."""
    print(f"\n=== Processing Extended VLSP Data ({train_count} train, {val_count} val) ===")
    
    vlsp_stream = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train", streaming=True)
    vlsp_stream = vlsp_stream.cast_column("audio", Audio(sampling_rate=16000))
    
    output_train_dir = output_dir / "train" / "vlsp"
    output_val_dir = output_dir / "val" / "vlsp"
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Take train samples
    train_iter = itertools.islice(vlsp_stream, train_count)
    print(f"Processing {train_count} VLSP training samples...")
    
    for idx, sample in enumerate(tqdm(train_iter, total=train_count, desc="VLSP train")):
        audio_info = sample["audio"]
        transcript = normalize_transcript(sample.get("transcription", ""))
        speaker = sample.get("speaker_id", "vlsp")
        
        output_file = output_train_dir / f"vlsp_train_{idx:06d}.wav"
        sf.write(str(output_file), np.asarray(audio_info["array"], dtype=np.float32), audio_info["sampling_rate"])
        
        manifest_rows.append({
            "file_path": str(output_file.resolve()),
            "transcript": transcript,
            "split": "train",
            "speaker_id": speaker
        })
    
    # Skip to validation samples
    val_iter = itertools.islice(vlsp_stream, val_count)
    print(f"Processing {val_count} VLSP validation samples...")
    
    for idx, sample in enumerate(tqdm(val_iter, total=val_count, desc="VLSP val")):
        audio_info = sample["audio"]
        transcript = normalize_transcript(sample.get("transcription", ""))
        speaker = sample.get("speaker_id", "vlsp")
        
        output_file = output_val_dir / f"vlsp_val_{idx:06d}.wav"
        sf.write(str(output_file), np.asarray(audio_info["array"], dtype=np.float32), audio_info["sampling_rate"])
        
        manifest_rows.append({
            "file_path": str(output_file.resolve()),
            "transcript": transcript,
            "split": "val",
            "speaker_id": speaker
        })


def write_manifest(rows: List[Dict], destination: Path) -> None:
    """Write manifest CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_path", "transcript", "split", "speaker_id"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = Path(__file__).resolve().parent.parent
    vivos_root = project_root / "data" / "external" / "vivos_raw" / "vivos"
    audio_output_dir = project_root / "data" / "raw" / "all_data"
    manifest_path = project_root / "data" / "manifests" / "all_data_manifest.csv"
    
    if not vivos_root.exists():
        print(f"ERROR: VIVOS data not found at {vivos_root}")
        print("Please run scripts/build_vlsp_vivos_dataset.py first to download VIVOS data.")
        return
    
    manifest_rows: List[Dict] = []
    
    # Process all VIVOS train data
    process_vivos_train(vivos_root, audio_output_dir, manifest_rows)
    
    # Process all VIVOS test data
    process_vivos_test(vivos_root, audio_output_dir, manifest_rows)
    
    # Process extended VLSP data (50k train, 5k val)
    process_vlsp_extended(audio_output_dir, manifest_rows, train_count=50000, val_count=5000)
    
    # Write manifest
    write_manifest(manifest_rows, manifest_path)
    
    print(f"\n=== Summary ===")
    print(f"Manifest written to: {manifest_path}")
    print(f"Total files: {len(manifest_rows)}")
    
    splits = {}
    for row in manifest_rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count:,} files")
    
    print(f"\nNext step: Import into database with:")
    print(f"  python scripts/prepare_data.py --csv {manifest_path} --audio_base /")


if __name__ == "__main__":
    main()

