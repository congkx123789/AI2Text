import subprocess, sys, torch

# simple wrapper that forwards to module entry-point with sensible defaults
device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = "data/processed/merged_dataset/train/manifest.csv"
timestamps = "data/processed/merged_dataset/train/timestamps.json"
cmd = [sys.executable, "-m", "src.asr.train_ctc"]
cmd += [
    "--manifest", manifest,
    "--audio_root", "data/processed/merged_dataset/train",
    "--timestamps", timestamps,
    "--trim_segments",
    "--vocab", "data/processed/vocab.json",
    "--epochs", "5",
    "--batch_size", "80",
    "--gradient_accumulation_steps", "2",
    "--lr", "0.001",
    "--device", device,
    "--num_workers", "10" if device == "cuda" else "6",
    "--amp",
]
raise SystemExit(subprocess.call(cmd))
