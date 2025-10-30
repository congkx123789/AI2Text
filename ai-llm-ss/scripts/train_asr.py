import subprocess, sys

# simple wrapper that forwards to module entry-point with defaults
cmd = [sys.executable, "-m", "src.asr.train_ctc"]
cmd += ["--train_audio", "data/raw/audio",
        "--train_text", "data/raw/text",
        "--vocab", "data/processed/vocab.json",
        "--epochs", "5",
        "--batch_size", "4",
        "--lr", "0.001",
        "--device", "cpu",
        "--out", "data/results/asr_ctc.pt"]
raise SystemExit(subprocess.call(cmd))
