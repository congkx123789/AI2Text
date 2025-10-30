
#!/usr/bin/env python3

"""
make_asr_scaffold.py — Generate a runnable ASR (speech-to-text) project structure with code.

Usage:
  python make_asr_scaffold.py --dest ./ai-llm-ss [--force]

It creates:
  ai-llm-ss/
    data/{raw/audio, raw/text, processed, results}
    src/asr/{tokenizer,features,dataset,model,train_ctc,decode,api}.py
    scripts/{prepare_data,train_asr,serve_asr}.py
    requirements.txt, README.md, pyproject.toml

By default, it WON'T overwrite an existing folder unless you use --force.
"""

import argparse, os, json, shutil, zipfile
from pathlib import Path
from textwrap import dedent

def w(root: Path, rel: str, content: str = ""):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(dedent(content).lstrip("\n"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="ai-llm-ss", help="Destination folder to create")
    ap.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    ap.add_argument("--zip", dest="zip_out", default="", help="Optional: also write a .zip at this path")
    args = ap.parse_args()

    root = Path(args.dest).resolve()
    if root.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {root}. Use --force to overwrite.")
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)

    # Top-level files
    w(root, "README.md", """
    # ai-llm-ss — Minimal Speech-to-Text (ASR) from Scratch

    This is a tiny end-to-end project: audio (.wav) + transcripts (.txt) → train a CTC model → serve a FastAPI `/transcribe` endpoint.

    ## Quickstart (Windows PowerShell)

    ```powershell
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt

    # put a few .wav files in data/raw/audio and matching .txt in data/raw/text
    python .\\scripts\\prepare_data.py --in data\\raw --out data\\processed

    # train (on CPU by default; use --device cuda if available)
    python .\\scripts\\train_asr.py

    # serve
    uvicorn src.asr.api:app --host 127.0.0.1 --port 8001 --reload
    ```
    """)

    w(root, "requirements.txt", """
    torch
    torchaudio
    numpy
    librosa
    jiwer
    fastapi
    uvicorn
    python-multipart
    pydantic>=2
    """)

    w(root, "pyproject.toml", """
    [project]
    name = "ai-llm-ss"
    version = "0.1.0"
    description = "Minimal ASR from scratch (CTC)"
    requires-python = ">=3.9"
    dependencies = []

    [tool.setuptools]
    package-dir = {"" = "src"}

    [tool.setuptools.packages.find]
    where = ["src"]
    """)

    w(root, ".env.example", """
    # Example environment variables (optional)
    ASR_SAMPLE_RATE=16000
    """)

    # Data & misc dirs
    for rel in [
        "data/raw/audio/.gitkeep",
        "data/raw/text/.gitkeep",
        "data/processed/.gitkeep",
        "data/results/.gitkeep",
        "experiments/reports/.gitkeep",
        "notebooks/.gitkeep",
        "tests/.gitkeep",
        "docs/.gitkeep",
    ]:
        w(root, rel, "")

    # src package
    w(root, "src/__init__.py", "")
    w(root, "src/asr/__init__.py", "")

    w(root, "src/asr/tokenizer.py", """
    import json
    from collections import OrderedDict
    from pathlib import Path

    BLANK = "<blank>"
    UNK = "<unk>"

    def build_char_vocab(transcripts, extra_tokens=(BLANK, UNK)):
        chars = set()
        for t in transcripts:
            t = t.lower().strip()
            chars.update(list(t))
        for ch in ["\\n","\\r","\\t"]:
            if ch in chars:
                chars.remove(ch)
        vocab = list(extra_tokens) + sorted(chars)
        stoi = OrderedDict((c,i) for i,c in enumerate(vocab))
        itos = {i:c for c,i in stoi.items()}
        return {"vocab":vocab, "stoi":dict(stoi), "itos":itos}

    def encode(text, stoi):
        return [stoi.get(c, stoi[UNK]) for c in text.lower()]

    def save_vocab(path, vocab):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab["vocab"], f, ensure_ascii=False, indent=2)
    """)

    w(root, "src/asr/features.py", """
    import torch, torchaudio

    def ensure_mono16k(waveform, sr):
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(0, keepdim=True)
        return waveform, 16000

    def wav_to_logmelspec(waveform, sr=16000, n_fft=400, hop=160, n_mels=80):
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )(waveform)
        logmel = torch.log(mel + 1e-6)       # (n_mels, frames)
        return logmel.transpose(0,1)         # (frames, n_mels)
    """)

    w(root, "src/asr/dataset.py", """
    import torch, torchaudio, glob, os, json
    from .features import wav_to_logmelspec, ensure_mono16k

    class ASRDataset(torch.utils.data.Dataset):
        def __init__(self, audio_dir, text_dir, vocab_path):
            self.audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
            self.text_dir = text_dir
            self.vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
            self.stoi = {c:i for i,c in enumerate(self.vocab)}

        def __len__(self): return len(self.audio_paths)

        def __getitem__(self, idx):
            ap = self.audio_paths[idx]
            name = os.path.splitext(os.path.basename(ap))[0]
            tp = os.path.join(self.text_dir, f"{name}.txt")
            transcript = open(tp, "r", encoding="utf-8").read().strip().lower()

            wav, sr = torchaudio.load(ap)
            wav, sr = ensure_mono16k(wav, sr)
            x = wav_to_logmelspec(wav, sr)               # (T, 80)

            y = torch.tensor([self.stoi.get(c, 1) for c in transcript], dtype=torch.long)
            return x, y

    def collate_batch(batch):
        xs, ys = zip(*batch)
        x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
        y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
        X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)                # (B, T, F)
        Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
        return X, x_lens, Y, y_lens
    """)

    w(root, "src/asr/model.py", """
    import torch, torch.nn as nn

    class CRNNCTC(nn.Module):
        def __init__(self, n_mels=80, vocab_size=40, cnn_channels=128, rnn_hidden=256, rnn_layers=3):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_mels, cnn_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.rnn = nn.LSTM(
                input_size=cnn_channels, hidden_size=rnn_hidden,
                num_layers=rnn_layers, batch_first=True, bidirectional=True
            )
            self.head = nn.Linear(rnn_hidden*2, vocab_size)

        def forward(self, x, x_lens):
            # x: (B, T, F)
            x = x.transpose(1,2)    # (B, F, T)
            x = self.cnn(x)         # (B, C, T)
            x = x.transpose(1,2)    # (B, T, C)
            x, _ = self.rnn(x)      # (B, T, 2H)
            logits = self.head(x)   # (B, T, V)
            return logits.transpose(0,1), x_lens  # to (T, B, V)
    """)

    w(root, "src/asr/decode.py", """
    import torch

    def greedy_decode(logits, itos):
        # logits: (T,B,V)
        pred = logits.argmax(dim=-1).transpose(0,1)  # (B,T)
        texts = []
        for seq in pred:
            prev = None; out=[]
            for idx in seq.tolist():
                if idx != 0 and idx != prev:
                    out.append(itos[idx])
                prev = idx
            texts.append("".join(out))
        return texts
    """)

    w(root, "src/asr/train_ctc.py", """
    import argparse, json, torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader
    from pathlib import Path
    from .dataset import ASRDataset, collate_batch
    from .model import CRNNCTC

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--train_audio", default="data/raw/audio")
        ap.add_argument("--train_text",  default="data/raw/text")
        ap.add_argument("--vocab",       default="data/processed/vocab.json")
        ap.add_argument("--epochs", type=int, default=5)
        ap.add_argument("--batch_size", type=int, default=4)
        ap.add_argument("--lr", type=float, default=1e-3)
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--out", default="data/results/asr_ctc.pt")
        args = ap.parse_args()

        ds = ASRDataset(args.train_audio, args.train_text, args.vocab)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
        model = CRNNCTC(n_mels=80, vocab_size=len(ds.vocab)).to(args.device)
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        opt = optim.AdamW(model.parameters(), lr=args.lr)

        for ep in range(1, args.epochs+1):
            model.train(); total = 0.0
            for X, Xlen, Y, Ylen in dl:
                X, Xlen, Y, Ylen = X.to(args.device), Xlen.to(args.device), Y.to(args.device), Ylen.to(args.device)
                logits, out_lens = model(X, Xlen)      # (T,B,V)
                log_probs = logits.log_softmax(dim=-1)
                loss = ctc(log_probs, Y, out_lens, Ylen)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            print(f"epoch {ep} | loss {total/len(dl):.3f}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"Saved checkpoint to {args.out}")

    if __name__ == "__main__":
        main()
    """)

    w(root, "src/asr/api.py", """
    from fastapi import FastAPI, UploadFile, File
    import torch, torchaudio, json
    from .model import CRNNCTC
    from .features import wav_to_logmelspec, ensure_mono16k
    from .decode import greedy_decode

    app = FastAPI(title="ASR CTC API")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB = json.load(open("data/processed/vocab.json","r",encoding="utf-8"))
    ITOS = {i:c for i,c in enumerate(VOCAB)}
    MODEL = CRNNCTC(n_mels=80, vocab_size=len(VOCAB))
    try:
        MODEL.load_state_dict(torch.load("data/results/asr_ctc.pt", map_location=device))
    except FileNotFoundError:
        pass  # allow API to start before training
    MODEL.to(device).eval()

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        wav, sr = torchaudio.load(file.file)
        wav, sr = ensure_mono16k(wav, sr)
        feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)  # (1,T,F)
        with torch.no_grad():
            logits, lens = MODEL(feats, torch.tensor([feats.shape[1]], device=device))
        text = greedy_decode(logits.cpu(), ITOS)[0]
        return {"text": text}
    """)

    # scripts
    w(root, "scripts/prepare_data.py", """
    import argparse, glob, os
    from pathlib import Path
    from src.asr.tokenizer import build_char_vocab, save_vocab

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--in", dest="in_dir", default="data/raw")
        ap.add_argument("--out", dest="out_dir", default="data/processed")
        args = ap.parse_args()

        text_dir = os.path.join(args.in_dir, "text")
        txts = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
        transcripts = [open(p, "r", encoding="utf-8").read().strip() for p in txts]
        vocab = build_char_vocab(transcripts)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        save_vocab(os.path.join(args.out_dir, "vocab.json"), vocab)
        print(f"Saved vocab to {os.path.join(args.out_dir, 'vocab.json')} with {len(vocab['vocab'])} tokens.")

    if __name__ == "__main__":
        main()
    """)

    w(root, "scripts/train_asr.py", """
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
    """)

    w(root, "scripts/serve_asr.py", """
    import subprocess, sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "uvicorn", "src.asr.api:app", "--host", "127.0.0.1", "--port", "8001", "--reload"]))
    """)

    # docs & tests
    w(root, "docs/design.md", "# Design Notes\\n\\nA minimal ASR pipeline using log-Mel + CRNN + CTC.")
    w(root, "docs/eval_protocol.md", "# Evaluation\\n\\nUse CER and WER on a held-out dev set.")
    w(root, "tests/test_decode.py", """
    from src.asr.decode import greedy_decode
    import torch

    def test_greedy_decode_shapes():
        logits = torch.zeros(5, 2, 4)  # (T,B,V)
        itos = {0:"<blank>",1:"a",2:"b",3:" "}
        out = greedy_decode(logits, itos)
        assert len(out) == 2
    """)

    # Example tiny text so vocab builds
    w(root, "data/raw/text/example.txt", "why hello there.")

    # Optional zip
    if args.zip_out:
        zip_path = Path(args.zip_out).resolve()
        from zipfile import ZipFile, ZIP_DEFLATED
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
            for path in root.rglob("*"):
                z.write(path, path.relative_to(root.parent))
        print(f"Zipped project to {zip_path}")

    print(f"Created ASR scaffold at {root}")

if __name__ == "__main__":
    main()
