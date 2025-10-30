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
