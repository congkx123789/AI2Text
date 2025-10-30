from __future__ import annotations
import argparse, json
from pathlib import Path


"""Build a manifest from audio files under data/raw/audio/*.wav|mp3|m4a"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="raw", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()


    audio_dir = Path(args.raw) / "audio"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)


    exts = {".wav", ".mp3", ".m4a", ".flac"}
    with out.open("w", encoding="utf-8") as fo:
        for p in sorted(audio_dir.glob("**/*")):
            if p.suffix.lower() in exts:
                rec = {"id": p.stem, "audio": str(p.resolve())}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Manifest: {out}")


if __name__ == "__main__":
    main()