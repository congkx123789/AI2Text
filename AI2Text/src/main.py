# src/main.py
import argparse, sys, os, uvicorn
from ai2text.audio.inference import transcribe
from ai2text.image.inference import extract_text
from ai2text.config import CONFIG
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_cli(args):
    """Run the old CLI pipeline."""
    path = args.input
    if not os.path.exists(path):
        print(f"Input not found: {path}", file=sys.stderr)
        sys.exit(1)

    audio_exts = {".wav",".mp3",".m4a",".flac",".ogg",".aac",".wma",".webm",".mp4"}
    image_exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".pbm",".ppm"}

    mode = args.mode
    if mode == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext in audio_exts: mode = "audio"
        elif ext in image_exts: mode = "image"
        else:
            print("Cannot auto-detect file type. Use --mode {audio|image}.", file=sys.stderr)
            sys.exit(2)

    if mode == "audio":
        text = transcribe(path, backend=args.stt_backend)
    else:
        text = extract_text(path, backend=args.ocr_backend)

    print(text)

def run_server():
    """Start FastAPI server (calls app in server.py)."""
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)

def main():
    ap = argparse.ArgumentParser(description="AI2Text: sound/image → text (offline).")
    ap.add_argument("input", nargs="?", help="Path to an audio or image file (optional)")
    ap.add_argument("--mode", choices=["auto", "audio", "image"], default="auto")
    ap.add_argument("--stt-backend", choices=["vosk","whisper"], default=None)
    ap.add_argument("--ocr-backend", choices=["tesseract","easyocr"], default=None)
    ap.add_argument("--server", action="store_true", help="Run FastAPI server instead of CLI")
    args = ap.parse_args()

    if args.server or args.input is None:
        run_server()
    else:
        run_cli(args)

if __name__ == "__main__":
    main()
