#!/usr/bin/env python3
"""
Offline Vietnamese TTS with word-level timing, inspired by CN pipelines.

Pipeline:
1) Load multilingual TTS model (Coqui XTTS v2) locally
2) Convert input text (or .epub) to audio (WAV)
3) Run forced alignment with aeneas to obtain per-word timestamps (JSON)
4) Optionally export SRT with sentence-level or word-level highlighting

Requirements
    # Core
    pip install TTS==0.22.0 ebooklib beautifulsoup4 tqdm

    # Alignment
    pip install aeneas
    # System deps for aeneas (Linux/macOS):
    #   - ffmpeg
    #   - espeak-ng (or espeak)
    # On macOS: brew install ffmpeg espeak
    # On Ubuntu/Debian: sudo apt-get install ffmpeg espeak-ng

    # Windows: aeneas is trickier to install; consider WSL Ubuntu

Usage
    python vi_tts_timing.py \
        --input input.txt \
        --output out_dir \
        --language vi \
        --word-srt

    # EPUB input
    python vi_tts_timing.py --input book.epub --output out_dir --word-srt

Outputs
    - out_dir/audio.wav
    - out_dir/word_timing.json   (per-word timing)
    - out_dir/sentences.srt      (sentence-level subtitle)
    - out_dir/words.srt          (word-level subtitle, optional)

Notes
    - XTTS v2 is large (~1.5GB); first run will download weights.
    - For faster CPU-only inference, reduce text chunk size.
    - If you have a short speaker WAV you like, you can pass --speaker-wav to do light timbre conditioning.
"""

import argparse
import os
from pathlib import Path
import re
import json
from typing import List, Tuple

from tqdm import tqdm
from bs4 import BeautifulSoup
from ebooklib import epub

from TTS.api import TTS

# aeneas imports
from aeneas.exacttiming import TimeValue
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…]|\n)\s+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    text = soup.get_text(" ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_text(input_path: Path) -> str:
    if input_path.suffix.lower() == ".epub":
        book = epub.read_epub(str(input_path))
        chunks: List[str] = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                html = item.get_content().decode(errors="ignore")
                txt = clean_text_from_html(html)
                if txt:
                    chunks.append(txt)
        return "\n\n".join(chunks)
    else:
        return input_path.read_text(encoding="utf-8")


def split_sentences(text: str, max_chars: int = 1500) -> List[str]:
    # coarse sentence split then pack to max_chars
    initial = re.split(SENTENCE_SPLIT_RE, text)
    initial = [s.strip() for s in initial if s and s.strip()]
    pieces: List[str] = []
    cur = ""
    for s in initial:
        if len(cur) + len(s) + 1 > max_chars:
            if cur:
                pieces.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        pieces.append(cur.strip())
    return pieces


def synthesize_xtts(pieces: List[str], out_wav: Path, language: str = "vi", speaker_wav: str = None, use_gpu: bool = True):
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name)
    if use_gpu and tts.is_multi_spkr:
        # TTS() auto-detects GPU if available; nothing else needed
        pass

    # Stitch pieces by concatenating audio segments in sequence
    import soundfile as sf
    import numpy as np
    sr = 24000
    audio_all = np.zeros(0, dtype=np.float32)
    for s in tqdm(pieces, desc="Synth"):
        wav = tts.tts(text=s, speaker_wav=speaker_wav, language=language)
        audio_all = np.concatenate([audio_all, wav])
        # add short pause between pieces
        silence = np.zeros(int(0.2 * sr), dtype=np.float32)
        audio_all = np.concatenate([audio_all, silence])
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio_all, sr)


def forced_align_words(audio_wav: Path, full_text: str, out_json: Path, language: str = "vi"):
    """Use aeneas to align words. We write each word on its own line to get word-level fragments."""
    tokens = tokenize_words_vi(full_text)
    temp_txt = out_json.with_suffix(".words.txt")
    temp_txt.write_text("\n".join(tokens), encoding="utf-8")

    cfg_lang = {
        "vi": "vie",
        "en": "eng",
        "zh": "cmn"
    }.get(language, "vie")

    config_string = f"task_language={cfg_lang}|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = str(audio_wav)
    task.text_file_path_absolute = str(temp_txt)
    task.sync_map_file_path_absolute = str(out_json)

    ExecuteTask(task).execute()

    # clean temp
    try:
        temp_txt.unlink()
    except Exception:
        pass


def tokenize_words_vi(text: str) -> List[str]:
    # Simple tokenization: split on whitespace/punct, keep Vietnamese letters
    # For better results, integrate underthesea or pyvi (optional)
    words = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    return words


def sentences_to_srt(pieces: List[str], audio_wav: Path, out_srt: Path):
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(str(audio_wav))
    # naive duration based on proportional piece lengths
    total_len = len(data) / float(sr)
    lengths = [len(p) for p in pieces]
    total_chars = sum(lengths)
    t = 0.0
    idx = 1
    lines = []
    for L, s in zip(lengths, pieces):
        seg = total_len * (L / total_chars) if total_chars > 0 else 0
        start = t
        end = t + seg
        lines.append(f"{idx}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{s}\n\n")
        idx += 1
        t = end
    out_srt.write_text("".join(lines), encoding="utf-8")


def words_json_to_srt(words_json: Path, out_srt: Path):
    obj = json.loads(words_json.read_text(encoding="utf-8"))
    # aeneas JSON has fragments with 'begin' and 'end' times for each line (here each word)
    entries = []
    fragments = obj.get("fragments", [])
    for i, f in enumerate(fragments, 1):
        text = f.get("lines", [""])[0]
        start = float(f.get("begin", 0))
        end = float(f.get("end", 0))
        entries.append((i, text, start, end))
    with out_srt.open("w", encoding="utf-8") as w:
        for i, text, start, end in entries:
            w.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{text}\n\n")


def fmt_srt(t: float) -> str:
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int((t - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".txt or .epub input")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--language", default="vi", help="language code (vi/en/zh)")
    ap.add_argument("--speaker-wav", default=None, help="optional 2-10s WAV to condition timbre")
    ap.add_argument("--word-srt", action="store_true", help="also export words.srt from alignment JSON")
    ap.add_argument("--no-gpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading input…")
    full_text = extract_text(in_path)

    print("[2/4] Chunking…")
    pieces = split_sentences(full_text, max_chars=1200)

    audio_wav = out_dir / "audio.wav"
    print("[3/4] Synthesizing TTS…")
    synthesize_xtts(pieces, audio_wav, language=args.language, speaker_wav=args.speaker_wav, use_gpu=not args.no_gpu)

    print("[4/4] Forced alignment (words)…")
    words_json = out_dir / "word_timing.json"
    forced_align_words(audio_wav, full_text, words_json, language=args.language)

    print("[SRT] Writing sentence-level SRT…")
    sentences_srt = out_dir / "sentences.srt"
    sentences_to_srt(pieces, audio_wav, sentences_srt)

    if args.word_srt:
        print("[SRT] Writing word-level SRT…")
        words_srt = out_dir / "words.srt"
        words_json_to_srt(words_json, words_srt)

    print("Done.")


if __name__ == "__main__":
    main()
