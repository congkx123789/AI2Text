#!/usr/bin/env python3
"""
Offline TTS with word-level timing (Coqui XTTS v2 + aeneas)

Features
- Read .txt or .epub (with fallback for odd EPUBs)
- Optional chapter-range extract: --start-chap / --end-chap
- Stream-synthesize single full WAV (no huge RAM usage)
- Forced alignment (aeneas) to per-word JSON + SRTs

Runtime deps (versions known to work with TTS==0.22.0):
  pip install "transformers==4.38.2" "tokenizers==0.15.2"
  # Torch CPU:
  pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
  # or Torch CUDA 11.8 for NVIDIA (e.g., MX330):
  pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118

Python deps:
  pip install TTS==0.22.0 ebooklib beautifulsoup4 tqdm aeneas soundfile

System deps for aeneas: ffmpeg, espeak/espeak-ng (Windows is trickier; WSL recommended)
"""

import argparse
from builtins import Exception
from pathlib import Path
import re
import json
from typing import List, Optional

from tqdm import tqdm
from bs4 import BeautifulSoup
from ebooklib import epub

# aeneas
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


def _read_text_file_best_effort(p: Path) -> str:
    # Try utf-8 first, then a few common fallbacks.
    encodings = ["utf-8", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    # Last resort: ignore errors
    return p.read_text(encoding="utf-8", errors="ignore")


def extract_text(input_path: Path, start_chap: Optional[str] = None, end_chap: Optional[str] = None) -> str:
    """
    Read .epub (proper container) or 'zip-ish' epubs that lack META-INF/container.xml.
    Falls back to scanning all HTML/XHTML files inside the archive.

    If start_chap/end_chap are provided, include from the first item containing start_chap
    up to and including the first item containing end_chap. Matching is case-insensitive.
    """
    def _should_start(txt: str) -> bool:
        return (start_chap is None) or (start_chap.lower() in txt.lower())

    def _should_stop(txt: str) -> bool:
        return (end_chap is not None) and (end_chap.lower() in txt.lower())

    def _collect_from_items(html_items: List[str]) -> str:
        chunks: List[str] = []
        capturing = (start_chap is None)
        for html in html_items:
            txt = clean_text_from_html(html)
            if not txt:
                continue
            if not capturing and _should_start(txt):
                capturing = True
            if capturing:
                chunks.append(txt)
                if _should_stop(txt):
                    break
        return "\n\n".join(chunks) if chunks else ""

    if input_path.suffix.lower() == ".epub":
        # Try normal EPUB via EbookLib
        try:
            book = epub.read_epub(str(input_path))
            html_items: List[str] = []
            # Prefer official constant if available; older ebooklib used type id 9 for docs
            ITEM_DOCUMENT = getattr(epub, "ITEM_DOCUMENT", 9)
            for item in book.get_items():
                if getattr(item, "get_type", None) and item.get_type() == ITEM_DOCUMENT:
                    html_items.append(item.get_content().decode(errors="ignore"))
                elif getattr(item, "media_type", "") in ("application/xhtml+xml", "text/html"):
                    html_items.append(item.get_content().decode(errors="ignore"))
            text = _collect_from_items(html_items)
            if text:
                return text
        except Exception:
            # Fall through to zip scan
            pass

        # Fallback: open as a plain zip and read *.html/*.xhtml
        import zipfile
        html_items = []
        with zipfile.ZipFile(str(input_path), "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".html", ".htm", ".xhtml"))]
            names.sort()
            for name in names:
                try:
                    html_items.append(zf.read(name).decode(errors="ignore"))
                except Exception:
                    continue
        text = _collect_from_items(html_items)
        if text:
            return text
        raise RuntimeError("EPUB has no HTML/XHTML entries (or no matches in requested range).")

    # Plain text input
    raw = _read_text_file_best_effort(input_path)
    if start_chap is None and end_chap is None:
        return raw
    start_pos = 0
    if start_chap:
        i = raw.lower().find(start_chap.lower())
        if i >= 0:
            start_pos = i
    end_pos = len(raw)
    if end_chap:
        j = raw.lower().find(end_chap.lower(), start_pos)
        if j >= 0:
            end_pos = j + len(end_chap)
    return raw[start_pos:end_pos]


def split_sentences(text: str, max_chars: int = 1500) -> List[str]:
    # Simple sentence-ish split then pack to ~max_chars chunks
    # Guard against gigantic whitespace runs first:
    text = WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    initial = re.split(SENTENCE_SPLIT_RE, text)
    initial = [s.strip() for s in initial if s and s.strip()]
    pieces: List[str] = []
    cur = ""
    for s in initial:
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}"
    if cur:
        pieces.append(cur)
    return pieces


def _validate_speaker_wav(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Speaker WAV not found: {path}")
    import soundfile as sf
    data, sr = sf.read(str(path), always_2d=False)
    dur = len(data) / float(sr)
    if dur < 2.0 or dur > 10.0:
        raise ValueError(f"speaker_wav must be 2–10 s; got ~{dur:.2f}s")
    # XTTS expects mono; resampling will be done internally if needed, but mono is best.
    if data.ndim > 1:
        raise ValueError("speaker_wav must be mono (1 channel)")


def synthesize_xtts(pieces: List[str], out_wav: Path, language: str = "vi", speaker_wav: str = None, use_gpu: bool = True):
    # Import here to avoid loading heavy libs when only extracting text
    from TTS.api import TTS
    import soundfile as sf
    import numpy as np

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name)

    # Put model on GPU if available/desired (optional)
    try:
        import torch
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        # Some TTS versions expose .to(); if missing, ignore.
        if hasattr(tts, "to"):
            tts.to(device)
    except Exception:
        pass

    # Determine proper sample rate from the synthesizer (don’t hardcode 24k)
    sr = None
    try:
        # Works for TTS==0.22.0
        sr = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
        if sr is None:
            # Fallback path in some versions
            sr = getattr(getattr(tts, "synthesizer", None), "tts_config", None)
            sr = getattr(getattr(sr, "audio", None), "sample_rate", 24000)
    except Exception:
        sr = 24000
    if not isinstance(sr, int):
        sr = 24000

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Validate speaker_wav early for clearer error messages
    if not speaker_wav:
        raise RuntimeError(
            "XTTS v2 is multi-speaker. Provide --speaker-wav path_to_2-10s_mono_24kHz_wav."
        )
    _validate_speaker_wav(Path(speaker_wav))

    # Stream to disk: one single WAV
    # libsndfile will convert float -> PCM_16 as needed.
    with sf.SoundFile(str(out_wav), mode="w", samplerate=sr, channels=1, subtype="PCM_16") as f:
        for s in tqdm(pieces, desc="Synth", dynamic_ncols=True):
            try:
                wav = tts.tts(text=s, speaker_wav=speaker_wav, language=language)
            except Exception as e:
                raise RuntimeError(
                    f"XTTS synthesis failed. If using GPU, try --no-gpu. "
                    f"Also ensure speaker_wav is mono 24kHz 16-bit PCM (2–10 s)."
                ) from e

            # Coqui returns float np array in [-1,1], shape (N,) or (N,1)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            f.write(wav)
            # Short pause between chunks (~200 ms)
            import math
            pause_len = int(math.ceil(0.2 * sr))
            if pause_len > 0:
                f.write(np.zeros(pause_len, dtype=np.float32))


def forced_align_words(audio_wav: Path, full_text: str, out_json: Path, language: str = "vi"):
    """Use aeneas to align words. We write each word on its own line to get word-level fragments."""
    tokens = tokenize_words_vi(full_text)
    temp_txt = out_json.with_suffix(".words.txt")
    temp_txt.write_text("\n".join(tokens), encoding="utf-8")

    # Map our simple CLI language to aeneas language tags
    lang_map = {"vi": "vie", "en": "eng", "zh": "cmn"}
    cfg_lang = lang_map.get(language, "vie")

    # Keep defaults simple; users can tweak later if needed
    config_string = f"task_language={cfg_lang}|is_text_type=plain|os_task_file_format=json"

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = str(audio_wav)
    task.text_file_path_absolute = str(temp_txt)
    task.sync_map_file_path_absolute = str(out_json)

    ExecuteTask(task).execute()

    try:
        temp_txt.unlink(missing_ok=True)
    except Exception:
        pass


def tokenize_words_vi(text: str) -> List[str]:
    # Simple tokenization: split on whitespace/punct, keep Vietnamese letters (basic range)
    # Swap in underthesea or vncorenlp for smarter splitting if desired.
    return re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)


def sentences_to_srt(pieces: List[str], audio_wav: Path, out_srt: Path):
    import soundfile as sf
    data, sr = sf.read(str(audio_wav))
    total_len = len(data) / float(sr) if sr else 0.0
    lengths = [len(p) for p in pieces]
    total_chars = sum(lengths)
    t = 0.0
    idx = 1
    lines = []
    for L, s in zip(lengths, pieces):
        seg = total_len * (L / total_chars) if total_chars > 0 else 0.0
        start, end = t, t + seg
        lines.append(f"{idx}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{s}\n\n")
        idx += 1
        t = end
    out_srt.write_text("".join(lines), encoding="utf-8")


def words_json_to_srt(words_json: Path, out_srt: Path):
    obj = json.loads(words_json.read_text(encoding="utf-8"))
    fragments = obj.get("fragments", [])
    with out_srt.open("w", encoding="utf-8") as w:
        for i, f in enumerate(fragments, 1):
            lines = f.get("lines", []) or [""]
            text = lines[0]
            try:
                start = float(f.get("begin", 0))
                end = float(f.get("end", 0))
            except Exception:
                start, end = 0.0, 0.0
            w.write(f"{i}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{text}\n\n")


def fmt_srt(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".txt or .epub input")
    ap.add_argument("--start-chap", default=None, help="Start when this text appears (case-insensitive)")
    ap.add_argument("--end-chap", default=None, help="Stop after the first section that contains this text (case-insensitive)")

    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--language", default="vi", help="language code (vi/en/zh)")
    ap.add_argument("--speaker-wav", default=None, help="2–10s WAV to condition timbre (mono, 24kHz, 16-bit)")
    ap.add_argument("--word-srt", action="store_true", help="also export words.srt from alignment JSON")
    ap.add_argument("--no-gpu", action="store_true", help="force CPU even if CUDA available")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading input…")
    full_text = extract_text(in_path, start_chap=args.start_chap, end_chap=args.end_chap)

    if not full_text.strip():
        raise SystemExit("No text found in the requested range.")

    print("[2/4] Chunking…")
    pieces = split_sentences(full_text, max_chars=1200)
    if not pieces:
        raise SystemExit("Nothing to synthesize after chunking.")

    audio_wav = out_dir / "audio.wav"
    print("[3/4] Synthesizing TTS…")
    synthesize_xtts(
        pieces,
        audio_wav,
        language=args.language,
        speaker_wav=args.speaker_wav,
        use_gpu=not args.no_gpu,
    )

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
