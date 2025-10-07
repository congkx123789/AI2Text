# src/ai2text/audio/inference.py
from __future__ import annotations
import os, tempfile, subprocess, sys
from pathlib import Path
from typing import Optional
import soundfile as sf

from ..config import CONFIG

def _ensure_wav_16k_mono(path: str) -> str:
    """Return a 16k mono WAV path (converts via ffmpeg if needed)."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            data, sr = sf.read(str(p), always_2d=False)
            if sr == 16000 and (data.ndim == 1):
                return str(p)
        except Exception:
            pass
    out = Path(tempfile.gettempdir()) / f"ai2text_16kmono_{p.stem}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", str(out)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(out)

# ---------- VOSK ----------
def _stt_vosk(audio_path: str) -> str:
    import vosk, json, wave
    model_dir = CONFIG.stt.vosk_model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Vosk model dir not found: {model_dir}. Download a model from https://alphacephei.com/vosk/models"
        )
    wav = _ensure_wav_16k_mono(audio_path)
    rec = vosk.KaldiRecognizer(vosk.Model(model_dir), 16000)
    with wave.open(wav, "rb") as wf:
        result_text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                j = json.loads(rec.Result())
                if "text" in j: result_text.append(j["text"])
        j = json.loads(rec.FinalResult())
        if "text" in j: result_text.append(j["text"])
    return " ".join(t for t in result_text if t).strip()

# ---------- WHISPER ----------
def _stt_whisper(audio_path: str) -> str:
    import whisper
    model_name = CONFIG.stt.model
    model = whisper.load_model(model_name)
    opts = {}
    if CONFIG.stt.language and CONFIG.stt.language.lower() != "auto":
        opts["language"] = CONFIG.stt.language
        opts["task"] = "transcribe"
    result = model.transcribe(audio_path, **opts)
    return (result.get("text") or "").strip()

def transcribe(audio_path: str, backend: Optional[str] = None) -> str:
    """
    Unified STT entry point.
    backend: "vosk" | "whisper" | None (use CONFIG).
    """
    be = (backend or CONFIG.stt.backend).lower()
    if be == "vosk":
        return _stt_vosk(audio_path)
    if be == "whisper":
        return _stt_whisper(audio_path)
    raise ValueError(f"Unknown STT backend: {be}")
