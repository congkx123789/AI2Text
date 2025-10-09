
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple
import os
import io
import tempfile
import pathlib
import json

# Optional backends will be loaded lazily
_faster_whisper = None
_vosk = None
_pydub = None
_pytesseract = None
_easyocr = None
_PIL_Image = None

# ---------- Config ----------
DEFAULT_STT = os.getenv("DEFAULT_STT", "whisper")  # "whisper" or "vosk"
DEFAULT_OCR = os.getenv("DEFAULT_OCR", "tesseract")  # "tesseract" or "easyocr"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
VOSK_MODEL_DIR = os.getenv("VOSK_MODEL_DIR")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# ---------- App ----------
app = FastAPI(title="AI2Text Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # CRA/Next
        "http://127.0.0.1:3000",
        "*",  # relax during local dev; tighten in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ConvertResponse(BaseModel):
    text: str
    meta: Optional[dict] = None

# ---------- Utils ----------
AUDIO_EXTS = {"wav","mp3","m4a","flac","ogg","aac","wma","webm","mp4"}
IMAGE_EXTS = {"png","jpg","jpeg","bmp","tif","tiff","webp","pbm","ppm"}

def sniff_kind(filename: str) -> str:
    ext = pathlib.Path(filename).suffix.lower().lstrip(".")
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in IMAGE_EXTS:
        return "image"
    return ""

# ---------- Audio (STT) ----------

def ensure_pydub():
    global _pydub
    if _pydub is None:
        from pydub import AudioSegment
        _pydub = AudioSegment
    return _pydub


def stt_with_whisper(wav_path: str, language_hint: Optional[str]) -> Tuple[str, dict]:
    global _faster_whisper
    if _faster_whisper is None:
        from faster_whisper import WhisperModel
        _faster_whisper = WhisperModel(WHISPER_MODEL, device="cpu")  # use device="cuda" if you have GPU
    # faster-whisper supports many formats directly; we'll pass the path.
    segments, info = _faster_whisper.transcribe(
        wav_path,
        language=None if not language_hint or language_hint == "auto" else language_hint,
        vad_filter=True,
        beam_size=5,
    )
    texts = []
    seg_meta = []
    for s in segments:
        texts.append(s.text)
        seg_meta.append({
            "start": s.start,
            "end": s.end,
            "prob": getattr(s, "avg_logprob", None)
        })
    return "".join(texts).strip(), {
        "backend": "whisper",
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "segments": seg_meta,
        "model": WHISPER_MODEL,
    }


def stt_with_vosk(wav_path: str, language_hint: Optional[str]) -> Tuple[str, dict]:
    global _vosk
    if _vosk is None:
        if not VOSK_MODEL_DIR or not os.path.isdir(VOSK_MODEL_DIR):
            raise RuntimeError("VOSK_MODEL_DIR is not set or invalid.")
        import vosk
        _vosk = vosk
    import wave, json
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000,16000,32000,44100,48000):
        wf.close()
        raise RuntimeError("Vosk expects mono PCM WAV; use convert_to_wav_mono16k first.")
    model = _vosk.Model(VOSK_MODEL_DIR)
    rec = _vosk.KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    wf.close()
    full_text = " ".join((r.get("text", "") for r in results)).strip()
    return full_text, {"backend": "vosk", "results": results}


def convert_to_wav_mono16k(src_path: str) -> str:
    AudioSegment = ensure_pydub()
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    out_path = src_path + ".mono16k.wav"
    audio.export(out_path, format="wav")
    return out_path

# ---------- Image (OCR) ----------

def ocr_with_tesseract(image_bytes: bytes, language_hint: Optional[str]) -> Tuple[str, dict]:
    global _pytesseract, _PIL_Image
    if _pytesseract is None:
        import pytesseract
        from PIL import Image
        _pytesseract = pytesseract
        _PIL_Image = Image
        if TESSERACT_CMD:
            _pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img = _PIL_Image.open(io.BytesIO(image_bytes)).convert("RGB")
    lang = None if not language_hint or language_hint == "auto" else language_hint
    text = _pytesseract.image_to_string(img, lang=lang)
    return text.strip(), {"backend": "tesseract", "lang": lang}


def ocr_with_easyocr(image_bytes: bytes, language_hint: Optional[str]) -> Tuple[str, dict]:
    global _easyocr
    if _easyocr is None:
        import easyocr
        _easyocr = easyocr
    langs = ["en"] if not language_hint or language_hint == "auto" else [language_hint]
    reader = _easyocr.Reader(langs, gpu=False)
    # easyocr expects a path or ndarray; we'll give ndarray
    import numpy as np
    from PIL import Image
    arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=True)
    # results: list of (bbox, text, conf)
    lines = [r[1] for r in results]
    return "\n".join(lines).strip(), {"backend": "easyocr", "lang": langs, "raw": [str(r[0]) for r in results]}

# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/convert", response_model=ConvertResponse)
async def api_convert(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    stt_backend: str = Form("") ,
    ocr_backend: str = Form(""),
    language: str = Form("")
):
    # Save upload to temp file
    suffix = pathlib.Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw_path = tmp.name
        content = await file.read()
        tmp.write(content)
    kind = mode if mode in {"audio","image"} else sniff_kind(file.filename or "")
    if not kind:
        # fallback based on MIME
        ct = file.content_type or ""
        if ct.startswith("audio/"): kind = "audio"
        elif ct.startswith("image/"): kind = "image"
    if not kind:
        os.unlink(raw_path)
        raise HTTPException(400, detail="Cannot determine file type; set mode=audio|image explicitly.")

    meta = {"filename": file.filename, "size": len(content), "kind": kind}

    try:
        if kind == "audio":
            backend = (stt_backend or DEFAULT_STT).lower()
            # Convert to mono16k wav for safety, then pass path to backend
            wav16_path = convert_to_wav_mono16k(raw_path)
            if backend == "whisper":
                text, bmeta = stt_with_whisper(wav16_path, language or None)
            elif backend == "vosk":
                text, bmeta = stt_with_vosk(wav16_path, language or None)
            else:
                raise HTTPException(400, detail=f"Unsupported stt_backend: {backend}")
            meta.update(bmeta)
            return ConvertResponse(text=text, meta=meta)

        elif kind == "image":
            backend = (ocr_backend or DEFAULT_OCR).lower()
            if backend == "tesseract":
                text, bmeta = ocr_with_tesseract(content, language or None)
            elif backend == "easyocr":
                text, bmeta = ocr_with_easyocr(content, language or None)
            else:
                raise HTTPException(400, detail=f"Unsupported ocr_backend: {backend}")
            meta.update(bmeta)
            return ConvertResponse(text=text, meta=meta)

        else:
            raise HTTPException(400, detail=f"Unsupported mode/kind: {kind}")

    finally:
        # cleanup temp files
        try:
            os.unlink(raw_path)
        except Exception:
            pass
        try:
            extra = raw_path + ".mono16k.wav"
            if os.path.exists(extra):
                os.unlink(extra)
        except Exception:
            pass
