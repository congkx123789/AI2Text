# src/server.py
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, tempfile

from ai2text.audio.inference import transcribe
from ai2text.image.inference import extract_text

# NEW: NLP modules
from ai2text.nlp.sentiment import SentimentService
from ai2text.nlp.translate import TranslatorService
from ai2text.nlp.summarize import SummarizeService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load NLP services once ---
sentiment_service = SentimentService()      # tries to load LSTM/logreg model
translator_service = TranslatorService()    # loads seq2seq model if present; else dummy fallback
summarize_service = SummarizeService()      # loads transformer summarizer

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    stt_backend: str = Form(None),
    ocr_backend: str = Form(None),
    language: str = Form(""),
):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    data = await file.read()

    audio_exts = {".wav",".mp3",".m4a",".flac",".ogg",".aac",".wma",".webm",".mp4"}
    image_exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".pbm",".ppm"}

    if mode == "auto":
        if suffix in audio_exts: mode = "audio"
        elif suffix in image_exts: mode = "image"
        else:
            return {"error": f"Unknown file type: {suffix}"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if mode == "audio":
            text = transcribe(tmp_path, backend=stt_backend)
        else:
            text = extract_text(tmp_path, backend=ocr_backend)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    return {"text": text, "meta": {"filename": file.filename, "mode": mode}}

# ---------- NLP endpoints ----------
class TextIn(BaseModel):
    text: str

@app.post("/api/sentiment")
def api_sentiment(inp: TextIn):
    label, score = sentiment_service.predict(inp.text)
    return {"sentiment": label, "confidence": score}

@app.post("/api/translate")
def api_translate(inp: TextIn):
    out = translator_service.translate_en_vi(inp.text)
    return {"translation": out}

@app.post("/api/summarize")
def api_summarize(inp: TextIn):
    out = summarize_service.summarize(inp.text)
    return {"summary": out}
