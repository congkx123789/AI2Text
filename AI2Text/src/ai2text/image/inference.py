# src/ai2text/image/inference.py
from __future__ import annotations
import os
from typing import Optional
from pathlib import Path
from PIL import Image

from ..config import CONFIG

# ---------- Tesseract ----------
def _ocr_tesseract(image_path: str) -> str:
    import pytesseract
    if CONFIG.ocr.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = CONFIG.ocr.tesseract_cmd
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=CONFIG.ocr.lang)
    return text.strip()

# ---------- EasyOCR ----------
def _ocr_easyocr(image_path: str):
    import easyocr
    reader = easyocr.Reader(list(CONFIG.ocr.easyocr_langs), gpu=False)
    result = reader.readtext(image_path, detail=0, paragraph=True)
    # result is a list[str]; join with newlines
    return "\n".join([r.strip() for r in result if r.strip()])

def extract_text(image_path: str, backend: Optional[str] = None) -> str:
    be = (backend or CONFIG.ocr.backend).lower()
    if be == "tesseract":
        return _ocr_tesseract(image_path)
    if be == "easyocr":
        return _ocr_easyocr(image_path)
    raise ValueError(f"Unknown OCR backend: {be}")
