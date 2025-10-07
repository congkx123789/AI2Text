# src/ai2text/config.py
from dataclasses import dataclass

@dataclass
class STTConfig:
    backend: str = "vosk"  # "vosk" or "whisper"
    model: str = "small"   # whisper: tiny|base|small|medium|large
    vosk_model_dir: str = "models/vosk-small-en-us"  # put extracted model dir here
    language: str = "en"   # whisper lang code when forced (None = auto)

@dataclass
class OCRConfig:
    backend: str = "tesseract"  # "tesseract" or "easyocr"
    tesseract_cmd: str | None = None  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    lang: str = "eng"  # tesseract language
    easyocr_langs: tuple[str, ...] = ("en",)  # e.g. ("vi","en")

@dataclass
class AppConfig:
    stt: STTConfig = STTConfig()
    ocr: OCRConfig = OCRConfig()

CONFIG = AppConfig()
