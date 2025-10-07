# tests/test_image.py
from src.ai2text.image.inference import extract_text
from PIL import Image, ImageDraw

def test_ocr_runs_smoke(tmp_path):
    p = tmp_path / "hello.png"
    img = Image.new("RGB", (400, 120), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 40), "HELLO", fill="black")
    img.save(p)
    text = extract_text(str(p), backend="tesseract")
    assert isinstance(text, str)
