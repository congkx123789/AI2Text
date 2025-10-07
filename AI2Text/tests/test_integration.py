# tests/test_integration.py
import subprocess, sys, os, shutil

def test_cli_smoke(tmp_path):
    # generate a fake image
    from PIL import Image, ImageDraw
    imgp = tmp_path / "hi.png"
    img = Image.new("RGB", (320, 120), "white")
    d = ImageDraw.Draw(img)
    d.text((10,50), "hi", fill="black")
    img.save(imgp)

    cmd = [sys.executable, "src/main.py", str(imgp), "--mode", "image", "--ocr-backend", "tesseract"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert isinstance(res.stdout, str)
