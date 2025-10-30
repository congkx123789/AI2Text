## make a new file 
#!/usr/bin/env bash
set -euo pipefail


if command -v apt >/dev/null 2>&1; then
sudo apt update -y
sudo apt install -y ffmpeg tesseract-ocr
elif command -v brew >/dev/null 2>&1; then
brew install ffmpeg tesseract
else
echo "Please install ffmpeg and tesseract manually for your OS."
fi