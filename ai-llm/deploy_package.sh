#!/bin/bash
# Script để tạo deployment package

echo "📦 Creating deployment package..."

# Tạo thư mục
mkdir -p whisper-deploy-package
cd whisper-deploy-package

# Copy model
echo "📂 Copying model..."
cp -r ../models/final/whisper-vi-en-ct2/ .

# Copy script
echo "📝 Copying script..."
cp ../run_whisper.py .

# Copy requirements
echo "📋 Copying requirements..."
cp ../requirements_whisper.txt requirements.txt

# Tạo README
cat > README.md << 'READMEEOF'
# Whisper Model Deployment Package

## Quick Start

1. Cài đặt dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Chạy model:
   ```bash
   python run_whisper.py audio.wav whisper-vi-en-ct2
   ```

## Files

- `whisper-vi-en-ct2/` - Model files
- `run_whisper.py` - Script chạy model
- `requirements.txt` - Dependencies

## Usage

```bash
python run_whisper.py <audio_file> [model_path]
```

Examples:
```bash
python run_whisper.py audio.wav
python run_whisper.py audio.wav whisper-vi-en-ct2
```
READMEEOF

cd ..

# Tạo archives
echo "📦 Creating archives..."

# Tạo tar.gz
echo "   Creating tar.gz..."
tar -czf whisper-deploy-package.tar.gz whisper-deploy-package/

# Tạo zip
echo "   Creating zip..."
zip -r whisper-deploy-package.zip whisper-deploy-package/ > /dev/null

echo ""
echo "✅ Packages created:"
echo "   📦 whisper-deploy-package.tar.gz ($(du -h whisper-deploy-package.tar.gz | cut -f1))"
echo "   📦 whisper-deploy-package.zip ($(du -h whisper-deploy-package.zip | cut -f1))"
echo ""
echo "📤 To deploy (tar.gz):"
echo "   scp whisper-deploy-package.tar.gz user@new-machine:/path/to/"
echo "   tar -xzf whisper-deploy-package.tar.gz"
echo ""
echo "📤 To deploy (zip):"
echo "   scp whisper-deploy-package.zip user@new-machine:/path/to/"
echo "   unzip whisper-deploy-package.zip"
echo ""
echo "📥 On new machine:"
echo "   cd whisper-deploy-package"
echo "   pip install -r requirements.txt"
echo "   python run_whisper.py audio.wav"
