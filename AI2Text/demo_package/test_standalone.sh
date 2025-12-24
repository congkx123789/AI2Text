#!/bin/bash
# Quick test script to verify demo_package can run standalone

set -e

echo "🧪 Testing Standalone Demo Package"
echo "=================================="
echo ""

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMO_DIR"

# Test 1: Check required files
echo "1️⃣  Checking required files..."
REQUIRED_FILES=(
    "best_model.pt"
    "models/tokenizer_vi_en_3500.model"
    "models/tokenizer_vi_en_3500.vocab"
    "configs/default.yaml"
    "run_demo.py"
    "requirements.txt"
    "ai2text_src/models/asr_base.py"
    "ai2text_src/preprocessing/audio_processing.py"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "   ❌ Some required files are missing!"
    exit 1
fi

# Test 2: Test imports
echo ""
echo "2️⃣  Testing Python imports..."
if python3 -c "import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'ai2text_src'); from models.asr_base import ASRModel; from preprocessing.audio_processing import AudioProcessor; from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer; print('   ✅ All imports successful')" 2>&1; then
    echo ""
else
    echo "   ❌ Import failed!"
    exit 1
fi

# Test 3: Test script help
echo ""
echo "3️⃣  Testing run_demo.py help..."
if python3 run_demo.py --help >/dev/null 2>&1; then
    echo "   ✅ Script help works"
else
    echo "   ❌ Script help failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ All basic tests passed!"
echo ""
echo "📝 To transcribe audio:"
echo "   python3 run_demo.py audio.wav vi"
echo "   python3 run_demo.py audio.wav en"
echo "   python3 run_demo.py audio.wav    # auto-detect"

