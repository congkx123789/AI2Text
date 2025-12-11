"""
Convert Whisper HuggingFace model sang CTranslate2 format cho faster-whisper

faster-whisper chỉ đọc được CTranslate2 format, không đọc được HuggingFace format.
Script này sẽ convert model HuggingFace sang CTranslate2 để sử dụng với faster-whisper.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def check_ct2_installed():
    """Check if ctranslate2 is installed"""
    try:
        import ctranslate2
        return True
    except ImportError:
        return False


def convert_whisper_to_ct2(
    model_path: str,
    output_path: str,
    quantization: str = "float16"
):
    """
    Convert Whisper HuggingFace model sang CTranslate2 format
    
    Args:
        model_path: Path to HuggingFace Whisper model
        output_path: Output directory for CTranslate2 model
        quantization: Quantization type (float16, int8_float16, int8)
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path not found: {model_path}")
    
    if not check_ct2_installed():
        print("❌ ctranslate2 not installed!")
        print("📦 Installing ctranslate2...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ctranslate2"])
        print("✅ ctranslate2 installed")
    
    print("=" * 80)
    print("🔄 CONVERT WHISPER HUGGINGFACE → CT2 FORMAT")
    print("=" * 80)
    print(f"Input (HuggingFace): {model_path}")
    print(f"Output (CTranslate2): {output_path}")
    print(f"Quantization: {quantization}")
    print("=" * 80)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Remove output directory if exists (to avoid conflicts)
    if output_path.exists():
        print(f"🗑️  Removing existing output directory: {output_path}")
        import shutil
        shutil.rmtree(output_path)
    
    # Build command - ctranslate2 will auto-copy necessary files
    cmd = [
        "ct2-transformers-converter",
        "--model", str(model_path),
        "--output_dir", str(output_path),
        "--quantization", quantization
    ]
    
    # Optionally copy tokenizer files (ctranslate2 usually handles this automatically)
    # Only add if files exist and we want explicit control
    if (model_path / "tokenizer.json").exists() and (model_path / "preprocessor_config.json").exists():
        # Use separate --copy_files for each file (if supported)
        # Otherwise, ctranslate2 should auto-copy these
        pass
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        
        print()
        print("=" * 80)
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 CTranslate2 model saved to: {output_path}")
        print()
        print("📝 Next steps:")
        print(f"   1. Update .env: ASR_MODEL={output_path}")
        print("   2. Restart API server")
        print("   3. faster-whisper will now use the converted model")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("❌ CONVERSION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   - Make sure ctranslate2 is installed: pip install ctranslate2")
        print("   - Check if model path is correct")
        print("   - Try different quantization (float16, int8_float16, int8)")
        return 1
    except FileNotFoundError:
        print()
        print("=" * 80)
        print("❌ ctranslate2 NOT FOUND")
        print("=" * 80)
        print("ct2-transformers-converter command not found!")
        print()
        print("📦 Install ctranslate2:")
        print("   pip install ctranslate2")
        print()
        print("Or install from source:")
        print("   pip install ctranslate2 --upgrade")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper HuggingFace model to CTranslate2 format for faster-whisper"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to HuggingFace Whisper model (e.g., models/finetuned/whisper-mixed)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for CTranslate2 model (e.g., models/final/whisper-vi-en-ct2)"
    )
    parser.add_argument(
        "--quantization",
        default="float16",
        choices=["float16", "int8_float16", "int8"],
        help="Quantization type (default: float16 for GPU, int8_float16 for CPU)"
    )
    
    args = parser.parse_args()
    
    return convert_whisper_to_ct2(
        model_path=args.model,
        output_path=args.output,
        quantization=args.quantization
    )


if __name__ == "__main__":
    exit(main())

