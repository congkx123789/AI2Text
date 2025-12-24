#!/usr/bin/env python3
"""
Demo script for ASR model inference.
Usage: 
    python3 demo_inference.py <audio_file> [language]
    python3 demo_inference.py --audio <audio_file> [--language <lang_code>]
    
Examples:
    python3 demo_inference.py audio.wav vi
    python3 demo_inference.py audio.wav --language en
    python3 demo_inference.py --audio audio.wav --language vi
"""
import argparse
import json
import torch
import torchaudio
from pathlib import Path
from model_code.model import CRNNCTC
from model_code.features import wav_to_logmelspec, ensure_mono16k
from model_code.decode import greedy_decode
from model_code.language_detection import clean_text_with_language, get_language_info, SUPPORTED_LANGUAGES

def load_model(checkpoint_path, vocab_path, device='cpu'):
    """Load model and vocabulary."""
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    itos = {i: c for i, c in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)}")
    
    print(f"Loading model from {checkpoint_path}...")
    model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', None)
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    return model, itos

def transcribe_audio(model, itos, audio_path, device='cpu', detect_lang=False, expected_lang=None):
    """Transcribe audio file to text.
    
    Args:
        model: The ASR model
        itos: Index to string mapping for vocabulary
        audio_path: Path to audio file
        device: Device to run inference on
        detect_lang: Whether to detect language from audio/text
        expected_lang: Expected language code (vi, en) - used for validation
    """
    print(f"Loading audio: {audio_path}")
    
    if expected_lang:
        lang_name = SUPPORTED_LANGUAGES.get(expected_lang, expected_lang)
        print(f"Expected language: {lang_name} ({expected_lang})")
    
    # Load audio (use soundfile backend for compatibility)
    try:
        import soundfile as sf
        data, sample_rate = sf.read(str(audio_path))
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)  # (samples, channels) -> (channels, samples)
    except ImportError:
        # Fallback to torchaudio with explicit backend
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path), backend="soundfile")
        except Exception:
            waveform, sample_rate = torchaudio.load(str(audio_path))
    
    # Preprocess: ensure mono 16kHz
    waveform, sr = ensure_mono16k(waveform, sample_rate)
    
    # Convert to log mel spectrogram
    features = wav_to_logmelspec(waveform, sr)
    
    # Add batch dimension: (1, T, F)
    features = features.unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[1]], device=device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        logits, _ = model(features, lengths)
    
    # Decode
    text = greedy_decode(logits, itos)[0]
    
    # Language detection from text (AFTER transcription)
    lang_info = None
    if detect_lang or expected_lang:
        lang_info = get_language_info(text, method='auto')
        if expected_lang:
            # Validate detected language matches expected
            detected_lang = lang_info.get('language_code', 'unknown')
            if detected_lang != expected_lang:
                print(f"⚠️  Warning: Expected language '{expected_lang}' but detected '{detected_lang}'")
            else:
                print(f"✓ Language matches expected: {expected_lang}")
    
    return text, lang_info

def main():
    parser = argparse.ArgumentParser(
        description="ASR Model Demo - Transcribe audio to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage with positional arguments
  python3 demo_inference.py audio.wav vi
  python3 demo_inference.py audio.wav en
  
  # Using flags
  python3 demo_inference.py --audio audio.wav --language vi
  python3 demo_inference.py --audio audio.wav --language en --device cuda
  
  # With language detection
  python3 demo_inference.py audio.wav --detect_language
        """
    )
    
    # Positional arguments for simpler usage
    parser.add_argument(
        "audio_positional",
        nargs="?",
        type=str,
        help="Path to audio file (positional argument)"
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        type=str,
        choices=["vi", "en"],
        help="Language code: vi (Vietnamese) or en (English) - positional argument"
    )
    
    # Optional arguments
    parser.add_argument(
        "--audio", 
        type=str,
        help="Path to audio file (wav, mp3, flac, etc.)"
    )
    parser.add_argument(
        "--language",
        "--lang",
        type=str,
        choices=["vi", "en"],
        help="Language code: vi (Vietnamese) or en (English). Helps model perform better by validating output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_epoch_12.pt",
        help="Path to model checkpoint (default: checkpoint_epoch_12.pt)"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="vocab.json",
        help="Path to vocabulary file (default: vocab.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto)"
    )
    parser.add_argument(
        "--detect_language",
        action="store_true",
        help="Detect language from transcribed text (default: False)"
    )
    
    args = parser.parse_args()
    
    # Determine audio path (positional or --audio flag)
    audio_path = args.audio_positional or args.audio
    if not audio_path:
        parser.error("Audio file path is required. Use: python3 demo_inference.py <audio_file> [language] or --audio <audio_file>")
    
    # Determine language (positional or --language flag)
    language = args.language_positional or args.language
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}\n")
    
    # Get absolute paths
    demo_dir = Path(__file__).parent
    checkpoint_path = demo_dir / args.checkpoint
    vocab_path = demo_dir / args.vocab
    audio_path = Path(audio_path)
    
    # Check files exist
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    if not vocab_path.exists():
        print(f"Error: Vocabulary not found at {vocab_path}")
        return
    
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Load model
    model, itos = load_model(str(checkpoint_path), str(vocab_path), device)
    
    # Transcribe
    try:
        # Enable language detection if language is specified or --detect_language flag is set
        detect_lang = args.detect_language or (language is not None)
        text, lang_info = transcribe_audio(
            model, itos, str(audio_path), device, 
            detect_lang=detect_lang, 
            expected_lang=language
        )
        
        # Clean text (remove language tags if present)
        clean_text, _ = clean_text_with_language(text, remove_tag=True)
        
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULT")
        print("="*60)
        print(f"Audio file: {audio_path}")
        if language:
            print(f"Specified language: {SUPPORTED_LANGUAGES.get(language, language)} ({language})")
        print(f"Transcribed text: {text}")
        
        if lang_info:
            print(f"\nLanguage Detection:")
            print(f"  Language: {lang_info['language_name']} ({lang_info['language_code']})")
            print(f"  Confidence: {lang_info['confidence']:.2%}")
            print(f"  Method: {lang_info['method']}")
            print(f"  Source: {lang_info['source']}")
            
            if language and lang_info['language_code'] != language:
                print(f"  ⚠️  Warning: Detected language '{lang_info['language_code']}' differs from specified '{language}'")
        
        if clean_text != text:
            print(f"\nCleaned text (no tags): {clean_text}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

