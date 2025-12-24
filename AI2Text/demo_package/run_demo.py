#!/usr/bin/env python3
"""
AI2Text Demo - Standalone inference script
Chạy transcribe audio mà không cần full project structure
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path for imports
DEMO_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DEMO_DIR))

# Standalone mode: prefer bundled source in demo_package/ai2text_src/
AI2TEXT_SRC = DEMO_DIR / "ai2text_src"
if AI2TEXT_SRC.exists():
    sys.path.insert(0, str(AI2TEXT_SRC))
    logger.info("✅ Using bundled modules from demo_package/ai2text_src (standalone mode)")
else:
    # Fallback: allow importing from parent project if user placed source there
    parent_project = DEMO_DIR.parent
    sys.path.insert(0, str(parent_project))
    logger.info("ℹ️ ai2text_src not found; trying to import modules from parent directory")

try:
    from models.asr_base import ASRModel
    from preprocessing.audio_processing import AudioProcessor
    from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
except Exception as e:
    logger.error("❌ Cannot import required modules (models/, preprocessing/).")
    logger.error("   Fix: ensure demo_package contains ai2text_src/ with those packages, or place them in the parent folder.")
    logger.error(f"   Import error: {e}")
    sys.exit(1)


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Get model parameters from config
    input_dim = config.get('n_mels', 80)
    d_model = config.get('d_model', 256)
    num_encoder_layers = config.get('num_encoder_layers', 14)
    num_decoder_layers = config.get('num_decoder_layers', 6)
    num_heads = config.get('num_heads', 8)
    d_ff = config.get('d_ff', 2048)
    dropout = config.get('dropout', 0.2)
    
    # Get vocab_size
    vocab_size = checkpoint.get('vocab_size', config.get('vocab_size', 3500))
    
    logger.info(f"Model config: d_model={d_model}, vocab_size={vocab_size}, "
                f"encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}")
    
    # Create model
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        use_gradient_checkpointing=False
    )
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("✅ Model weights loaded (strict mode)")
        except Exception as e:
            logger.warning(f"Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Model weights loaded (non-strict mode)")
    else:
        raise ValueError("No model weights found in checkpoint")
    
    model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer."""
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    logger.info(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    return tokenizer


def _resolve_demo_relative_path(demo_dir: Path, rel_path: str) -> Path:
    """
    Resolve a path used by the original project in a standalone demo_package.
    Preference order:
      1) demo_package/<rel_path>
      2) demo_package/ai2text_src/<rel_path> (bundled code assets)
    """
    p1 = demo_dir / rel_path
    if p1.exists():
        return p1
    p2 = demo_dir / "ai2text_src" / rel_path
    return p2


def transcribe_audio(model, tokenizer, audio_processor, audio_path: str, device: str = "cpu", language: str = None):
    """Transcribe audio file.
    
    Args:
        model: ASR model
        tokenizer: Tokenizer
        audio_processor: Audio processor
        audio_path: Path to audio file
        device: Device to use
        language: Language code ('vi' for Vietnamese, 'en' for English, None for auto)
    """
    logger.info(f"Processing audio: {audio_path}")
    if language:
        logger.info(f"Language specified: {language}")
    
    # Load and process audio
    audio_data, sample_rate = audio_processor.load_audio(audio_path)
    mel_spec = audio_processor.extract_mel_spectrogram(audio_data)
    
    # Prepare input: (n_mels, time) -> (1, time, n_mels)
    features = torch.from_numpy(mel_spec.T).unsqueeze(0).float().to(device)
    lengths = torch.tensor([features.size(1)]).to(device)
    
    # Prepare language IDs: 0 = Vietnamese, 1 = English
    language_ids = None
    if language:
        if language.lower() == 'vi':
            language_ids = torch.tensor([0]).to(device)
            logger.info("Using Vietnamese language embedding (language_id=0)")
        elif language.lower() == 'en':
            language_ids = torch.tensor([1]).to(device)
            logger.info("Using English language embedding (language_id=1)")
        else:
            logger.warning(f"Unknown language '{language}', using auto-detect (language_ids=None)")
    
    # Generate transcription
    sos_token_id = getattr(tokenizer, 'sos_token_id', 2)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 3)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    with torch.no_grad():
        generated = model.generate(
            features,
            lengths=lengths,
            language_ids=language_ids,
            max_len=512,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=1.0
        )
    
    # Decode tokens
    gen_seq = generated[0].cpu().tolist()
    decoded_tokens = []
    for token in gen_seq:
        if token == eos_token_id:
            break
        if token not in (sos_token_id, pad_token_id):
            decoded_tokens.append(token)
    
    text = tokenizer.decode(decoded_tokens)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="AI2Text Demo - Transcribe audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage with positional arguments
  python run_demo.py audio.wav
  python run_demo.py audio.wav vi
  python run_demo.py audio.wav en
  
  # With flags (backward compatible)
  python run_demo.py --audio audio.wav --language vi
  python run_demo.py --audio audio.wav --device cuda
        """
    )
    
    # Positional arguments (simpler usage)
    parser.add_argument("audio", nargs='?', default=None,
                       help="Path to audio file (WAV, MP3, etc.) - positional argument")
    parser.add_argument("language", nargs='?', default=None, choices=["vi", "en"],
                       help="Language code: 'vi' for Vietnamese, 'en' for English (optional positional argument)")
    
    # Optional flags (backward compatible)
    parser.add_argument("--audio", dest="audio_flag", default=None,
                       help="Path to audio file (alternative to positional argument)")
    parser.add_argument("--checkpoint", default="best_model.pt", 
                       help="Path to checkpoint file (default: best_model.pt)")
    parser.add_argument("--tokenizer", default="models/tokenizer_vi_en_3500.model",
                       help="Path to tokenizer model (default: models/tokenizer_vi_en_3500.model)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use (default: cpu)")
    parser.add_argument("--config", default="configs/default.yaml",
                       help="Path to config file (default: configs/default.yaml)")
    parser.add_argument("--language", dest="language_flag", default=None, choices=["vi", "en"],
                       help="Language code: 'vi' for Vietnamese, 'en' for English (alternative to positional argument)")
    
    args = parser.parse_args()
    
    # Resolve audio path: positional argument takes precedence over --audio flag
    audio_path_str = args.audio if args.audio else args.audio_flag
    if not audio_path_str:
        parser.error("Audio file is required. Use: python run_demo.py <audio_file> [language]")
    
    # Resolve language: positional argument takes precedence over --language flag
    language = args.language if args.language else args.language_flag
    
    # Resolve paths relative to demo directory (standalone-friendly)
    demo_dir = Path(__file__).parent.absolute()
    checkpoint_path = demo_dir / args.checkpoint
    tokenizer_path = _resolve_demo_relative_path(demo_dir, args.tokenizer)
    
    # Check files exist
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not tokenizer_path.exists():
        logger.error(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    if language:
        logger.info(f"Language: {language.upper()}")
    else:
        logger.info("Language: Auto-detect")
    
    # Load components
    try:
        model = load_model(str(checkpoint_path), device=args.device)
        tokenizer = load_tokenizer(str(tokenizer_path))
        
        # Load config for audio processor settings
        import yaml
        config_path = _resolve_demo_relative_path(demo_dir, args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            sample_rate = config.get('sample_rate', 16000)
            n_mels = config.get('n_mels', 80)
        else:
            logger.warning(f"Config not found, using defaults")
            sample_rate = 16000
            n_mels = 80
        
        audio_processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        
        # Transcribe
        logger.info("Starting transcription...")
        text = transcribe_audio(model, tokenizer, audio_processor, str(audio_path), 
                               device=args.device, language=language)
        
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULT:")
        if language:
            print(f"Language: {language.upper()}")
        else:
            print("Language: Auto-detect")
        print("="*60)
        print(text)
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error during transcription: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

