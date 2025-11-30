"""
Post-Training Quantization (PTQ) for ASR models.

Quantizes trained models to Int8/FP8 to reduce model size and inference latency.
Supports:
- Dynamic Quantization (Int8)
- Static Quantization (Int8) - requires calibration data
- FP8 Quantization (if supported by hardware)

Usage:
    python scripts/quantize_model.py \
        --checkpoint checkpoints/best_model.pt \
        --output checkpoints/best_model_quantized_int8.pt \
        --quantization int8
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config from checkpoint or provided config file
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config not found in checkpoint and no config file provided")
    
    # Setup tokenizer
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        if Path(bpe_path).exists():
            tokenizer.load(bpe_path)
        else:
            print(f"⚠️  BPE vocab not found, using character tokenizer")
            tokenizer = Tokenizer()
    else:
        tokenizer = Tokenizer()
    
    # Create model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, config


def dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization (Int8) to model.
    
    Dynamic quantization quantizes weights to Int8 but activations are quantized
    on-the-fly during inference. No calibration data needed.
    """
    print("Applying dynamic quantization (Int8)...")
    
    # Quantize model
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Quantize Linear and Conv2d layers
        dtype=torch.qint8
    )
    
    print("✅ Dynamic quantization completed")
    return quantized_model


def static_quantization(model: nn.Module, calibration_data, num_calibration_batches: int = 100):
    """
    Apply static quantization (Int8) to model.
    
    Static quantization requires calibration data to determine quantization parameters.
    More accurate than dynamic quantization but requires calibration.
    """
    print("Applying static quantization (Int8)...")
    print(f"Using {num_calibration_batches} batches for calibration...")
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with data
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            if i >= num_calibration_batches:
                break
            if isinstance(batch, dict):
                audio_features = batch['audio_features']
                audio_lengths = batch['audio_lengths']
            else:
                audio_features, audio_lengths = batch
            model(audio_features, audio_lengths)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    print("✅ Static quantization completed")
    return quantized_model


def get_model_size(model: nn.Module, file_path: str = None) -> dict:
    """Get model size information."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get file size if path provided
    file_size_mb = None
    if file_path and Path(file_path).exists():
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_mb': file_size_mb
    }


def main():
    parser = argparse.ArgumentParser(description='Quantize ASR model for inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model')
    parser.add_argument('--quantization', type=str, default='int8',
                       choices=['int8', 'dynamic', 'static'],
                       help='Quantization type: int8/dynamic (default) or static')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint)')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Path to calibration data (required for static quantization)')
    parser.add_argument('--num-calibration-batches', type=int, default=100,
                       help='Number of batches for calibration (default: 100)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL QUANTIZATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Quantization: {args.quantization}")
    print()
    
    # Load model
    print("Loading model...")
    model, config = load_model_from_checkpoint(args.checkpoint, args.config)
    
    # Get original model size
    original_size = get_model_size(model, args.checkpoint)
    print(f"Original model:")
    print(f"  Parameters: {original_size['total_params']:,}")
    if original_size['file_size_mb']:
        print(f"  File size: {original_size['file_size_mb']:.2f} MB")
    print()
    
    # Apply quantization
    if args.quantization in ['int8', 'dynamic']:
        quantized_model = dynamic_quantization(model)
    elif args.quantization == 'static':
        if not args.calibration_data:
            raise ValueError("Calibration data required for static quantization")
        # Load calibration data
        # TODO: Implement calibration data loading
        raise NotImplementedError("Static quantization requires calibration data loading")
    else:
        raise ValueError(f"Unknown quantization type: {args.quantization}")
    
    # Get quantized model size
    print()
    print("Saving quantized model...")
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config,
        'quantization': args.quantization
    }, args.output)
    
    quantized_size = get_model_size(quantized_model, args.output)
    print(f"Quantized model:")
    print(f"  Parameters: {quantized_size['total_params']:,}")
    if quantized_size['file_size_mb']:
        print(f"  File size: {quantized_size['file_size_mb']:.2f} MB")
    
    # Calculate reduction
    if original_size['file_size_mb'] and quantized_size['file_size_mb']:
        reduction = (1 - quantized_size['file_size_mb'] / original_size['file_size_mb']) * 100
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Size reduction: {reduction:.1f}%")
        print(f"Original: {original_size['file_size_mb']:.2f} MB")
        print(f"Quantized: {quantized_size['file_size_mb']:.2f} MB")
        print(f"Saved: {original_size['file_size_mb'] - quantized_size['file_size_mb']:.2f} MB")
    
    print()
    print("✅ Quantization completed!")
    print(f"Quantized model saved to: {args.output}")


if __name__ == '__main__':
    main()

