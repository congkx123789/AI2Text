"""
Script to find the maximum batch size that fits in GPU memory.
Automatically tests increasing batch sizes until OOM occurs.

Usage:
    python scripts/find_max_batch_size.py --config configs/librispeech_rtx5060ti.yaml
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from training.dataset import create_data_loaders
import pandas as pd


def find_max_batch_size(config_path: str, start_batch_size: int = 8, max_batch_size: int = 128):
    """
    Find the maximum batch size that fits in GPU memory.
    
    Args:
        config_path: Path to config YAML file
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test (safety limit)
    """
    print("=" * 60)
    print("BATCH SIZE OPTIMIZATION FOR RTX 5060 Ti")
    print("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Starting batch size: {start_batch_size}")
    print(f"Max batch size to test: {max_batch_size}")
    print()
    
    # Setup preprocessing
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
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
    
    # Setup mixed precision
    use_amp = config.get('use_amp', True)
    amp_dtype_str = config.get('amp_dtype', 'fp16').lower()
    if amp_dtype_str == 'bf16' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
        print("Using BF16 mixed precision")
    else:
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        print("Using FP16 mixed precision")
    
    model.to(device)
    
    # Setup optimizer and loss
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=tokenizer.blank_token_id, zero_infinity=True)
    
    # Create dummy data loader (we'll use small dataset for testing)
    # Try to load actual data, but if not available, create dummy data
    try:
        train_manifest = config.get('train_manifest')
        audio_base_dir = config.get('audio_base_dir', 'data/processed/librispeech_alignments')
        if train_manifest and Path(train_manifest).exists():
            df = pd.read_csv(train_manifest)
            # Use only first 100 samples for testing
            df = df.head(100)
            train_loader, _ = create_data_loaders(
                train_df=df,
                val_df=df.head(10),  # Dummy val
                audio_processor=audio_processor,
                tokenizer=tokenizer,
                batch_size=start_batch_size,
                num_workers=0,  # Single process for testing
                persistent_workers=False
            )
            print(f"✅ Using real data: {len(df)} samples")
        else:
            raise FileNotFoundError("Manifest not found")
    except Exception as e:
        print(f"⚠️  Could not load real data: {e}")
        print("Creating dummy data for testing...")
        # Create dummy dataset
        from torch.utils.data import Dataset, DataLoader
        
        class DummyDataset(Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create dummy audio features (typical shape: [seq_len, n_mels])
                seq_len = torch.randint(100, 500, (1,)).item()
                audio_features = torch.randn(seq_len, config.get('n_mels', 80))
                audio_length = torch.tensor(seq_len)
                
                # Create dummy text tokens
                text_len = torch.randint(10, 50, (1,)).item()
                text_tokens = torch.randint(0, len(tokenizer), (text_len,))
                text_length = torch.tensor(text_len)
                
                return {
                    'audio_features': audio_features,
                    'audio_lengths': audio_length,
                    'text_tokens': text_tokens,
                    'text_lengths': text_length
                }
        
        def dummy_collate_fn(batch):
            # Simple collate function
            audio_features = [item['audio_features'] for item in batch]
            audio_lengths = torch.tensor([item['audio_lengths'] for item in batch])
            text_tokens = [item['text_tokens'] for item in batch]
            text_lengths = torch.tensor([item['text_lengths'] for item in batch])
            
            # Pad audio features
            max_audio_len = max(f.shape[0] for f in audio_features)
            padded_audio = torch.zeros(len(batch), max_audio_len, audio_features[0].shape[1])
            for i, f in enumerate(audio_features):
                padded_audio[i, :f.shape[0]] = f
            
            # Pad text tokens
            max_text_len = max(t.shape[0] for t in text_tokens)
            padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
            for i, t in enumerate(text_tokens):
                padded_text[i, :t.shape[0]] = t
            
            return {
                'audio_features': padded_audio,
                'audio_lengths': audio_lengths,
                'text_tokens': padded_text,
                'text_lengths': text_lengths
            }
        
        dummy_dataset = DummyDataset(num_samples=100)
        train_loader = DataLoader(
            dummy_dataset,
            batch_size=start_batch_size,
            shuffle=False,
            collate_fn=dummy_collate_fn,
            num_workers=0
        )
        print("✅ Using dummy data for testing")
    
    print()
    print("Testing batch sizes...")
    print("-" * 60)
    
    # Test increasing batch sizes
    current_batch_size = start_batch_size
    max_successful_batch = None
    
    while current_batch_size <= max_batch_size:
        print(f"Testing batch_size = {current_batch_size}...", end=" ", flush=True)
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get a batch
            batch = next(iter(train_loader))
            audio_features = batch['audio_features'].to(device, non_blocking=True)
            audio_lengths = batch['audio_lengths'].to(device, non_blocking=True)
            text_tokens = batch['text_tokens'].to(device, non_blocking=True)
            text_lengths = batch['text_lengths'].to(device, non_blocking=True)
            
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits, output_lengths = model(audio_features, audio_lengths)
                    logits = logits.transpose(0, 1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logits, output_lengths = model(audio_features, audio_lengths)
                logits = logits.transpose(0, 1)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
                loss.backward()
            
            # Check VRAM usage
            vram_used = torch.cuda.memory_allocated(device) / 1e9
            vram_reserved = torch.cuda.memory_reserved(device) / 1e9
            
            print(f"✅ OK (VRAM: {vram_used:.2f}GB used, {vram_reserved:.2f}GB reserved)")
            
            max_successful_batch = current_batch_size
            
            # Clean up
            del batch, audio_features, audio_lengths, text_tokens, text_lengths
            del logits, output_lengths, log_probs, loss
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
            # Increase batch size
            current_batch_size = int(current_batch_size * 1.5)  # Increase by 50%
            if current_batch_size == max_successful_batch:
                current_batch_size += 4  # At least increase by 4
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM (Out of Memory)")
                break
            else:
                print(f"❌ Error: {e}")
                break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
    
    print("-" * 60)
    print()
    
    if max_successful_batch:
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"✅ Maximum batch size: {max_successful_batch}")
        print()
        print("RECOMMENDATIONS:")
        print(f"  1. Set batch_size: {max_successful_batch} in your config")
        print(f"  2. Or use batch_size: {max_successful_batch - 4} (safer, leaves headroom)")
        print()
        
        # Calculate gradient accumulation recommendations
        target_effective_batch = max_successful_batch * 2
        if target_effective_batch > max_successful_batch:
            recommended_batch = max_successful_batch - 4
            grad_accum = (target_effective_batch + recommended_batch - 1) // recommended_batch
            print(f"  3. For effective batch size of {target_effective_batch}:")
            print(f"     - batch_size: {recommended_batch}")
            print(f"     - gradient_accumulation_steps: {grad_accum}")
            print(f"     - Effective batch size: {recommended_batch * grad_accum}")
        print()
        
        # VRAM info
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM Usage at batch_size={max_successful_batch}:")
        print(f"  - Total VRAM: {vram_total:.2f} GB")
        print(f"  - Recommended to leave ~2GB headroom for system")
        print(f"  - Safe batch_size: {max_successful_batch - 4}")
    else:
        print("❌ Could not find a working batch size. Check your model size or VRAM.")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Find maximum batch size for training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/librispeech_rtx5060ti.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--start-batch-size',
        type=int,
        default=8,
        help='Starting batch size to test (default: 8)'
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=128,
        help='Maximum batch size to test (default: 128)'
    )
    
    args = parser.parse_args()
    
    find_max_batch_size(
        config_path=args.config,
        start_batch_size=args.start_batch_size,
        max_batch_size=args.max_batch_size
    )


if __name__ == '__main__':
    main()

