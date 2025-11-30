#!/usr/bin/env python3
"""
Test script để kiểm tra loss calculation trong training với một vài batches.
Chạy nhanh để verify loss calculation đúng.
"""

import torch
import sys
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from training.train import ASRTrainer


def test_training_loss(config_path: str, num_batches: int = 5, checkpoint_path: str = None):
    """
    Test loss calculation với một vài batches.
    
    Args:
        config_path: Đường dẫn đến config file
        num_batches: Số batches để test
        checkpoint_path: Đường dẫn đến checkpoint (optional)
    """
    print("=" * 80)
    print("TEST TRAINING LOSS CALCULATION")
    print("=" * 80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {config_path}")
    print(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"Gradient Accumulation Steps: {config.get('gradient_accumulation_steps', 1)}")
    print(f"Batch Size: {config.get('batch_size', 'N/A')}")
    print()
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load data
    split_version = config.get('split_version', 'v1')
    print(f"Using split_version: {split_version}")
    train_df = db.get_split_data('train', split_version)
    val_df = db.get_split_data('val', split_version)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print()
    
    # Create data loaders
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()
    
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Expected initial loss: ~{torch.log(torch.tensor(len(tokenizer))):.4f}")
    print()
    
    train_loader, val_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter,
        persistent_workers=False,  # Don't need persistent for quick test
        prefetch_factor=2,
        sort_by_length=config.get('sort_by_length', True),
        use_bucketing=config.get('use_bucketing', False),
        num_buckets=config.get('num_buckets', 10),
        cache_in_ram=False
    )
    
    # Initialize trainer
    trainer = ASRTrainer(config, db)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        print()
    
    print(f"Testing with {num_batches} batches...")
    print()
    
    # Test training loss calculation
    trainer.model.train()
    total_loss = 0
    num_processed = 0
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    losses = []
    true_losses = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Testing", total=num_batches)):
        if batch_idx >= num_batches:
            break
        
        # Move to device
        audio_features = batch['audio_features'].to(trainer.device, non_blocking=True)
        audio_lengths = batch['audio_lengths'].to(trainer.device, non_blocking=True)
        text_tokens = batch['text_tokens'].to(trainer.device, non_blocking=True)
        text_lengths = batch['text_lengths'].to(trainer.device, non_blocking=True)
        
        # Forward pass
        with torch.no_grad():
            logits, output_lengths = trainer.model(audio_features, audio_lengths)
            
            # Check output_lengths >= text_lengths
            invalid_mask = output_lengths < text_lengths
            if invalid_mask.any():
                invalid_count = invalid_mask.sum().item()
                print(f"⚠️  Batch {batch_idx}: {invalid_count} samples with output_length < text_length")
            
            # CTC loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = trainer.criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Batch {batch_idx}: Loss is NaN/Inf!")
                continue
            
            # True loss (for display)
            true_loss = loss.item()
            losses.append(loss.item())
            true_losses.append(true_loss)
            
            total_loss += true_loss
            num_processed += 1
    
    if num_processed == 0:
        print("❌ No batches processed!")
        return
    
    avg_loss = total_loss / num_processed
    
    # Summary
    print()
    print("=" * 80)
    print("KẾT QUẢ TEST")
    print("=" * 80)
    print(f"Batches processed: {num_processed}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Min loss: {min(losses):.4f}")
    print(f"Max loss: {max(losses):.4f}")
    print(f"Std loss: {torch.tensor(losses).std().item():.4f}")
    print()
    print("Loss per batch:")
    for i, loss_val in enumerate(losses):
        print(f"  Batch {i}: {loss_val:.4f}")
    print()
    
    # Check if loss is reasonable
    expected_loss = torch.log(torch.tensor(len(tokenizer))).item()
    print(f"Expected initial loss (ln(vocab_size)): {expected_loss:.4f}")
    
    if avg_loss < 0.1:
        print("🚨 CẢNH BÁO: Loss quá thấp (< 0.1)!")
        print("   Có thể do:")
        print("   1. Model đã collapse (output blank)")
        print("   2. Loss function scale sai")
        return False
    elif avg_loss < expected_loss * 0.1:
        print("⚠️  CẢNH BÁO: Loss thấp hơn expected!")
        print(f"   Expected: ~{expected_loss:.4f}")
        print(f"   Actual: {avg_loss:.4f}")
        return False
    elif avg_loss > expected_loss * 10:
        print("⚠️  CẢNH BÁO: Loss cao hơn expected!")
        print(f"   Expected: ~{expected_loss:.4f}")
        print(f"   Actual: {avg_loss:.4f}")
        return False
    else:
        print("✅ Loss scale hợp lý!")
        print(f"   Loss: {avg_loss:.4f} (expected: ~{expected_loss:.4f})")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test training loss calculation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches to test (default: 5)')
    
    args = parser.parse_args()
    
    success = test_training_loss(
        config_path=args.config,
        num_batches=args.num_batches,
        checkpoint_path=args.checkpoint
    )
    
    sys.exit(0 if success else 1)

