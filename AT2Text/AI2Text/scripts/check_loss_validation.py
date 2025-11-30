#!/usr/bin/env python3
"""
Script để kiểm tra Train Loss vs Validation Loss.

Nếu Train Loss = 2, nhưng Val Loss = 15 → Overfitting/Leakage.
Nếu cả 2 đều giảm → Có thể do Learning Rate cao hoặc Loss function scale lạ.
"""

import torch
import sys
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from training.train import ASRTrainer


def check_loss_gap(config_path: str, checkpoint_path: str = None, num_batches: int = 50):
    """
    Kiểm tra gap giữa train loss và validation loss.
    
    Args:
        config_path: Đường dẫn đến file config
        checkpoint_path: Đường dẫn đến checkpoint (nếu None, dùng model mới)
        num_batches: Số batches để kiểm tra
    """
    print("=" * 80)
    print("KIỂM TRA TRAIN LOSS VS VALIDATION LOSS")
    print("=" * 80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
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
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
    normalizer = BilingualTextNormalizer()
    
    # Setup model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load checkpoint nếu có
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint train loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        print("Using randomly initialized model (no checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Setup loss function
    criterion = torch.nn.CTCLoss(
        blank=tokenizer.blank_token_id,
        zero_infinity=True
    )
    
    # Load data
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    val_df = db.get_split_data('val', config.get('split_version', 'v1'))
    
    # Create data loaders
    from preprocessing.audio_processing import AudioAugmenter
    augmenter = AudioAugmenter()
    
    train_loader, val_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter,
        persistent_workers=False,  # Don't need persistent for one-time check
        prefetch_factor=2,
        sort_by_length=config.get('sort_by_length', True),
        use_bucketing=config.get('use_bucketing', False),
        num_buckets=config.get('num_buckets', 10),
        cache_in_ram=False  # Don't cache for quick check
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Calculate train loss (without augmentation for fair comparison)
    print("Calculating train loss (no augmentation)...")
    train_losses = []
    train_count = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train")):
        if batch_idx >= num_batches:
            break
        
        audio_features = batch['audio_features'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        with torch.no_grad():
            logits, output_lengths = model(audio_features, audio_lengths)
            
            # CTC loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            train_losses.append(loss.item())
            train_count += 1
    
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
    
    # Calculate validation loss
    print("\nCalculating validation loss...")
    val_losses = []
    val_count = 0
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
        if batch_idx >= num_batches:
            break
        
        audio_features = batch['audio_features'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        with torch.no_grad():
            logits, output_lengths = model(audio_features, audio_lengths)
            
            # CTC loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            val_losses.append(loss.item())
            val_count += 1
    
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
    
    # Summary
    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 80)
    print(f"Train batches checked: {train_count}")
    print(f"Val batches checked: {val_count}")
    print()
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print(f"Average Val Loss:   {avg_val_loss:.4f}")
    print(f"Gap (Val - Train):  {avg_val_loss - avg_train_loss:.4f}")
    print(f"Ratio (Val/Train):  {avg_val_loss / avg_train_loss:.2f}x")
    print()
    
    # Diagnosis
    gap = avg_val_loss - avg_train_loss
    ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else float('inf')
    
    if gap > 10 and ratio > 3:
        print("🚨 CẢNH BÁO: Gap quá lớn giữa Train và Val Loss!")
        print("   Có thể do:")
        print("   1. Overfitting: Model học thuộc lòng training data")
        print("   2. Data Leakage: Training data có thông tin mà validation không có")
        print("   3. Data Distribution Mismatch: Train và Val data khác nhau quá nhiều")
        print()
        print("   Giải pháp:")
        print("   1. Kiểm tra data leakage (xem check_data_leakage.py)")
        print("   2. Tăng regularization (dropout, weight decay)")
        print("   3. Kiểm tra train/val split có đúng không")
        return True
    elif avg_train_loss < 1.0 and avg_val_loss < 1.0:
        print("⚠️  CẢNH BÁO: Cả Train và Val Loss đều quá thấp!")
        print("   Loss < 1.0 thường là dấu hiệu:")
        print("   1. Loss function scale sai (xem check_loss_function.py)")
        print("   2. Model đã collapse (xem check_inference_collapse.py)")
        print("   3. Learning rate quá cao → Model nhảy vọt vào local minima")
        print()
        print("   Với CTC Loss, loss thường bắt đầu quanh ln(vocab_size).")
        print(f"   Vocab size: {len(tokenizer)}, ln(vocab_size) ≈ {torch.log(torch.tensor(len(tokenizer))):.2f}")
        return True
    elif gap < 1.0 and ratio < 1.5:
        print("✅ Train và Val Loss gần nhau.")
        print("   Model không bị overfitting nghiêm trọng.")
        print("   Tuy nhiên, nếu loss giảm quá nhanh, vẫn nên kiểm tra:")
        print("   1. Model có bị collapse không (xem check_inference_collapse.py)")
        print("   2. Loss function có đúng không (xem check_loss_function.py)")
        return False
    else:
        print("⚠️  Gap vừa phải giữa Train và Val Loss.")
        print("   Có thể là overfitting nhẹ hoặc bình thường.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check train vs validation loss gap')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--num_batches', type=int, default=50,
                       help='Number of batches to check (default: 50)')
    
    args = parser.parse_args()
    
    has_issue = check_loss_gap(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_batches=args.num_batches
    )
    
    sys.exit(1 if has_issue else 0)

