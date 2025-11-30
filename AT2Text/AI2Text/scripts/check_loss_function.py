#!/usr/bin/env python3
"""
Script để kiểm tra Loss Function có được tính đúng không.

Vấn đề: Nếu dùng Sum thay vì Mean cho loss, hoặc chưa normalize theo chiều dài sequence,
loss có thể có scale không ổn định.

Với Cross Entropy tiêu chuẩn, loss thường bắt đầu quanh ln(vocab_size).
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders


def check_loss_function(config_path: str, checkpoint_path: str = None, num_batches: int = 20):
    """
    Kiểm tra loss function có được tính đúng không.
    
    Args:
        config_path: Đường dẫn đến file config
        checkpoint_path: Đường dẫn đến checkpoint (nếu None, dùng model mới)
        num_batches: Số batches để kiểm tra
    """
    print("=" * 80)
    print("KIỂM TRA LOSS FUNCTION")
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
    
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    print(f"Expected initial loss (ln(vocab_size)): {np.log(vocab_size):.4f}")
    print()
    
    normalizer = BilingualTextNormalizer()
    
    # Setup model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=vocab_size,
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
    else:
        print("Using randomly initialized model (no checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Setup loss function (same as training)
    criterion = torch.nn.CTCLoss(
        blank=tokenizer.blank_token_id,
        zero_infinity=True,
        reduction='mean'  # CTC Loss uses mean by default
    )
    
    # Load data
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    
    # Create data loader
    from preprocessing.audio_processing import AudioAugmenter
    augmenter = AudioAugmenter()
    
    train_loader, _ = create_data_loaders(
        train_df=train_df,
        val_df=train_df.iloc[:100],  # Dummy val
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter,
        persistent_workers=False,
        prefetch_factor=2,
        sort_by_length=config.get('sort_by_length', True),
        use_bucketing=False,
        cache_in_ram=False
    )
    
    print(f"Checking {num_batches} batches...")
    print()
    
    # Statistics
    losses = []
    output_lengths_list = []
    text_lengths_list = []
    batch_sizes = []
    
    # Check batches
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Checking")):
        if batch_idx >= num_batches:
            break
        
        audio_features = batch['audio_features'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        with torch.no_grad():
            logits, output_lengths = model(audio_features, audio_lengths)
            
            # Check output_lengths vs text_lengths
            # CTC requires: output_lengths >= text_lengths (after subsampling)
            for i in range(len(output_lengths)):
                if output_lengths[i] < text_lengths[i]:
                    print(f"⚠️  Batch {batch_idx}, Sample {i}:")
                    print(f"   Output length ({output_lengths[i]}) < Text length ({text_lengths[i]})")
                    print(f"   This will cause CTC loss to fail!")
            
            # CTC loss
            logits_t = logits.transpose(0, 1)  # (T, N, C)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            
            loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            losses.append(loss.item())
            output_lengths_list.extend(output_lengths.cpu().tolist())
            text_lengths_list.extend(text_lengths.cpu().tolist())
            batch_sizes.append(len(audio_features))
    
    # Summary
    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 80)
    print(f"Batches checked: {len(losses)}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(f"Min loss: {np.min(losses):.4f}")
    print(f"Max loss: {np.max(losses):.4f}")
    print(f"Std loss: {np.std(losses):.4f}")
    print()
    
    # Check output lengths
    avg_output_len = np.mean(output_lengths_list)
    avg_text_len = np.mean(text_lengths_list)
    min_output_len = np.min(output_lengths_list)
    min_text_len = np.min(text_lengths_list)
    
    print("Length Statistics:")
    print(f"  Average output length: {avg_output_len:.2f}")
    print(f"  Average text length: {avg_text_len:.2f}")
    print(f"  Min output length: {min_output_len}")
    print(f"  Min text length: {min_text_len}")
    print()
    
    # Diagnosis
    expected_initial_loss = np.log(vocab_size)
    avg_loss = np.mean(losses)
    
    # Check 1: Loss scale
    if avg_loss < 0.1:
        print("🚨 CẢNH BÁO: Loss quá thấp (< 0.1)!")
        print("   Có thể do:")
        print("   1. Loss function scale sai (dùng sum thay vì mean)")
        print("   2. Model đã collapse (output blank)")
        print("   3. Loss được normalize sai")
        print()
        print(f"   Expected initial loss: ~{expected_initial_loss:.4f}")
        print(f"   Actual loss: {avg_loss:.4f}")
        return True
    elif avg_loss < expected_initial_loss * 0.1:
        print("⚠️  CẢNH BÁO: Loss thấp hơn expected!")
        print(f"   Expected: ~{expected_initial_loss:.4f} (ln(vocab_size))")
        print(f"   Actual: {avg_loss:.4f}")
        print("   Có thể do model đã học quá nhanh hoặc loss scale sai.")
        return True
    elif avg_loss > expected_initial_loss * 10:
        print("⚠️  CẢNH BÁO: Loss cao hơn expected!")
        print(f"   Expected: ~{expected_initial_loss:.4f} (ln(vocab_size))")
        print(f"   Actual: {avg_loss:.4f}")
        print("   Có thể do loss scale sai hoặc model chưa được khởi tạo đúng.")
        return True
    
    # Check 2: Output length < Text length
    invalid_count = sum(1 for ol, tl in zip(output_lengths_list, text_lengths_list) if ol < tl)
    if invalid_count > 0:
        print("🚨 CẢNH BÁO: Có samples với output_length < text_length!")
        print(f"   Số samples invalid: {invalid_count}/{len(output_lengths_list)}")
        print("   CTC Loss yêu cầu: output_length >= text_length")
        print("   Nguyên nhân: Subsampling trong model làm giảm sequence length quá nhiều")
        print()
        print("   Giải pháp:")
        print("   1. Giảm số lớp subsampling trong model")
        print("   2. Tăng hop_length trong audio processing")
        print("   3. Kiểm tra lại model architecture")
        return True
    
    # Check 3: Loss variance
    if np.std(losses) > np.mean(losses) * 2:
        print("⚠️  CẢNH BÁO: Loss variance quá cao!")
        print(f"   Mean: {np.mean(losses):.4f}")
        print(f"   Std: {np.std(losses):.4f}")
        print("   Có thể do:")
        print("   1. Batch size quá nhỏ")
        print("   2. Data distribution không đồng đều")
        print("   3. Loss function không stable")
        return False
    
    print("✅ Loss function có vẻ đúng.")
    print(f"   Loss scale hợp lý: {avg_loss:.4f} (expected: ~{expected_initial_loss:.4f})")
    print("   Output lengths >= Text lengths: OK")
    print("   Loss variance: OK")
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check loss function correctness')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--num_batches', type=int, default=20,
                       help='Number of batches to check (default: 20)')
    
    args = parser.parse_args()
    
    has_issue = check_loss_function(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_batches=args.num_batches
    )
    
    sys.exit(1 if has_issue else 0)

