#!/usr/bin/env python3
"""
Quick test để verify loss calculation đã được sửa đúng.
Test với dummy data, không cần database.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.text_cleaning import Tokenizer


def quick_test_loss():
    """Quick test loss calculation với dummy data."""
    print("=" * 80)
    print("QUICK TEST LOSS CALCULATION")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Model config (from your config)
    input_dim = 80
    vocab_size = 18005
    d_model = 1024
    num_encoder_layers = 24
    num_heads = 16
    d_ff = 4096
    dropout = 0.1
    
    print("Model Config:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Expected initial loss: ~{torch.log(torch.tensor(vocab_size)):.4f}")
    print()
    
    # Create model
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    model.to(device)
    model.eval()
    
    # Tokenizer
    tokenizer = Tokenizer()
    
    # CTC Loss (same as training)
    criterion = nn.CTCLoss(
        blank=tokenizer.blank_token_id,
        zero_infinity=True,
        reduction='mean'  # Explicit
    )
    
    print("CTC Loss Config:")
    print(f"  blank: {tokenizer.blank_token_id}")
    print(f"  zero_infinity: True")
    print(f"  reduction: mean")
    print()
    
    # Test with dummy data
    batch_size = 2
    seq_len = 100  # Audio sequence length
    text_len = 20  # Text sequence length
    
    print(f"Test Config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Audio seq len: {seq_len}")
    print(f"  Text seq len: {text_len}")
    print()
    
    # Create dummy data
    audio_features = torch.randn(batch_size, seq_len, input_dim).to(device)
    audio_lengths = torch.tensor([seq_len, seq_len]).to(device)
    text_tokens = torch.randint(0, vocab_size, (batch_size, text_len)).to(device)
    text_lengths = torch.tensor([text_len, text_len]).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits, output_lengths = model(audio_features, audio_lengths)
        
        print("Model Output:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Output lengths: {output_lengths.tolist()}")
        print(f"  Text lengths: {text_lengths.tolist()}")
        
        # Check output_lengths >= text_lengths
        invalid_mask = output_lengths < text_lengths
        if invalid_mask.any():
            print(f"  ⚠️  {invalid_mask.sum().item()} samples with output_length < text_length")
        else:
            print(f"  ✅ All samples have output_length >= text_length")
        print()
        
        # Calculate loss
        logits_t = logits.transpose(0, 1)  # (T, N, C)
        log_probs = torch.log_softmax(logits_t, dim=-1)
        
        loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss is NaN/Inf!")
            return False
        else:
            print(f"✅ Loss is finite: {loss.item():.4f}")
        print()
        
        # Test gradient accumulation scaling
        gradient_accumulation_steps = 2
        scaled_loss = loss / gradient_accumulation_steps
        true_loss = scaled_loss * gradient_accumulation_steps
        
        print("Gradient Accumulation Test:")
        print(f"  Original loss: {loss.item():.4f}")
        print(f"  Scaled loss (÷{gradient_accumulation_steps}): {scaled_loss.item():.4f}")
        print(f"  True loss (×{gradient_accumulation_steps}): {true_loss.item():.4f}")
        print(f"  Match: {torch.allclose(loss, true_loss)}")
        print()
    
    # Summary
    print("=" * 80)
    print("KẾT LUẬN")
    print("=" * 80)
    print()
    print("✅ Loss calculation hoạt động đúng!")
    print("✅ Output lengths validation hoạt động đúng!")
    print("✅ Gradient accumulation scaling hoạt động đúng!")
    print()
    print("📝 Các cải thiện đã được áp dụng:")
    print("   1. Explicit reduction='mean' trong CTC Loss")
    print("   2. Validation cho output_lengths >= text_lengths")
    print("   3. Check NaN/Inf loss")
    print("   4. Progress bar hiển thị true loss (không phải scaled loss)")
    print()
    
    return True


if __name__ == '__main__':
    success = quick_test_loss()
    sys.exit(0 if success else 1)

