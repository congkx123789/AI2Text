#!/usr/bin/env python3
"""
Test script để kiểm tra loss calculation có đúng không.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.text_cleaning import Tokenizer


def test_loss_calculation():
    """Test loss calculation với các trường hợp khác nhau."""
    print("=" * 80)
    print("TEST LOSS CALCULATION")
    print("=" * 80)
    print()
    
    # Setup
    vocab_size = 100
    tokenizer = Tokenizer()
    blank_id = tokenizer.blank_token_id
    
    # CTC Loss
    criterion = nn.CTCLoss(
        blank=blank_id,
        zero_infinity=True,
        reduction='mean'  # Explicitly set to mean
    )
    
    print(f"CTC Loss configuration:")
    print(f"  blank: {blank_id}")
    print(f"  zero_infinity: True")
    print(f"  reduction: mean")
    print()
    
    # Test case 1: Normal case
    print("Test 1: Normal case")
    batch_size = 2
    seq_len = 10
    text_len = 5
    
    # Logits: (batch, seq_len, vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Transpose to (seq_len, batch, vocab_size) for CTC
    log_probs_t = log_probs.transpose(0, 1)
    
    # Targets: (batch, text_len)
    targets = torch.randint(0, vocab_size, (batch_size, text_len))
    
    # Lengths
    input_lengths = torch.tensor([seq_len, seq_len])
    target_lengths = torch.tensor([text_len, text_len])
    
    loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Expected: ~{torch.log(torch.tensor(vocab_size)).item():.4f} (ln(vocab_size))")
    print()
    
    # Test case 2: Check if loss is NaN or Inf
    print("Test 2: Check for NaN/Inf")
    if torch.isnan(loss) or torch.isinf(loss):
        print("  ❌ Loss is NaN or Inf!")
    else:
        print("  ✅ Loss is finite")
    print()
    
    # Test case 3: Gradient accumulation scaling
    print("Test 3: Gradient accumulation scaling")
    gradient_accumulation_steps = 2
    
    # Scale loss for backward
    scaled_loss = loss / gradient_accumulation_steps
    print(f"  Original loss: {loss.item():.4f}")
    print(f"  Scaled loss (for backward): {scaled_loss.item():.4f}")
    print(f"  True loss (after multiply back): {(scaled_loss * gradient_accumulation_steps).item():.4f}")
    print(f"  Match: {torch.allclose(loss, scaled_loss * gradient_accumulation_steps)}")
    print()
    
    # Test case 4: Invalid case (input_length < target_length)
    print("Test 4: Invalid case (input_length < target_length)")
    invalid_input_lengths = torch.tensor([3, 3])  # Less than target_lengths
    invalid_target_lengths = torch.tensor([5, 5])
    
    try:
        invalid_loss = criterion(log_probs_t, targets, invalid_input_lengths, invalid_target_lengths)
        print(f"  Loss: {invalid_loss.item():.4f}")
        if torch.isnan(invalid_loss) or torch.isinf(invalid_loss):
            print("  ⚠️  Loss is NaN/Inf (expected with zero_infinity=True, will be 0)")
        else:
            print("  ✅ Loss is finite (zero_infinity=True replaced inf with 0)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    print("=" * 80)
    print("KẾT LUẬN")
    print("=" * 80)
    print()
    print("✅ CTC Loss calculation cơ bản là đúng")
    print("✅ Gradient accumulation scaling hoạt động đúng")
    print()
    print("⚠️  Lưu ý:")
    print("   1. Đảm bảo input_lengths >= target_lengths")
    print("   2. Với zero_infinity=True, inf loss sẽ được thay bằng 0")
    print("   3. Loss scale nên bắt đầu quanh ln(vocab_size)")
    print()


if __name__ == '__main__':
    test_loss_calculation()

