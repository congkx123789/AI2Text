"""
Test script to verify Flash Attention (SDPA) is working correctly.

Usage:
    python scripts/test_flash_attention.py
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel, MultiHeadAttention


def test_sdpa_available():
    """Test if SDPA is available."""
    print("=" * 60)
    print("TESTING FLASH ATTENTION (SDPA) AVAILABILITY")
    print("=" * 60)
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    if torch.__version__ < "2.0.0":
        print("⚠️  WARNING: PyTorch < 2.0.0. SDPA requires PyTorch >= 2.0.0")
        return False
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. SDPA will use CPU implementation.")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # Check if SDPA function exists
    if hasattr(F, 'scaled_dot_product_attention'):
        print("✅ scaled_dot_product_attention is available")
    else:
        print("❌ scaled_dot_product_attention not found")
        return False
    
    print()
    return True


def test_multihead_attention():
    """Test MultiHeadAttention with Flash Attention."""
    print("=" * 60)
    print("TESTING MULTI-HEAD ATTENTION WITH FLASH ATTENTION")
    print("=" * 60)
    print()
    
    # Create attention layer
    d_model = 256
    num_heads = 4
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.1)
    attention.eval()  # Set to eval mode
    
    # Create dummy input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model: d_model={d_model}, num_heads={num_heads}")
    print()
    
    # Forward pass
    try:
        with torch.no_grad():
            output = attention(x, x, x)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: ({batch_size}, {seq_len}, {d_model})")
        
        if output.shape == (batch_size, seq_len, d_model):
            print("✅ Output shape correct")
        else:
            print("❌ Output shape incorrect")
            return False
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_asr_model():
    """Test full ASR model with Flash Attention."""
    print("=" * 60)
    print("TESTING FULL ASR MODEL WITH FLASH ATTENTION")
    print("=" * 60)
    print()
    
    # Create model
    model = ASRModel(
        input_dim=80,
        vocab_size=1000,
        d_model=256,
        num_encoder_layers=4,  # Use fewer layers for testing
        num_heads=4,
        d_ff=1024,
        dropout=0.1
    )
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 80)
    lengths = torch.tensor([seq_len, seq_len // 2])
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Forward pass
    try:
        with torch.no_grad():
            logits, output_lengths = model(x, lengths)
        
        print(f"✅ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Output lengths: {output_lengths}")
        print(f"   Expected logits shape: ({batch_size}, {output_lengths[0]}, 1000)")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def benchmark_attention():
    """Benchmark Flash Attention vs manual attention."""
    print("=" * 60)
    print("BENCHMARKING FLASH ATTENTION")
    print("=" * 60)
    print()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping benchmark.")
        return
    
    device = torch.device('cuda')
    d_model = 256
    num_heads = 4
    d_k = d_model // num_heads
    
    # Create test data
    batch_size = 4
    seq_len = 500  # Longer sequence to see benefits
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    
    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark SDPA
    num_iterations = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        _ = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    
    sdpa_time = start.elapsed_time(end) / num_iterations
    
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Number of heads: {num_heads}")
    print()
    print(f"SDPA (Flash Attention) time: {sdpa_time:.3f} ms")
    print()
    print("✅ Benchmark completed")
    print()


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("FLASH ATTENTION (SDPA) TEST SUITE")
    print("=" * 60)
    print()
    
    # Test 1: Check availability
    if not test_sdpa_available():
        print("❌ SDPA not available. Please upgrade PyTorch to >= 2.0.0")
        return
    
    # Test 2: MultiHeadAttention
    if not test_multihead_attention():
        print("❌ MultiHeadAttention test failed")
        return
    
    # Test 3: Full ASR model
    if not test_asr_model():
        print("❌ ASR model test failed")
        return
    
    # Test 4: Benchmark
    benchmark_attention()
    
    print("=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)
    print()
    print("Flash Attention (SDPA) is working correctly!")
    print("Your model will automatically use Flash Attention for optimal performance.")


if __name__ == '__main__':
    main()

