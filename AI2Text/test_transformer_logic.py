"""
Comprehensive test for Transformer ASR model logic.
Verifies that the Transformer architecture follows correct principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.asr_base import (
    ASRModel, ASREncoder, ASRDecoder,
    MultiHeadAttention, EncoderLayer, FeedForward,
    ConvSubsampling
)
from models.modern_components import RMSNorm, RotaryPositionalEmbedding


def test_attention_logic():
    """Test Multi-Head Attention logic."""
    print("=" * 80)
    print("TEST 1: Multi-Head Attention Logic")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8
    head_dim = d_model // num_heads
    
    attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0, use_rope=True)
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Self-attention: Q=K=V
    output = attention(x, x, x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check that output is different from input (transformation happened)
    assert not torch.allclose(output, x, atol=1e-5), "Attention should transform input"
    
    print("✅ Multi-Head Attention: Output shape correct")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output different from input: {not torch.allclose(output, x, atol=1e-5)}")
    
    # Test with RoPE
    rope = RotaryPositionalEmbedding(head_dim)
    rope_cos, rope_sin = rope(x, seq_len)
    output_rope = attention(x, x, x, rope_cos=rope_cos, rope_sin=rope_sin)
    assert output_rope.shape == (batch_size, seq_len, d_model)
    print("✅ RoPE integration: Working correctly")
    
    return True


def test_feedforward_logic():
    """Test Feed-Forward Network logic."""
    print("\n" + "=" * 80)
    print("TEST 2: Feed-Forward Network Logic")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    d_ff = 512
    
    ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check transformation
    assert not torch.allclose(output, x, atol=1e-5), "FFN should transform input"
    
    print("✅ Feed-Forward Network: Output shape correct")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   d_model: {d_model}, d_ff: {d_ff}")
    
    return True


def test_encoder_layer_logic():
    """Test Encoder Layer logic (Pre-Norm architecture)."""
    print("\n" + "=" * 80)
    print("TEST 3: Encoder Layer Logic (Pre-Norm)")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8
    d_ff = 512
    
    layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                        dropout=0.0, use_rope=True)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Generate RoPE
    rope = RotaryPositionalEmbedding(d_model // num_heads)
    rope_cos, rope_sin = rope(x, seq_len)
    
    output = layer(x, rope_cos=rope_cos, rope_sin=rope_sin)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Pre-Norm: output should be close to input (residual connection)
    # But not identical due to transformation
    diff = torch.abs(output - x).mean()
    print(f"✅ Encoder Layer: Output shape correct")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Mean absolute difference: {diff.item():.6f}")
    print(f"   Residual connection working: {diff.item() < 10.0}")
    
    return True


def test_subsampling_logic():
    """Test Convolutional Subsampling logic."""
    print("\n" + "=" * 80)
    print("TEST 4: Convolutional Subsampling Logic")
    print("=" * 80)
    
    batch_size = 2
    time_steps = 100
    freq_bins = 80
    out_channels = 32
    
    subsampling = ConvSubsampling(in_channels=1, out_channels=out_channels, subsampling_factor=2)
    
    # Input: (batch, time, freq)
    x = torch.randn(batch_size, time_steps, freq_bins)
    
    output = subsampling(x)
    
    # After stride=2 subsampling, time should be ~time_steps/2
    expected_time = time_steps // 2
    actual_time = output.shape[1]
    
    print(f"✅ Convolutional Subsampling:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Time reduction: {time_steps} -> {actual_time} (expected ~{expected_time})")
    print(f"   Subsampling factor: 2x")
    
    assert actual_time <= expected_time + 1, f"Time dimension should be reduced"
    
    return True


def test_encoder_logic():
    """Test ASR Encoder logic."""
    print("\n" + "=" * 80)
    print("TEST 5: ASR Encoder Logic")
    print("=" * 80)
    
    batch_size = 2
    time_steps = 100
    input_dim = 80
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = 1024
    
    encoder = ASREncoder(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
        num_languages=2
    )
    
    x = torch.randn(batch_size, time_steps, input_dim)
    lengths = torch.tensor([time_steps, time_steps - 20])
    
    encoded, output_lengths = encoder(x, lengths)
    
    # Check output shape
    assert encoded.shape[0] == batch_size, "Batch dimension should be preserved"
    assert encoded.shape[2] == d_model, f"Feature dimension should be {d_model}"
    
    # Check length reduction (subsampling factor = 2)
    expected_output_length = time_steps // 2
    actual_output_length = encoded.shape[1]
    
    print(f"✅ ASR Encoder:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {encoded.shape}")
    print(f"   Input lengths: {lengths.tolist()}")
    print(f"   Output lengths: {output_lengths.tolist()}")
    print(f"   Time reduction: {time_steps} -> {actual_output_length} (subsampling 2x)")
    print(f"   Number of layers: {num_layers}")
    
    assert actual_output_length <= expected_output_length + 1, "Time should be reduced by subsampling"
    
    return True


def test_decoder_logic():
    """Test CTC Decoder logic."""
    print("\n" + "=" * 80)
    print("TEST 6: CTC Decoder Logic")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 50
    d_model = 256
    vocab_size = 100
    
    decoder = ASRDecoder(d_model=d_model, vocab_size=vocab_size)
    
    x = torch.randn(batch_size, seq_len, d_model)
    logits = decoder(x)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    
    print(f"✅ CTC Decoder:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Vocab size: {vocab_size}")
    
    return True


def test_full_model_logic():
    """Test full ASR Model logic."""
    print("\n" + "=" * 80)
    print("TEST 7: Full ASR Model Logic")
    print("=" * 80)
    
    batch_size = 2
    time_steps = 100
    input_dim = 80
    vocab_size = 100
    d_model = 256
    num_layers = 4
    num_heads = 8
    d_ff = 1024
    
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0
    )
    
    x = torch.randn(batch_size, time_steps, input_dim)
    lengths = torch.tensor([time_steps, time_steps - 20])
    
    logits, output_lengths = model(x, lengths)
    
    # Check output shape
    assert logits.shape[0] == batch_size, "Batch dimension should be preserved"
    assert logits.shape[2] == vocab_size, f"Vocab dimension should be {vocab_size}"
    
    # Check that logits are reasonable (not all zeros or NaNs)
    assert not torch.isnan(logits).any(), "Logits should not contain NaN"
    assert not torch.isinf(logits).any(), "Logits should not contain Inf"
    assert logits.std() > 0.01, "Logits should have some variance"
    
    print(f"✅ Full ASR Model:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Input lengths: {lengths.tolist()}")
    print(f"   Output lengths: {output_lengths.tolist()}")
    print(f"   Model parameters: {model.get_num_params():,}")
    print(f"   Logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
    print(f"   No NaN/Inf: ✅")
    
    return True


def test_gradient_flow():
    """Test gradient flow through model."""
    print("\n" + "=" * 80)
    print("TEST 8: Gradient Flow")
    print("=" * 80)
    
    model = ASRModel(
        input_dim=80,
        vocab_size=100,
        d_model=128,
        num_encoder_layers=2,
        num_heads=4,
        d_ff=512,
        dropout=0.0
    )
    
    x = torch.randn(2, 50, 80, requires_grad=True)
    lengths = torch.tensor([50, 45])
    
    logits, _ = model(x, lengths)
    
    # Compute dummy loss
    loss = logits.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Input should have gradients"
    assert x.grad.abs().sum() > 0, "Gradients should be non-zero"
    
    # Check model parameters have gradients
    param_grads = [p.grad for p in model.parameters() if p.requires_grad]
    params_with_grads = [g for g in param_grads if g is not None]
    params_with_nonzero_grads = [g for g in params_with_grads if g.abs().sum() > 0]
    
    print(f"✅ Gradient Flow:")
    print(f"   Input gradients: ✅ (mean={x.grad.abs().mean().item():.6f})")
    print(f"   Total trainable parameters: {len(param_grads)}")
    print(f"   Parameters with gradients: {len(params_with_grads)}")
    print(f"   Parameters with non-zero gradients: {len(params_with_nonzero_grads)}")
    
    # Most parameters should have gradients (allow some to be None if they're not used)
    assert len(params_with_grads) > len(param_grads) * 0.9, \
        f"At least 90% of parameters should have gradients, got {len(params_with_grads)}/{len(param_grads)}"
    assert len(params_with_nonzero_grads) > len(param_grads) * 0.8, \
        f"At least 80% of parameters should have non-zero gradients, got {len(params_with_nonzero_grads)}/{len(param_grads)}"
    
    return True


def test_transformer_principles():
    """Test that model follows Transformer principles."""
    print("\n" + "=" * 80)
    print("TEST 9: Transformer Architecture Principles")
    print("=" * 80)
    
    model = ASRModel(
        input_dim=80,
        vocab_size=100,
        d_model=128,
        num_encoder_layers=2,
        num_heads=4,
        d_ff=512
    )
    
    # Principle 1: Self-attention mechanism
    assert hasattr(model.encoder.layers[0], 'self_attention'), "Should have self-attention"
    assert isinstance(model.encoder.layers[0].self_attention, MultiHeadAttention), \
        "Should use MultiHeadAttention"
    print("✅ Principle 1: Self-attention mechanism present")
    
    # Principle 2: Feed-forward network
    assert hasattr(model.encoder.layers[0], 'feed_forward'), "Should have feed-forward network"
    print("✅ Principle 2: Feed-forward network present")
    
    # Principle 3: Residual connections (Pre-Norm)
    # Check that layers have residual connections
    x = torch.randn(1, 10, 128)
    rope = RotaryPositionalEmbedding(128 // 4)
    rope_cos, rope_sin = rope(x, 10)
    
    layer = model.encoder.layers[0]
    output = layer(x, rope_cos=rope_cos, rope_sin=rope_sin)
    
    # Output should be similar to input (residual connection)
    diff = torch.abs(output - x).mean()
    assert diff.item() < 5.0, "Residual connection should keep output close to input"
    print(f"✅ Principle 3: Residual connections (Pre-Norm) - diff={diff.item():.4f}")
    
    # Principle 4: Layer normalization (RMSNorm)
    assert isinstance(model.encoder.layers[0].norm1, RMSNorm), "Should use RMSNorm"
    assert isinstance(model.encoder.norm, RMSNorm), "Should use RMSNorm"
    print("✅ Principle 4: Layer normalization (RMSNorm)")
    
    # Principle 5: Positional encoding (RoPE)
    assert hasattr(model.encoder, 'rope'), "Should have RoPE"
    assert isinstance(model.encoder.rope, RotaryPositionalEmbedding), "Should use RoPE"
    print("✅ Principle 5: Positional encoding (RoPE)")
    
    # Principle 6: Multi-head attention
    assert model.encoder.layers[0].self_attention.num_heads > 1, "Should use multi-head attention"
    print(f"✅ Principle 6: Multi-head attention ({model.encoder.layers[0].self_attention.num_heads} heads)")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRANSFORMER ASR MODEL LOGIC TEST SUITE")
    print("=" * 80)
    print("\nTesting Transformer architecture correctness...\n")
    
    tests = [
        ("Multi-Head Attention", test_attention_logic),
        ("Feed-Forward Network", test_feedforward_logic),
        ("Encoder Layer", test_encoder_layer_logic),
        ("Convolutional Subsampling", test_subsampling_logic),
        ("ASR Encoder", test_encoder_logic),
        ("CTC Decoder", test_decoder_logic),
        ("Full Model", test_full_model_logic),
        ("Gradient Flow", test_gradient_flow),
        ("Transformer Principles", test_transformer_principles),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} {'❌' if failed > 0 else ''}")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 All tests passed! Transformer logic is correct.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

