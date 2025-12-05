"""
Modern Transformer Components (LLaMA-style).

This module implements modern architectural improvements over the original Transformer:
- RMSNorm: Root Mean Square Layer Normalization (simpler, more stable)
- RoPE: Rotary Positional Embedding (better relative position encoding)
- SiLU/Swish: Smooth activation function (prevents dead neurons)

These improvements significantly improve training stability, especially for deep models (24+ layers).
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    Proposed by Zhang and Sennrich (2019), used in LLaMA.
    Simplifies LayerNorm by removing mean centering and bias.
    
    Benefits:
    - Better gradient flow (prevents exploding gradients in deep models)
    - Computational efficiency (no mean calculation)
    - More stable training for 24+ layer models
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor (..., d_model)
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS: RMS(x) = sqrt(mean(x^2))
        pow_mean = x.pow(2).mean(-1, keepdim=True)
        norm_x = x * torch.rsqrt(pow_mean + self.eps)
        return norm_x * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes relative position by rotating query and key vectors.
    Better than absolute positional encoding for:
    - Capturing relative positions
    - Generalizing to longer sequences
    - Better attention patterns
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (should be head_dim, not full d_model)
            max_seq_len: Maximum sequence length to cache
        """
        super().__init__()
        self.d_model = d_model
        
        # Theta parameters
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos = None
        self.cached_sin = None
    
    def forward(self, x, seq_len: int):
        # x shape: [batch, seq_len, d_model]
        if self.cached_cos is None or self.cached_cos.size(1) < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self.cached_cos = emb.cos()[None, :, :]
            self.cached_sin = emb.sin()[None, :, :]
        
        return self.cached_cos[:, :seq_len, :], self.cached_sin[:, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.
    
    For RoPE: rotate [x1, x2] -> [-x2, x1]
    
    Args:
        x: Input tensor (..., d_model)
        
    Returns:
        Rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to queries and keys.
    
    Formula: q_rot = q * cos + rotate_half(q) * sin
             k_rot = k * cos + rotate_half(k) * sin
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine embeddings (1, seq_len, head_dim)
        sin: Sine embeddings (1, seq_len, head_dim)
        
    Returns:
        q_embed: Rotated query tensor
        k_embed: Rotated key tensor
    """
    # Expand cos/sin to match q/k dimensions: (1, seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)  # (1, 1, seq_len, head_dim)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

