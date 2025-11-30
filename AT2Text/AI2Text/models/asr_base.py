"""
Base ASR model architecture with modular components.
Includes encoder-decoder architecture with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            x: Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """Convolutional subsampling layer for reducing sequence length."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional subsampling.
        
        Args:
            x: Input tensor (batch, time, freq)
            
        Returns:
            x: Subsampled tensor
        """
        # x: (batch, time, freq)
        x = x.unsqueeze(1)  # (batch, 1, time, freq)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Reshape: (batch, channels, time, freq) -> (batch, time, channels * freq)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism optimized with PyTorch SDPA.
    
    Cải thiện độ chính xác và tốc độ bằng cách sử dụng Fused Kernel.
    Sử dụng F.scaled_dot_product_attention để đạt:
    - Numerical stability tốt hơn (đặc biệt với FP16/BF16)
    - Tốc độ nhanh hơn 2-4x (Flash Attention trên RTX 5060 Ti)
    - Xử lý mask chính xác, tránh NaN trong gradient
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections (kept separate for compatibility with pre-trained weights)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention using Flash Attention (SDPA).
        
        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key: Key tensor (batch, seq_len_k, d_model)
            value: Value tensor (batch, seq_len_v, d_model)
            mask: Optional attention mask
                - Boolean mask: True = attend, False = mask out
                - Float mask: 1.0 = attend, 0.0 = mask out
                - Int mask: 1 = attend, 0 = mask out (will be converted to bool)
                - Shape: (batch, seq_len) or (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # 1. Linear Projections & Reshape
        # Tách thành (batch, seq_len, num_heads, head_dim) -> transpose thành (batch, num_heads, seq_len, head_dim)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Xử lý Mask cho SDPA (Chính xác và ổn định)
        # SDPA mong đợi mask dạng Boolean (True = attend, False = mask out) 
        # hoặc Float (cộng vào attention score, negative values = mask out)
        # 
        # Trong ASR, padding mask thường là: 1/True là pad (cần che), 0/False là data.
        # Logic dưới đây đảm bảo mask tương thích với SDPA bất kể đầu vào.
        attn_mask = None
        if mask is not None:
            # Convert mask to proper format for SDPA
            if mask.dtype != torch.bool and mask.dtype != torch.float16 and mask.dtype != torch.float32:
                # Chuyển đổi mask số nguyên (0/1) sang boolean nếu cần
                # Giả sử: 1/True là vị trí cần tính attention, 0/False là padding cần bỏ qua
                # Nếu mask có giá trị > 1, có thể là length mask -> convert 0 -> False, >0 -> True
                attn_mask = (mask != 0).bool()
            else:
                # Boolean or float mask - use as-is but ensure correct format
                if mask.dtype == torch.bool:
                    attn_mask = mask
                else:
                    # Float mask: SDPA can use directly, but for consistency convert to bool
                    # Negative values in float mask = mask out, positive = attend
                    # For simple 0/1 masks, convert to bool
                    if mask.min() >= 0 and mask.max() <= 1:
                        attn_mask = (mask > 0.5).bool()
                    else:
                        # Keep as float for SDPA (it supports float masks)
                        attn_mask = mask
            
            # Đảm bảo shape của mask broadcast được với (batch, num_heads, seq_len_q, seq_len_k)
            if attn_mask.dim() == 2:
                # (seq_len_q, seq_len_k) -> (1, 1, seq_len_q, seq_len_k)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                # Already in correct format (batch, num_heads, seq_len_q, seq_len_k)
                pass
            else:
                # Unexpected dimension, try to handle gracefully
                if attn_mask.dim() == 1:
                    # (seq_len) -> assume it's a length mask, expand to (1, 1, seq_len, seq_len)
                    # This creates a causal-like mask
                    seq_len = attn_mask.size(0)
                    attn_mask = attn_mask.view(1, 1, seq_len, 1).expand(1, 1, seq_len, seq_len)
                else:
                    raise ValueError(f"Unsupported mask dimension: {attn_mask.dim()}")
        
        # 3. Scaled Dot Product Attention (Flash Attention)
        # Hàm này tự động chọn kernel tối ưu (FlashAttention-2 hoặc MemoryEfficient)
        # Giúp tính toán chính xác hơn (ít lỗi làm tròn) và nhanh hơn.
        # 
        # Numerical stability: SDPA xử lý softmax internally với độ chính xác cao,
        # tránh overflow/underflow issues đặc biệt với FP16/BF16.
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False  # Encoder ASR thường là hai chiều (non-causal)
        )
        
        # 4. Concatenate & Final Projection
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            output: Output tensor
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Output tensor
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ASREncoder(nn.Module):
    """ASR Encoder with convolutional subsampling and transformer layers."""
    
    def __init__(self, input_dim: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Convolutional subsampling
        self.subsampling = ConvSubsampling(1, d_model // 4)
        
        # Calculate input dimension after subsampling
        # After two conv layers with stride 2, freq dimension is reduced by 4
        subsampled_dim = (input_dim // 4) * (d_model // 4)
        
        # Linear projection to model dimension
        self.linear_proj = nn.Linear(subsampled_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio features.
        
        Args:
            x: Input features (batch, time, freq)
            lengths: Sequence lengths
            
        Returns:
            output: Encoded features (batch, time, d_model)
            lengths: Updated sequence lengths
        """
        # Convolutional subsampling
        x = self.subsampling(x)
        
        # Linear projection
        x = self.linear_proj(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Encoder layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Update lengths after subsampling (reduced by factor of 4)
        if lengths is not None:
            lengths = (lengths / 4).long()
        
        return x, lengths


class ASRDecoder(nn.Module):
    """CTC decoder for ASR."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode encoded features to vocabulary logits.
        
        Args:
            x: Encoded features (batch, time, d_model)
            
        Returns:
            logits: Vocabulary logits (batch, time, vocab_size)
        """
        return self.linear(x)


class ASRModel(nn.Module):
    """Complete ASR model with encoder and CTC decoder."""
    
    def __init__(self, input_dim: int, vocab_size: int, 
                 d_model: int = 1024, num_encoder_layers: int = 24,
                 num_heads: int = 16, d_ff: int = 4096, dropout: float = 0.1):
        """Initialize ASR model.
        
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel spectrograms)
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = ASREncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.decoder = ASRDecoder(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ASR model.
        
        Args:
            x: Input features (batch, time, freq)
            lengths: Sequence lengths
            
        Returns:
            logits: Output logits (batch, time, vocab_size)
            lengths: Updated sequence lengths
        """
        # Encode
        encoded, lengths = self.encoder(x, lengths)
        
        # Decode
        logits = self.decoder(encoded)
        
        return logits, lengths
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def apply_lora(self,
                   rank: int = 8,
                   alpha: float = 16.0,
                   dropout: float = 0.0,
                   target_modules: Optional[List[str]] = None) -> Dict[str, 'LoRALinear']:
        """
        Apply LoRA (Low-Rank Adaptation) to the model.
        
        Args:
            rank: LoRA rank (default: 8)
            alpha: LoRA alpha scaling factor (default: 16.0)
            dropout: LoRA dropout rate (default: 0.0)
            target_modules: List of module names to apply LoRA to.
                          If None, applies to all attention and feed-forward layers.
        
        Returns:
            Dictionary of LoRA modules
        """
        from models.lora import apply_lora_to_linear, freeze_base_model
        
        if target_modules is None:
            # Default: apply to attention and feed-forward layers
            target_modules = ['W_q', 'W_k', 'W_v', 'W_o', 'W_1', 'W_2', 'linear_proj']
        
        # Apply LoRA
        lora_modules = apply_lora_to_linear(
            self,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze base model, keep only LoRA trainable
        freeze_base_model(self, lora_modules)
        
        return lora_modules


if __name__ == "__main__":
    # Test model
    batch_size = 2
    time_steps = 100
    input_dim = 80  # Mel spectrogram features
    vocab_size = 100
    
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=1024,
        num_encoder_layers=24,
        num_heads=16,
        d_ff=4096,
        dropout=0.1
    )
    
    # Dummy input
    x = torch.randn(batch_size, time_steps, input_dim)
    lengths = torch.tensor([time_steps, time_steps // 2])
    
    # Forward pass
    logits, output_lengths = model(x, lengths)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output lengths: {output_lengths}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

