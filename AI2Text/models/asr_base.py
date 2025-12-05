"""
Base ASR model architecture with modular components.
Includes encoder-decoder architecture with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

# Import modern LLaMA-style components
from models.modern_components import RMSNorm, RotaryPositionalEmbedding, apply_rotary_pos_emb


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
    """
    Convolutional subsampling layer for reducing sequence length.
    
    Modernized with SiLU activation instead of ReLU.
    """
    
    def __init__(self, in_channels: int, out_channels: int, subsampling_factor: int = 2):
        super().__init__()
        # FIX: Giảm stride để tăng time resolution - giúp CTC có đủ frames để align
        # Thay vì stride=2 cho cả 2 lớp (tổng 4x), chỉ dùng 1 lớp stride=2 (tổng 2x)
        if subsampling_factor == 4:
            # Original: 2 layers with stride=2 each (4x total)
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.use_conv2 = True
        else:
            # FIXED: Only 1 layer with stride=2 (2x total) - gives more time steps for CTC
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.conv2 = None
            self.use_conv2 = False
        # MODERNIZATION: SiLU instead of ReLU
        self.activation = nn.SiLU()
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
        
        x = self.activation(self.conv1(x))  # SiLU instead of ReLU
        if self.use_conv2:
            x = self.activation(self.conv2(x))  # Only if using 2-layer subsampling
        
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
    - Numerical stability tốt hơn
    - Tốc độ nhanh hơn 2-4x (Flash Attention trên RTX 5060 Ti)
    - Xử lý mask chính xác, tránh NaN trong gradient
    
    Now supports RoPE (Rotary Positional Embedding) for better position encoding.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        
        # Linear projections (kept separate for compatibility with pre-trained weights)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # 2. Apply RoPE (Rotary Embeddings)
        if self.use_rope and rope_cos is not None:
            Q, K = apply_rotary_pos_emb(Q, K, rope_cos, rope_sin)

        # 3. Masking Logic (Keep your existing logic for compatibility)
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = mask
            else:
                if mask.min() >= 0 and mask.max() <= 1:
                    attn_mask = (mask > 0.5).bool()
                else:
                    attn_mask = mask
            
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        # 4. Flash Attention (SDPA)
        # Hàm này tự động chọn kernel tối ưu (FlashAttention-2 hoặc MemoryEfficient)
        # Giúp tính toán chính xác hơn (ít lỗi làm tròn) và nhanh hơn.
        # 
        # Numerical stability: SDPA xử lý softmax internally với độ chính xác cao,
        # tránh overflow/underflow issues.
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
    """
    Position-wise feed-forward network.
    
    Modernized with SiLU (Swish) activation instead of ReLU.
    SiLU prevents "dead neurons" and provides smoother gradient flow.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # MODERNIZATION: SiLU instead of ReLU
        self.activation = nn.SiLU()  # Swish activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            output: Output tensor
        """
        x = self.linear1(x)
        x = self.activation(x)  # SiLU instead of ReLU
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """LLaMA-style Encoder Layer: Pre-Norm, RMSNorm, SiLU, RoPE support."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, use_rope=use_rope)
        
        # LLaMA uses SiLU (Swish) instead of ReLU
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # LLaMA uses RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope_cos=None, rope_sin=None) -> torch.Tensor:
        # Pre-Norm Architecture (Norm -> Attention -> Add)
        residual = x
        x_norm = self.norm1(x)
        attn_output = self.self_attention(x_norm, x_norm, x_norm, mask, rope_cos, rope_sin)
        x = residual + self.dropout(attn_output)
        
        # Pre-Norm Architecture (Norm -> FFN -> Add)
        residual = x
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)
        
        return x


class ASREncoder(nn.Module):
    """ASR Encoder with RoPE and LLaMA-style blocks.
    
    Now supports Language Embedding to help model distinguish between languages.
    """
    
    def __init__(self, input_dim: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, dropout: float = 0.1,
                 num_languages: int = 2):
        """Initialize ASR Encoder.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            num_languages: Number of languages (default: 2 for Vietnamese + English)
        """
        super().__init__()
        
        # FIX: Giảm subsampling từ 4x xuống 2x để tăng time resolution cho CTC
        self.subsampling_factor = 2  # Changed from 4 to 2
        self.subsampling = ConvSubsampling(1, d_model // 4, subsampling_factor=2)
        subsampled_dim = (input_dim // 2) * (d_model // 4)  # Changed from //4 to //2
        self.linear_proj = nn.Linear(subsampled_dim, d_model)
        # Simple feature normalization after projection to stabilize scale
        self.input_norm = nn.LayerNorm(d_model)
        
        # Language Embedding: Helps model distinguish between languages
        # 0 = Vietnamese, 1 = English (or use language_id from data)
        self.language_embedding = nn.Embedding(num_languages, d_model)
        
        # New: Rotary Embeddings
        self.rope = RotaryPositionalEmbedding(d_model // num_heads)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Use new EncoderLayers with use_rope=True
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_rope=True)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                language_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional language embedding.
        
        Args:
            x: Input features (batch, time, freq)
            lengths: Sequence lengths
            language_ids: Language IDs (batch,) where 0=Vietnamese, 1=English
                         If None, no language embedding is added
        
        Returns:
            x: Encoded features (batch, time, d_model)
            lengths: Updated sequence lengths
        """
        # 1. Subsampling
        x = self.subsampling(x)
        x = self.linear_proj(x)
        # Normalize projected features to avoid saturation / scale drift
        x = self.input_norm(x)
        
        # 2. Add language embedding if provided
        if language_ids is not None:
            # language_ids: (batch,)
            # Get language embeddings: (batch, d_model)
            lang_emb = self.language_embedding(language_ids)  # (batch, d_model)
            # Add to all time steps: (batch, 1, d_model) -> broadcast to (batch, time, d_model)
            x = x + lang_emb.unsqueeze(1)
        
        # 3. No absolute position encoding added here (RoPE is applied inside layers)
        x = self.dropout_layer(x)
        
        # 4. Generate RoPE cache for this batch
        seq_len = x.size(1)
        rope_cos, rope_sin = self.rope(x, seq_len)
        
        # 5. Pass through layers
        for layer in self.layers:
            x = layer(x, mask=None, rope_cos=rope_cos, rope_sin=rope_sin)
        
        # 6. Final Norm
        x = self.norm(x)
        
        # Update lengths - FIX: Changed from /4 to /2 due to reduced subsampling
        if lengths is not None:
            lengths = (lengths / 2).long()  # Changed from /4 to /2
        
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
    """Complete ASR model with encoder and CTC decoder.
    
    Now supports Language Embedding for bilingual (Vietnamese + English) training.
    """
    
    def __init__(self, input_dim: int, vocab_size: int, 
                 d_model: int = 1024, num_encoder_layers: int = 24,
                 num_heads: int = 16, d_ff: int = 4096, dropout: float = 0.1,
                 num_languages: int = 2):
        """Initialize ASR model.
        
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel spectrograms)
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            num_languages: Number of languages (default: 2 for Vietnamese + English)
        """
        super().__init__()
        
        self.encoder = ASREncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            num_languages=num_languages
        )
        
        self.decoder = ASRDecoder(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None,
                language_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ASR model.
        
        Args:
            x: Input features (batch, time, freq)
            lengths: Sequence lengths
            language_ids: Language IDs (batch,) where 0=Vietnamese, 1=English
                         If None, no language embedding is added (backward compatible)
            
        Returns:
            logits: Output logits (batch, time, vocab_size)
            lengths: Updated sequence lengths
        """
        # Encode (with optional language embedding)
        encoded, lengths = self.encoder(x, lengths, language_ids=language_ids)
        
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

