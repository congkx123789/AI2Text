"""
Base ASR model architecture with modular components.
Includes encoder-decoder architecture with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


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
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attention output (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
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
                 d_model: int = 256, num_encoder_layers: int = 6,
                 num_heads: int = 4, d_ff: int = 1024, dropout: float = 0.1):
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


if __name__ == "__main__":
    # Test model
    batch_size = 2
    time_steps = 100
    input_dim = 80  # Mel spectrogram features
    vocab_size = 100
    
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=256,
        num_encoder_layers=4,
        num_heads=4,
        d_ff=1024,
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

