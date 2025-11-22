"""
LSTM-based ASR model with CTC loss.
Alternative architecture to Transformer for comparison and weak hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMBiRNN(nn.Module):
    """
    Bidirectional LSTM layer.
    Used for sequence modeling in LSTM-based ASR.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize bidirectional LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x: Input tensor (batch, time, features)
            lengths: Sequence lengths for padding
            
        Returns:
            output: LSTM output (batch, time, hidden_size * 2)
            lengths: Original lengths (unchanged)
        """
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x)
        
        # Unpack sequence
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        
        return output, lengths


class LSTMASRModel(nn.Module):
    """
    LSTM-based ASR model with CTC decoder.
    
    Architecture:
        Input → Conv → LSTM Layers → Linear → CTC Decoder
    
    Suitable for weak hardware and as baseline comparison.
    """
    
    def __init__(self,
                 input_dim: int,
                 vocab_size: int,
                 hidden_size: int = 256,
                 num_lstm_layers: int = 3,
                 dropout: float = 0.1,
                 use_conv: bool = True):
        """
        Initialize LSTM ASR model.
        
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel spectrograms)
            vocab_size: Vocabulary size
            hidden_size: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            use_conv: Whether to use convolutional layers before LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Optional convolutional layers for feature extraction
        if use_conv:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
            )
            # Calculate output size after convolutions
            conv_output_dim = 64 * (input_dim // 4)  # After 2 stride-2 convolutions
        else:
            self.conv_layers = None
            conv_output_dim = input_dim
        
        # Bidirectional LSTM layers
        self.lstm = LSTMBiRNN(
            input_size=conv_output_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # Output projection to vocabulary
        # Hidden size * 2 because bidirectional
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, time, features)
            lengths: Sequence lengths
            
        Returns:
            logits: Output logits (batch, time, vocab_size)
            lengths: Updated sequence lengths
        """
        # Apply convolutional layers if enabled
        if self.conv_layers is not None:
            # Reshape for conv2d: (batch, time, freq) -> (batch, 1, time, freq)
            x = x.unsqueeze(1)
            # Convolve
            x = self.conv_layers(x)
            # Reshape back: (batch, channels, time', freq') -> (batch, time', channels * freq')
            batch, channels, time, freq = x.size()
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(batch, time, channels * freq)
            
            # Update lengths (reduced by 4x due to stride-2 convolutions)
            if lengths is not None:
                lengths = (lengths / 4).long()
        
        # Apply dropout
        x = self.dropout(x)
        
        # LSTM layers
        x, lengths = self.lstm(x, lengths)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits, lengths
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test LSTM model
    model = LSTMASRModel(input_dim=80, vocab_size=100, hidden_size=256)
    x = torch.randn(2, 100, 80)  # (batch, time, features)
    lengths = torch.tensor([100, 80])
    
    logits, output_lengths = model(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {model.get_num_params():,}")

