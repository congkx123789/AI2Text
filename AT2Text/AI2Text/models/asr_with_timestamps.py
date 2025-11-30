"""
ASR Model with Timestamp Prediction Support.

Extends the base ASR model to predict word-level timestamps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from models.asr_base import ASRModel, ASREncoder


class TimestampHead(nn.Module):
    """Head for predicting word timestamps from encoder output."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize timestamp head.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        # Predict start and end times for each frame
        # Output: (batch, time, 2) where 2 = [start_time, end_time] in seconds
        self.linear1 = nn.Linear(d_model, d_model // 2)
        self.linear2 = nn.Linear(d_model // 2, 2)  # [start, end] times
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict timestamps.
        
        Args:
            x: Encoder output (batch, time, d_model)
            
        Returns:
            timestamps: (batch, time, 2) where 2 = [start_time, end_time] in seconds
        """
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        timestamps = self.linear2(x)
        # Ensure timestamps are non-negative
        timestamps = F.relu(timestamps)
        return timestamps


class ASRModelWithTimestamps(ASRModel):
    """ASR Model with timestamp prediction capability."""
    
    def __init__(self, input_dim: int, vocab_size: int, 
                 d_model: int = 1024, num_encoder_layers: int = 24,
                 num_heads: int = 16, d_ff: int = 4096, dropout: float = 0.1,
                 predict_timestamps: bool = True):
        """Initialize ASR model with timestamp support.
        
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel spectrograms)
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            predict_timestamps: If True, add timestamp prediction head
        """
        # Initialize base model
        super().__init__(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.predict_timestamps = predict_timestamps
        
        if predict_timestamps:
            self.timestamp_head = TimestampHead(d_model, dropout)
        else:
            self.timestamp_head = None
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None,
                return_timestamps: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through ASR model with optional timestamp prediction.
        
        Args:
            x: Input features (batch, time, freq)
            lengths: Sequence lengths
            return_timestamps: If True, also return timestamp predictions
            
        Returns:
            logits: Output logits (batch, time, vocab_size)
            lengths: Updated sequence lengths
            timestamps: Optional timestamp predictions (batch, time, 2) if return_timestamps=True
        """
        # Encode
        encoded, lengths = self.encoder(x, lengths)
        
        # Decode to vocabulary
        logits = self.decoder(encoded)
        
        # Predict timestamps if enabled
        timestamps = None
        if self.predict_timestamps and return_timestamps:
            timestamps = self.timestamp_head(encoded)
        
        return logits, lengths, timestamps
    
    def predict_word_timestamps(self, logits: torch.Tensor, 
                               timestamps: torch.Tensor,
                               text_tokens: torch.Tensor,
                               tokenizer,
                               subsampling_factor: int = 4,
                               sample_rate: int = 16000,
                               hop_length: int = 160) -> List[Dict]:
        """Convert frame-level timestamps to word-level timestamps.
        
        Args:
            logits: Model output logits (batch, time, vocab_size)
            timestamps: Frame-level timestamps (batch, time, 2)
            text_tokens: Decoded text tokens (list of token IDs)
            tokenizer: Tokenizer instance
            subsampling_factor: Model subsampling factor (default 4x)
            sample_rate: Audio sample rate
            hop_length: Hop length for mel spectrogram (frames per second)
            
        Returns:
            word_timestamps: List of dicts with 'word', 'start', 'end'
        """
        # Greedy decode to get token sequence
        predictions = torch.argmax(logits, dim=-1)  # (batch, time)
        
        # CTC collapse: remove blanks and duplicates
        collapsed_tokens = []
        prev = None
        for token in predictions[0].cpu().tolist():
            if token != prev and token != tokenizer.blank_token_id:
                collapsed_tokens.append(token.item())
            prev = token
        
        # Align timestamps to tokens
        # Account for subsampling: model output is 4x shorter than input
        frame_duration = (hop_length * subsampling_factor) / sample_rate  # seconds per frame
        
        word_timestamps = []
        token_idx = 0
        
        for frame_idx in range(timestamps.shape[1]):
            if token_idx >= len(collapsed_tokens):
                break
            
            # Get timestamp for this frame
            start_time = timestamps[0, frame_idx, 0].item()
            end_time = timestamps[0, frame_idx, 1].item()
            
            # Map to token
            if token_idx < len(collapsed_tokens):
                token_id = collapsed_tokens[token_idx]
                word = tokenizer.decode([token_id])
                
                word_timestamps.append({
                    'word': word,
                    'start': start_time,
                    'end': end_time,
                    'token_id': token_id
                })
                token_idx += 1
        
        return word_timestamps


def create_timestamp_targets(word_timestamps: List[Dict], 
                            num_frames: int,
                            subsampling_factor: int = 4,
                            sample_rate: int = 16000,
                            hop_length: int = 160) -> torch.Tensor:
    """Create target timestamps for training.
    
    Args:
        word_timestamps: List of dicts with 'word', 'start', 'end'
        num_frames: Number of output frames (after subsampling)
        subsampling_factor: Model subsampling factor
        sample_rate: Audio sample rate
        hop_length: Hop length for mel spectrogram
        
    Returns:
        targets: (num_frames, 2) tensor with [start, end] times for each frame
    """
    targets = torch.zeros(num_frames, 2)
    
    frame_duration = (hop_length * subsampling_factor) / sample_rate
    
    for word_info in word_timestamps:
        start_time = word_info['start']
        end_time = word_info['end']
        
        # Find frame indices
        start_frame = int(start_time / frame_duration)
        end_frame = int(end_time / frame_duration)
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame, min(end_frame, num_frames - 1))
        
        # Assign timestamps to frames
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx < num_frames:
                # Interpolate timestamps
                frame_start = frame_idx * frame_duration
                frame_end = (frame_idx + 1) * frame_duration
                
                targets[frame_idx, 0] = max(frame_start, start_time)
                targets[frame_idx, 1] = min(frame_end, end_time)
    
    return targets


if __name__ == "__main__":
    # Test model
    batch_size = 2
    time_steps = 100
    input_dim = 80
    vocab_size = 100
    
    model = ASRModelWithTimestamps(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=1024,
        num_encoder_layers=24,
        num_heads=16,
        d_ff=4096,
        dropout=0.1,
        predict_timestamps=True
    )
    
    # Dummy input
    x = torch.randn(batch_size, time_steps, input_dim)
    lengths = torch.tensor([time_steps, time_steps // 2])
    
    # Forward pass
    logits, output_lengths, timestamps = model(x, lengths, return_timestamps=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Timestamps shape: {timestamps.shape if timestamps is not None else None}")
    print(f"Output lengths: {output_lengths}")
    print(f"Total parameters: {model.get_num_params():,}")

