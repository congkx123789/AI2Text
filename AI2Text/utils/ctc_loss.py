"""
CTC Loss for Hybrid CTC/Attention training.

CTC helps the encoder learn better alignment between audio and text,
preventing the decoder from ignoring encoder outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    """CTC Loss wrapper for hybrid CTC/Attention training."""
    
    def __init__(self, blank_id: int = 0, reduction: str = 'mean'):
        """Initialize CTC Loss.
        
        Args:
            blank_id: ID of blank token (usually 0)
            reduction: 'mean' or 'none'
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=True)
        self.blank_id = blank_id
    
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss.
        
        Args:
            logits: Encoder CTC logits (batch, time, vocab_size)
            targets: Target token IDs (batch, target_len)
            input_lengths: Encoder output lengths (batch,)
            target_lengths: Target sequence lengths (batch,)
            
        Returns:
            ctc_loss: Scalar loss value
        """
        # CTC expects (time, batch, vocab_size)
        logits = logits.transpose(0, 1)  # (time, batch, vocab_size)
        
        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss

