"""
Scheduled Sampling for reducing Teacher Forcing during training.

This helps prevent the decoder from ignoring encoder outputs and only
learning language patterns. By gradually reducing teacher forcing,
the model is forced to use encoder information.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class ScheduledSampling:
    """Scheduled Sampling scheduler for reducing teacher forcing.
    
    Gradually reduces the probability of using ground truth tokens,
    forcing the model to use its own predictions.
    """
    
    def __init__(self, 
                 initial_prob: float = 1.0,
                 final_prob: float = 0.5,
                 decay_type: str = 'linear',
                 decay_steps: Optional[int] = None):
        """Initialize scheduled sampling.
        
        Args:
            initial_prob: Initial probability of using ground truth (1.0 = always use GT)
            final_prob: Final probability of using ground truth (0.0 = never use GT)
            decay_type: 'linear' or 'exponential'
            decay_steps: Number of steps to decay from initial to final (None = use epochs)
        """
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.decay_type = decay_type
        self.decay_steps = decay_steps
        self.current_step = 0
        self.current_prob = initial_prob
    
    def get_probability(self, epoch: int, total_epochs: int) -> float:
        """Get current teacher forcing probability.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Probability of using ground truth tokens
        """
        if self.decay_steps is not None:
            progress = min(self.current_step / self.decay_steps, 1.0)
        else:
            progress = min(epoch / total_epochs, 1.0)
        
        if self.decay_type == 'linear':
            self.current_prob = self.initial_prob - (self.initial_prob - self.final_prob) * progress
        elif self.decay_type == 'exponential':
            # Exponential decay: p = initial * (final/initial)^progress
            decay_rate = (self.final_prob / self.initial_prob) if self.initial_prob > 0 else 0
            self.current_prob = self.initial_prob * (decay_rate ** progress)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        return max(self.current_prob, self.final_prob)
    
    def sample_tokens(self, 
                     ground_truth: torch.Tensor,
                     predicted: torch.Tensor,
                     epoch: int,
                     total_epochs: int) -> torch.Tensor:
        """Sample tokens based on scheduled sampling probability.
        
        Args:
            ground_truth: Ground truth tokens (batch, seq_len)
            predicted: Predicted tokens from previous step (batch, seq_len)
            epoch: Current epoch
            total_epochs: Total epochs
            
        Returns:
            Mixed tokens to use as decoder input
        """
        prob = self.get_probability(epoch, total_epochs)
        
        # Create random mask: True = use ground truth, False = use prediction
        batch_size, seq_len = ground_truth.shape
        device = ground_truth.device
        use_gt = torch.rand(batch_size, seq_len, device=device) < prob
        
        # Mix ground truth and predictions
        mixed_tokens = torch.where(use_gt, ground_truth, predicted)
        
        return mixed_tokens
    
    def step(self):
        """Increment step counter (if using step-based decay)."""
        self.current_step += 1

