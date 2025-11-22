"""
Confidence scoring for ASR predictions.

Computes confidence scores for predictions to filter low-confidence results
and improve quality control.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


def compute_confidence_from_logits(logits: torch.Tensor,
                                   predictions: torch.Tensor,
                                   lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute confidence scores from logits.
    
    Confidence is computed as the average of max probabilities for each time step.
    
    Args:
        logits: Model logits (batch, time, vocab_size)
        predictions: Predicted token IDs (batch, time)
        lengths: Sequence lengths (batch,)
        
    Returns:
        Confidence scores (batch,)
    """
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get max probabilities for each time step
    max_probs, _ = torch.max(probs, dim=-1)  # (batch, time)
    
    # Average over sequence (with length masking)
    if lengths is not None:
        batch_size = logits.size(0)
        confidences = torch.zeros(batch_size, device=logits.device)
        
        for i in range(batch_size):
            seq_len = lengths[i].item()
            confidences[i] = max_probs[i, :seq_len].mean()
    else:
        confidences = max_probs.mean(dim=1)
    
    return confidences


def compute_word_level_confidence(logits: torch.Tensor,
                                  decoded_tokens: List[List[int]],
                                  vocab_size: int,
                                  blank_token_id: int) -> List[float]:
    """
    Compute word-level confidence scores.
    
    Args:
        logits: Model logits (batch, time, vocab_size)
        decoded_tokens: List of decoded token sequences (one per batch item)
        vocab_size: Vocabulary size
        blank_token_id: Blank token ID
        
    Returns:
        List of average confidence scores per decoded sequence
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)  # (batch, time)
    
    confidences = []
    
    for b in range(logits.size(0)):
        # Get confidence for decoded tokens only
        token_probs = []
        
        # CTC collapse for decoding
        prev_token = None
        for token in decoded_tokens[b]:
            if token != prev_token and token != blank_token_id:
                # Find time step with this token (simplified - uses max prob)
                # In practice, would align tokens to time steps
                token_probs.append(max_probs[b, len(token_probs)].item())
            prev_token = token
        
        if token_probs:
            avg_confidence = np.mean(token_probs)
        else:
            avg_confidence = 0.0
        
        confidences.append(avg_confidence)
    
    return confidences


def filter_by_confidence(results: List[Dict],
                         min_confidence: float = 0.5) -> List[Dict]:
    """
    Filter results by confidence threshold.
    
    Args:
        results: List of result dictionaries with 'confidence' key
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered results
    """
    filtered = [r for r in results if r.get('confidence', 0.0) >= min_confidence]
    return filtered


def add_confidence_to_predictions(predictions: List[str],
                                  logits: torch.Tensor,
                                  token_ids: List[List[int]],
                                  lengths: Optional[torch.Tensor] = None) -> List[Dict]:
    """
    Add confidence scores to predictions.
    
    Args:
        predictions: List of predicted texts
        logits: Model logits (batch, time, vocab_size)
        token_ids: Token ID sequences (one per prediction)
        lengths: Sequence lengths (optional)
        
    Returns:
        List of dictionaries with 'text' and 'confidence'
    """
    # Compute confidence
    if lengths is not None:
        confidences = compute_confidence_from_logits(logits, None, lengths)
    else:
        # Use max probabilities
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        confidences = max_probs.mean(dim=1)
    
    results = []
    for i, text in enumerate(predictions):
        results.append({
            'text': text,
            'confidence': confidences[i].item() if isinstance(confidences[i], torch.Tensor) else confidences[i]
        })
    
    return results


class ConfidenceScorer:
    """
    Confidence scorer for ASR predictions.
    """
    
    def __init__(self, method: str = 'max_prob'):
        """
        Initialize confidence scorer.
        
        Args:
            method: Confidence computation method ('max_prob' or 'entropy')
        """
        self.method = method
    
    def compute(self,
               logits: torch.Tensor,
               predictions: Optional[torch.Tensor] = None,
               lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute confidence scores.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            predictions: Predicted tokens (optional)
            lengths: Sequence lengths (optional)
            
        Returns:
            Confidence scores (batch,)
        """
        if self.method == 'max_prob':
            return compute_confidence_from_logits(logits, predictions, lengths)
        elif self.method == 'entropy':
            # Lower entropy = higher confidence
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            if lengths is not None:
                batch_size = logits.size(0)
                confidences = torch.zeros(batch_size, device=logits.device)
                for i in range(batch_size):
                    seq_len = lengths[i].item()
                    avg_entropy = entropy[i, :seq_len].mean()
                    # Convert entropy to confidence (inverse, normalized)
                    confidences[i] = 1.0 / (1.0 + avg_entropy)
            else:
                avg_entropy = entropy.mean(dim=1)
                confidences = 1.0 / (1.0 + avg_entropy)
            
            return confidences
        else:
            raise ValueError(f"Unknown method: {self.method}")


if __name__ == "__main__":
    # Test confidence computation
    logits = torch.randn(2, 50, 100)
    predictions = torch.randint(0, 100, (2, 50))
    lengths = torch.tensor([50, 45])
    
    scorer = ConfidenceScorer(method='max_prob')
    confidences = scorer.compute(logits, predictions, lengths)
    
    print(f"Confidences: {confidences}")
    print(f"Shape: {confidences.shape}")

