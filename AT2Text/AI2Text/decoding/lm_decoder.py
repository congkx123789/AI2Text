"""
Language Model decoder with KenLM integration.

Integrates KenLM language model with CTC decoding for significant WER improvement.
This is the highest-impact improvement from the roadmap.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False
    print("Warning: pyctcdecode not available. Install with: pip install pyctcdecode")

try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    print("Warning: kenlm not available. Install with: pip install kenlm")


class LMBeamSearchDecoder:
    """
    Beam search decoder with KenLM language model integration.
    
    Uses KenLM for language model scoring combined with CTC acoustic scores.
    Provides significant WER improvement (typically 10-30%).
    """
    
    def __init__(self,
                 vocab: List[str],
                 lm_path: Optional[str] = None,
                 unigrams: Optional[List[Tuple[str, float]]] = None,
                 alpha: float = 0.5,
                 beta: float = 1.5,
                 beam_width: int = 128,
                 blank_token_id: int = 0,
                 vocab_size: Optional[int] = None):
        """
        Initialize LM beam search decoder.
        
        Args:
            vocab: Vocabulary list (token strings)
            lm_path: Path to KenLM .arpa file (optional)
            unigrams: List of (word, count) tuples for unigram frequencies
            alpha: LM weight (default: 0.5)
            beta: Word bonus (default: 1.5)
            beam_width: Beam search width (default: 128)
            blank_token_id: ID of blank token for CTC
            vocab_size: Vocabulary size (inferred from vocab if None)
        """
        self.vocab = vocab
        self.lm_path = lm_path
        self.blank_token_id = blank_token_id
        self.vocab_size = vocab_size or len(vocab)
        
        # Build CTC decoder with LM if available
        if PYCTCDECODE_AVAILABLE:
            if lm_path and Path(lm_path).exists():
                try:
                    self.decoder = build_ctcdecoder(
                        labels=vocab,
                        kenlm_model_path=lm_path,
                        unigrams=unigrams,
                        alpha=alpha,
                        beta=beta,
                        beam_width=beam_width
                    )
                    self.has_lm = True
                except Exception as e:
                    print(f"Warning: Failed to load LM from {lm_path}: {e}")
                    print("Falling back to basic beam search")
                    self.decoder = None
                    self.has_lm = False
            else:
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    alpha=alpha,
                    beta=beta,
                    beam_width=beam_width
                )
                self.has_lm = False
        else:
            self.decoder = None
            self.has_lm = False
            print("Warning: pyctcdecode not available. Using basic beam search.")
    
    def decode(self,
               logits: torch.Tensor,
               lengths: Optional[torch.Tensor] = None,
               beam_size: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:
        """
        Decode logits using LM-enhanced beam search.
        
        Args:
            logits: Model logits (batch, time, vocab_size) or (batch, time, vocab_size)
            lengths: Sequence lengths (batch,)
            beam_size: Override beam width (optional)
            
        Returns:
            List of dictionaries with:
                - 'text': Decoded text
                - 'score': Decoding score
                - 'lm_score': Language model score (if LM used)
        """
        if self.decoder is None:
            # Fallback to basic decoding
            from decoding.beam_search import BeamSearchDecoder
            basic_decoder = BeamSearchDecoder(
                vocab_size=self.vocab_size,
                blank_token_id=self.blank_token_id,
                beam_width=beam_size or 5
            )
            results = basic_decoder.decode(logits, lengths)
            return [{"text": " ".join([self.vocab[t] for t in hyp['text']]),
                     "score": hyp['length_penalty_score']} 
                    for hyp in results[0]]
        
        batch_size = logits.size(0)
        results = []
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()
        
        for b in range(batch_size):
            if lengths is not None:
                seq_len = lengths[b].item()
                batch_log_probs = log_probs[b, :seq_len, :]
            else:
                batch_log_probs = log_probs[b, :, :]
            
            # Decode with LM
            if self.has_lm:
                # Use pyctcdecode with KenLM
                decoded_output = self.decoder.decode(
                    batch_log_probs,
                    beam_width=beam_size or 128
                )
                text = decoded_output.text
                lm_score = getattr(decoded_output, 'lm_score', 0.0)
                score = getattr(decoded_output, 'score', 0.0)
                
                results.append({
                    "text": text,
                    "score": float(score),
                    "lm_score": float(lm_score) if self.has_lm else 0.0
                })
            else:
                # Basic CTC decoding without LM
                decoded_output = self.decoder.decode(batch_log_probs)
                results.append({
                    "text": decoded_output.text,
                    "score": float(getattr(decoded_output, 'score', 0.0)),
                    "lm_score": 0.0
                })
        
        return results
    
    def decode_batch(self,
                    logits: torch.Tensor,
                    lengths: Optional[torch.Tensor] = None) -> List[str]:
        """
        Decode batch and return best hypotheses.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            lengths: Sequence lengths
            
        Returns:
            List of best decoded texts
        """
        results = self.decode(logits, lengths)
        return [r["text"] for r in results]


def create_lm_decoder(vocab: List[str],
                     lm_path: Optional[str] = None,
                     alpha: float = 0.5,
                     beta: float = 1.5) -> LMBeamSearchDecoder:
    """
    Factory function to create LM decoder.
    
    Args:
        vocab: Vocabulary list
        lm_path: Path to KenLM .arpa file
        alpha: LM weight
        beta: Word bonus
        
    Returns:
        LMBeamSearchDecoder instance
    """
    return LMBeamSearchDecoder(
        vocab=vocab,
        lm_path=lm_path,
        alpha=alpha,
        beta=beta
    )


if __name__ == "__main__":
    # Test LM decoder
    vocab = ["<blank>", "<pad>", "a", "b", "c", " ", "xin", "chào"]
    
    # Without LM (basic)
    decoder = LMBeamSearchDecoder(vocab=vocab, blank_token_id=0)
    
    # Dummy logits
    logits = torch.randn(2, 50, len(vocab))
    lengths = torch.tensor([50, 45])
    
    results = decoder.decode(logits, lengths)
    print(f"Decoded {len(results)} hypotheses")
    for i, r in enumerate(results):
        print(f"  {i+1}: {r['text']} (score: {r['score']:.4f})")

