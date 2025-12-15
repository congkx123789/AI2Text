"""
Beam search decoding for ASR.

Implements beam search decoding for seq2seq models
by exploring multiple hypotheses simultaneously.

OPTIMIZED FOR RTX 5060TI 16GB:
- GPU-accelerated operations where possible
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np


class BeamSearchDecoder:
    """
    Beam search decoder for seq2seq ASR.
    
    Explores multiple hypotheses simultaneously to find the best transcription,
    better than greedy decoding for handling ambiguous cases.
    """
    
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int,
                 sos_token_id: int,
                 eos_token_id: int,
                 beam_width: int = 5,
                 length_penalty: float = 0.6,
                 log_probs: bool = True):
        """
        Initialize beam search decoder.
        
        Args:
            vocab_size: Vocabulary size
            pad_token_id: ID of padding token
            sos_token_id: ID of start-of-sequence token
            eos_token_id: ID of end-of-sequence token
            beam_width: Number of hypotheses to keep (default: 5)
            length_penalty: Length penalty factor (default: 0.6)
            log_probs: Whether input is log probabilities (default: True)
        """
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.log_probs = log_probs
    
    def decode(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        Decode using beam search.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            lengths: Sequence lengths (batch,)
            
        Returns:
            List of dictionaries, each containing:
                - 'text': Decoded text (as token IDs)
                - 'score': Beam score
                - 'text_decoded': Decoded text (as string, if tokenizer provided)
        """
        batch_size = logits.size(0)
        
        # Convert logits to log probabilities
        if not self.log_probs:
            log_probs = F.log_softmax(logits, dim=-1)
        else:
            log_probs = logits
        
        # Get sequence lengths
        if lengths is None:
            lengths = torch.tensor([logits.size(1)] * batch_size, device=logits.device)
        else:
            lengths = lengths.to(logits.device)
        
        results = []
        
        # Move log_probs to CPU for beam search (beam search is sequential, CPU is fine)
        # For very large batches, could optimize with GPU but current approach is simpler
        log_probs_cpu = log_probs.cpu()
        
        for b in range(batch_size):
            seq_len = int(lengths[b].item())
            batch_log_probs = log_probs_cpu[b, :seq_len, :]  # (time, vocab)
            
            # Initialize beam: list of (prefix, score, last_token)
            beam = [([], 0.0, None)]
            
            # Process each time step
            for t in range(seq_len):
                frame_log_probs = batch_log_probs[t, :]  # (vocab,)
                
                # Expand all current hypotheses
                candidates = []
                
                for prefix, score, last_token in beam:
                    # Try extending with each token
                    # OPTIMIZATION: Pre-compute scores for all tokens
                    # Convert to float32 before numpy (NumPy doesn't support bfloat16)
                    token_scores = frame_log_probs.float().cpu().numpy()  # Convert to numpy for faster access
                    
                    for token_id in range(self.vocab_size):
                        token_log_prob = float(token_scores[token_id])
                        new_score = score + token_log_prob
                        
                        # Create new prefix
                        if token_id == self.pad_token_id or token_id == self.eos_token_id:
                            # Skip padding and EOS tokens during generation
                            continue
                        else:
                            # New token - append to prefix
                            new_prefix = prefix + [token_id]
                        
                        candidates.append((new_prefix, new_score, token_id))
                
                # Keep top beam_width hypotheses
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:self.beam_width]
            
            # Prepare results
            beam_results = []
            for prefix, score, _ in beam:
                # Remove SOS token if present
                filtered_prefix = [t for t in prefix if t != self.sos_token_id]
                
                # Apply length penalty
                length_penalty_score = score / ((len(filtered_prefix) + 1) ** self.length_penalty)
                
                beam_results.append({
                    'text': filtered_prefix,
                    'score': score,
                    'length_penalty_score': length_penalty_score
                })
            
            # Sort by score
            beam_results.sort(key=lambda x: x['length_penalty_score'], reverse=True)
            results.append(beam_results)
        
        return results
    
    def decode_batch(self, logits: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                     tokenizer=None) -> List[Dict]:
        """
        Decode batch with optional tokenizer for text output.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            lengths: Sequence lengths
            tokenizer: Optional tokenizer for decoding text
            
        Returns:
            List of best hypotheses, each with:
                - 'text': Token IDs
                - 'text_decoded': Text (if tokenizer provided)
                - 'score': Decoding score
        """
        beam_results = self.decode(logits, lengths)
        
        decoded_results = []
        for i, batch_results in enumerate(beam_results):
            # Get best hypothesis (top of beam)
            best = batch_results[0]
            
            result = {
                'text': best['text'],
                'score': best['length_penalty_score']
            }
            
            # Decode to text if tokenizer provided
            if tokenizer is not None:
                result['text_decoded'] = tokenizer.decode(best['text'])
            
            decoded_results.append(result)
        
        return decoded_results


def generate_nbest(logits: torch.Tensor,
                   decoder: BeamSearchDecoder,
                   n: int = 5,
                   lengths: Optional[torch.Tensor] = None) -> List[List[Dict]]:
    """
    Generate N-best hypotheses using beam search.
    
    Args:
        logits: Model logits (batch, time, vocab_size)
        decoder: BeamSearchDecoder instance
        n: Number of hypotheses to return
        lengths: Sequence lengths
        
    Returns:
        List of N-best hypotheses for each sample in batch
    """
    beam_results = decoder.decode(logits, lengths)
    
    nbest_results = []
    for batch_results in beam_results:
        # Get top N
        nbest = batch_results[:n]
        
        # Format as N-best list
        formatted_nbest = []
        for rank, hyp in enumerate(nbest):
            formatted_nbest.append({
                'rank': rank + 1,
                'text': hyp['text'],
                'score': hyp['score'],
                'length_penalty_score': hyp['length_penalty_score']
            })
        
        nbest_results.append(formatted_nbest)
    
    return nbest_results

