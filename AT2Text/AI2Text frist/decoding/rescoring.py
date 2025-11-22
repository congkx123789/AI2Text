"""
N-best rescoring with semantic and phonetic embeddings.

Rescores N-best hypotheses using Word2Vec semantic embeddings and Phon2Vec
phonetic embeddings. Enables contextual biasing and better handling of
rare/OOV words.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from gensim.models import KeyedVectors

# Import phonetic processing
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.phonetic import phonetic_tokens


def _sent_embedding(tokens: List[str], kv: KeyedVectors) -> np.ndarray:
    """
    Compute sentence embedding as mean of word embeddings.
    
    Args:
        tokens: List of tokens (words or phonetic tokens)
        kv: KeyedVectors model
        
    Returns:
        Normalized sentence embedding vector
    """
    vecs = []
    for t in tokens:
        if t in kv:
            vecs.append(kv[t])
    
    if not vecs:
        return np.zeros((kv.vector_size,), dtype=np.float32)
    
    # Mean pooling
    v = np.mean(vecs, axis=0)
    
    # L2 normalization
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    
    return v.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    
    if na == 0 or nb == 0:
        return 0.0
    
    return float(np.dot(a, b) / (na * nb + 1e-9))


def rescore_nbest(
    nbest: List[Dict[str, Any]],
    semantic_kv: Optional[KeyedVectors] = None,
    phon_kv: Optional[KeyedVectors] = None,
    context_text: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.5,
    delta: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Rescore N-best hypotheses using semantic and phonetic embeddings.
    
    Combines acoustic scores with semantic similarity (Word2Vec) and
    phonetic similarity (Phon2Vec) for better hypothesis selection.
    
    Args:
        nbest: List of hypotheses, each with:
            - 'text': Hypothesis text (string)
            - 'am_score': Acoustic model score (float)
            - 'lm_score': Optional language model score (float)
        semantic_kv: Word2Vec keyed vectors for semantic similarity
        phon_kv: Phon2Vec keyed vectors for phonetic similarity
        context_text: Optional context text for biasing (e.g., domain-specific terms)
        alpha: Weight for acoustic score (default: 1.0)
        beta: Weight for language model score (default: 0.0)
        gamma: Weight for semantic similarity (default: 0.5)
        delta: Weight for phonetic similarity (default: 0.5)
        
    Returns:
        Rescored hypotheses sorted by re_score (descending)
        
    Example:
        >>> nbest = [
        ...     {'text': 'toi muon dat ban', 'am_score': -12.5},
        ...     {'text': 'toi muon dat banh', 'am_score': -12.8}
        ... ]
        >>> context = "đặt bánh gato sinh nhật"
        >>> rescored = rescore_nbest(
        ...     nbest,
        ...     semantic_kv=semantic_model,
        ...     phon_kv=phon_model,
        ...     context_text=context,
        ...     gamma=0.5,
        ...     delta=0.3
        ... )
        >>> print(rescored[0]['text'])  # Best hypothesis after rescoring
    """
    # Build context embeddings if context provided
    ctx_sem = None
    ctx_ph = None
    
    if context_text:
        if semantic_kv:
            ctx_tokens = context_text.lower().split()
            ctx_sem = _sent_embedding(ctx_tokens, semantic_kv)
        
        if phon_kv:
            ph_toks = phonetic_tokens(context_text, telex=True, tone_token=True)
            ctx_ph = _sent_embedding(ph_toks, phon_kv)
    
    rescored = []
    
    for hyp in nbest:
        text = hyp.get("text", "")
        am_score = float(hyp.get("am_score", 0.0))
        lm_score = float(hyp.get("lm_score", 0.0))
        
        # Base score: acoustic + language model
        base_score = alpha * am_score + beta * lm_score
        
        # Semantic similarity score
        sem_score = 0.0
        if semantic_kv:
            text_tokens = text.lower().split()
            text_emb = _sent_embedding(text_tokens, semantic_kv)
            
            if ctx_sem is not None:
                # Similarity to context
                sem_score = cosine_similarity(text_emb, ctx_sem)
            else:
                # Use embedding norm as proxy (words that exist in model)
                sem_score = float(np.linalg.norm(text_emb))
            
            base_score += gamma * sem_score
        
        # Phonetic similarity score
        ph_score = 0.0
        if phon_kv:
            ph_toks = phonetic_tokens(text, telex=True, tone_token=True)
            ph_emb = _sent_embedding(ph_toks, phon_kv)
            
            if ctx_ph is not None:
                # Similarity to context
                ph_score = cosine_similarity(ph_emb, ctx_ph)
            else:
                # Use embedding norm
                ph_score = float(np.linalg.norm(ph_emb))
            
            base_score += delta * ph_score
        
        # Create rescored hypothesis
        nh = dict(hyp)
        nh["re_score"] = float(base_score)
        nh["sem_score"] = sem_score
        nh["ph_score"] = ph_score
        
        rescored.append(nh)
    
    # Sort by rescore (descending)
    rescored.sort(key=lambda x: x["re_score"], reverse=True)
    
    return rescored


def contextual_biasing(
    nbest: List[Dict[str, Any]],
    bias_list: List[str],
    semantic_kv: Optional[KeyedVectors] = None,
    phon_kv: Optional[KeyedVectors] = None,
    bias_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Apply contextual biasing to N-best list using bias words.
    
    Boosts hypotheses that contain or are similar to words in the bias list.
    Useful for domain-specific terms, names, or rare words.
    
    Args:
        nbest: List of hypotheses
        bias_list: List of words to bias towards (e.g., names, domain terms)
        semantic_kv: Word2Vec for semantic similarity
        phon_kv: Phon2Vec for phonetic similarity
        bias_weight: Weight for bias boost (default: 0.3)
        
    Returns:
        Biased and rescored hypotheses
    """
    # Build bias embeddings
    bias_sem = None
    bias_ph = None
    
    if semantic_kv and bias_list:
        bias_tokens = [word.lower() for word in bias_list]
        bias_sem = _sent_embedding(bias_tokens, semantic_kv)
    
    if phon_kv and bias_list:
        bias_ph_tokens = []
        for word in bias_list:
            bias_ph_tokens.extend(phonetic_tokens(word, telex=True, tone_token=True))
        if bias_ph_tokens:
            bias_ph = _sent_embedding(bias_ph_tokens, phon_kv)
    
    # Rescore with bias context
    biased_text = ' '.join(bias_list)
    
    rescored = rescore_nbest(
        nbest,
        semantic_kv=semantic_kv,
        phon_kv=phon_kv,
        context_text=biased_text,
        gamma=bias_weight,
        delta=bias_weight * 0.5  # Slightly less weight for phonetic
    )
    
    return rescored

