from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from gensim.models import KeyedVectors
from nlp.phonetic import phonetic_tokens

def _sent_embedding(tokens: List[str], kv: KeyedVectors) -> np.ndarray:
    vecs = []
    for t in tokens:
        if t in kv:
            vecs.append(kv[t])
    if not vecs:
        return np.zeros((kv.vector_size,), dtype=np.float32)
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-9))

def rescore_nbest(
    nbest: List[Dict[str, Any]],
    semantic_kv: Optional[KeyedVectors],
    phon_kv: Optional[KeyedVectors],
    context_text: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.5,
    delta: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    nbest: list of { 'text': str, 'am_score': float, 'lm_score': float (optional) }
    Returns the same list with 're_score' and sorted by it (desc).
    """
    # Build context vector for semantics & phonetics
    ctx_sem = None; ctx_ph = None
    if semantic_kv and context_text:
        ctx_sem = _sent_embedding(context_text.split(), semantic_kv)
    if phon_kv and context_text:
        ph_toks = phonetic_tokens(context_text, telex=True, tone_token=True)
        ctx_ph = _sent_embedding(ph_toks, phon_kv)

    rescored = []
    for hyp in nbest:
        text = hyp.get("text","")
        am = float(hyp.get("am_score", 0.0))
        lm = float(hyp.get("lm_score", 0.0))
        score = alpha * am + beta * lm

        if semantic_kv:
            v = _sent_embedding(text.split(), semantic_kv)
            sem = cosine(v, ctx_sem) if ctx_sem is not None else float(np.linalg.norm(v))
            score += gamma * sem

        if phon_kv:
            ph = phonetic_tokens(text, telex=True, tone_token=True)
            v = _sent_embedding(ph, phon_kv)
            phs = cosine(v, ctx_ph) if ctx_ph is not None else float(np.linalg.norm(v))
            score += delta * phs

        nh = dict(hyp)
        nh["re_score"] = float(score)
        rescored.append(nh)

    rescored.sort(key=lambda x: x["re_score"], reverse=True)
    return rescored
