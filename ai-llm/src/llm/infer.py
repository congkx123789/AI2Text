from __future__ import annotations
from typing import List, Tuple
from transformers import TextStreamer
from .load import load_llm


_tok, _model = None, None


def _ensure():
    global _tok, _model
    if _tok is None:
        _tok, _model = load_llm()




def generate_with_citations(prompt: str, hits: List[Tuple[str, str, float]]):
    _ensure()
    ctx = "\n\n".join([f"[{i+1}] {h[1]}" for i, h in enumerate(hits)])
    full = f"Answer the question using the sources and cite like [1], [2].\n\nQuestion: {prompt}\n\nSources:\n{ctx}\n\nAnswer:"
    ids = _tok([full], return_tensors="pt").to(_model.device)
    out = _model.generate(**ids, max_new_tokens=400)
    text = _tok.decode(out[0], skip_special_tokens=True)
    answer = text.split("Answer:")[-1].strip()
    cites = [{"id": hits[i][0], "text": hits[i][1]} for i in range(len(hits))]
    return answer, cites