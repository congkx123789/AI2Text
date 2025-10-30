from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)


    def rerank(self, query: str, docs: List[Tuple[str, str, float]], top_k: int = 5):
        pairs = [(query, d[1]) for d in docs]
        scores = self.model.predict(pairs).tolist()
        reranked = [(docs[i][0], docs[i][1], float(scores[i])) for i in range(len(docs))]
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked[:top_k]