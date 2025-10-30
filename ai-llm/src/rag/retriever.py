from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self, index, embed_model: str):
        self.index = index
        self.model = SentenceTransformer(embed_model)


    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
    # BM25 scores
        tokens = query.split()
        bm25_scores = self.index.bm25.get_scores(tokens)
        bm25_ranks = np.argsort(-bm25_scores)[:k*5]


        # Dense scores
        q = self.model.encode([query], normalize_embeddings=True)[0]
        D, I = self.index.faiss_index.search(np.asarray([q], dtype=np.float32), k*5)
        dense_scores = D[0]


        # Hybrid: sum of normalized ranks
        candidates = set(bm25_ranks.tolist()) | set(I[0].tolist())
        out = []
        for i in candidates:
            out.append((self.index.ids[i], self.index.texts[i], float(bm25_scores[i]) + float(dense_scores[list(I[0]).index(i)]) if i in I[0] else float(bm25_scores[i])))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:k]