from __future__ import annotations
from typing import Dict, Optional
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.llm.infer import generate_with_citations
from src.config import LLM_PROVIDER


class RAGPipeline:
    def __init__(self, index, embed_model: str, reranker_model: str, llm_provider: Optional[str] = None):
        self.retriever = HybridRetriever(index, embed_model)
        self.reranker = Reranker(reranker_model)
        self.llm_provider = llm_provider or LLM_PROVIDER


    def ask(self, query: str, k: int = 5) -> Dict:
        hits = self.retriever.search(query, k=k)
        hits = self.reranker.rerank(query, hits, top_k=k)
        answer, cites = generate_with_citations(query, hits, provider=self.llm_provider)
        return {"answer": answer, "contexts": cites}