from __future__ import annotations
import json
from pathlib import Path
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss


class HybridIndex:
    def __init__(self, embed_model: str):
        self.model = SentenceTransformer(embed_model)
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.bm25 = None
        self.faiss_index = None
        self.emb_dim = self.model.get_sentence_embedding_dimension()


    def add_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.ids.append(ex["id"]) ; self.texts.append(ex["text"])


    def build(self):
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        embs = self.model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
        self.faiss_index = faiss.IndexFlatIP(self.emb_dim)
        self.faiss_index.add(embs.astype(np.float32))


    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(out_dir / "dense.faiss"))
        (out_dir / "ids.txt").write_text("\n".join(self.ids), encoding="utf-8")
        (out_dir / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")
        # BM25 pickling kept minimal; rebuild at load for simplicity


    @classmethod
    def load(cls, out_dir: Path, embed_model: str):
        idx = cls(embed_model)
        idx.faiss_index = faiss.read_index(str(out_dir / "dense.faiss"))
        idx.ids = (out_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        idx.texts = json.loads((out_dir / "texts.json").read_text(encoding="utf-8"))
        tokenized = [t.split() for t in idx.texts]
        from rank_bm25 import BM25Okapi
        idx.bm25 = BM25Okapi(tokenized)
        return idx