from __future__ import annotations
import argparse
from pathlib import Path
from src.config import EMBEDDING_MODEL, VECTOR_DIR
from src.rag.ingest import transcripts_to_chunks
from src.rag.indexer import HybridIndex


"""Transcripts JSONL → chunk JSONL → Hybrid index (BM25 + FAISS)"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="transcripts", required=True)
    ap.add_argument("--out", dest="out_dir", default=str(VECTOR_DIR))
    args = ap.parse_args()


    chunks = Path("data/processed/chunks.jsonl")
    transcripts_to_chunks(Path(args.transcripts), chunks)


    idx = HybridIndex(EMBEDDING_MODEL)
    idx.add_jsonl(chunks)
    idx.build()
    idx.save(Path(args.out_dir))
    print("Index built at:", args.out_dir)


if __name__ == "__main__":
    main()