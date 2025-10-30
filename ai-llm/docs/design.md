# Design Overview
- **ASR** via faster-whisper → transcripts JSONL.
- **Ingest** chunks transcripts for retrieval.
- **Index** builds BM25 (lexical) and FAISS (dense) hybrid.
- **RAG** retrieves, reranks, and answers with citations from transcripts.
- **API** exposes `/transcribe` and `/ask` for app integration.