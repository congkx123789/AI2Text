from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.tools.ai2text_bridge import transcribe
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, VECTOR_DIR
from src.rag.indexer import HybridIndex
from src.rag.pipeline import RAGPipeline

class TranscribeRequest(BaseModel):
    audio_path: str

class AskRequest(BaseModel):
    query: str

app = FastAPI(title="ai-llm STT + RAG API")

# CORS: allow Swagger UI and any local client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}

# Lazy singletons so startup is fast and failures are visible
_index = None
_pipe = None

def _ensure_pipeline():
    global _index, _pipe
    if _pipe is None:
        # Load hybrid index from disk; raise clear error if missing
        vecdir = Path(VECTOR_DIR)
        if not vecdir.exists():
            raise HTTPException(status_code=500, detail=f"Vector store not found at {vecdir}")
        _index = HybridIndex.load(vecdir, EMBEDDING_MODEL)
        _pipe = RAGPipeline(_index, EMBEDDING_MODEL, RERANKER_MODEL)
    return _pipe

@app.post("/transcribe")
def api_transcribe(req: TranscribeRequest):
    try:
        # Normalize and validate path
        p = Path(req.audio_path).expanduser()
        if not p.is_absolute():
            # Interpret relative to server working dir
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Audio file not found: {p}")
        # Force CPU defaults if env not set (Windows-safe)
        os.environ.setdefault("CT2_FORCE_CPU", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        return transcribe(str(p))
    except HTTPException:
        raise
    except Exception as e:
        # Bubble a readable error instead of “Failed to fetch”
        raise HTTPException(status_code=500, detail=f"/transcribe error: {type(e).__name__}: {e}")

@app.post("/ask")
def api_ask(req: AskRequest):
    try:
        pipe = _ensure_pipeline()
        out = pipe.ask(req.query)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/ask error: {type(e).__name__}: {e}")
