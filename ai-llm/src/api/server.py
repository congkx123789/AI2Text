from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.tools.ai2text_bridge import transcribe
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, VECTOR_DIR
from src.rag.indexer import HybridIndex
from src.rag.pipeline import RAGPipeline

class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file on server")

class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    segments: list = Field(..., description="List of transcription segments with timestamps")
    language: str = Field(..., description="Detected language code")

class AskRequest(BaseModel):
    query: str = Field(..., description="Question to ask")
    top_k: Optional[int] = Field(5, description="Number of top results to retrieve")

class Citation(BaseModel):
    id: str
    text: str

class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    contexts: list[Citation] = Field(..., description="Source citations")

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    index_available: bool

app = FastAPI(
    title="AI-LLM API",
    description="Speech-to-Text and RAG API for transcription and question answering",
    version="1.0.0"
)

# CORS: allow Swagger UI and any local client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
def root():
    """Root endpoint - API information"""
    return {"ok": True, "docs": "/docs", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint - check if models and index are loaded"""
    index_available = Path(VECTOR_DIR).exists()
    models_loaded = _check_pipeline_status()
    status = "healthy" if (index_available and models_loaded) else "degraded"
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        index_available=index_available
    )

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

def _check_pipeline_status():
    """Check if pipeline is loaded without raising errors"""
    global _pipe
    try:
        if _pipe is None:
            # Try to load it
            _ensure_pipeline()
        return True
    except:
        return False

@app.post("/transcribe", response_model=TranscribeResponse)
def api_transcribe(req: TranscribeRequest):
    """
    Transcribe audio file from server path.
    
    - **audio_path**: Path to audio file on the server
    """
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
        result = transcribe(str(p))
        return TranscribeResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {type(e).__name__}: {e}")

@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def api_transcribe_upload(file: UploadFile = File(...)):
    """
    Upload and transcribe audio file.
    
    Supports common audio formats: wav, mp3, m4a, flac, etc.
    """
    try:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Force CPU defaults
        os.environ.setdefault("CT2_FORCE_CPU", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        
        # Transcribe
        result = transcribe(tmp_path)
        
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return TranscribeResponse(**result)
    except Exception as e:
        # Cleanup on error
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Transcription error: {type(e).__name__}: {e}")

@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    """
    Ask a question using RAG pipeline.
    
    - **query**: Your question
    - **top_k**: Number of top results to retrieve (default: 5)
    """
    try:
        pipe = _ensure_pipeline()
        out = pipe.ask(req.query, k=req.top_k)
        # Convert contexts to Citation models
        citations = [Citation(id=c.get("id", ""), text=c.get("text", "")) for c in out.get("contexts", [])]
        return AskResponse(answer=out.get("answer", ""), contexts=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {type(e).__name__}: {e}")
