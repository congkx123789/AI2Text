"""
REST API v2 for ASR system.
Improved version supporting ASRModelWithTimestamps and BPE tokenizer.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import numpy as np
from pathlib import Path
import tempfile
import yaml
import sys
import time
import logging

sys.path.append(str(Path(__file__).parent.parent))

from api.asr_service import ASRService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese ASR API v2",
    description="REST API for Vietnamese Speech-to-Text system with timestamp support",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
asr_services: Dict[str, ASRService] = {}


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    timestamps: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    model_name: str
    processing_time: float


class BatchTranscriptionRequest(BaseModel):
    """Request model for batch transcription."""
    audio_paths: List[str]
    return_timestamps: bool = False
    language_id: Optional[int] = None


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch transcription."""
    results: List[Dict[str, Any]]
    total_time: float


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("ASR API v2 initialized")
    logger.info("Use /load_model endpoint to load a model")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vietnamese ASR API v2",
        "version": "2.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "transcribe_batch": "POST /transcribe_batch",
            "load_model": "POST /load_model",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(asr_services.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/load_model")
async def load_model_endpoint(
    checkpoint_path: str,
    model_name: str = "default",
    device: str = "cuda"
):
    """Load a model into memory.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        model_name: Name to assign to the model
        device: Device to use ('cuda' or 'cpu')
    """
    try:
        # Resolve checkpoint path
        project_root = Path(__file__).parent.parent
        checkpoint_path_resolved = project_root / checkpoint_path
        
        if not checkpoint_path_resolved.exists():
            # Try relative to checkpoints directory
            checkpoint_path_resolved = project_root / "checkpoints" / checkpoint_path
            if not checkpoint_path_resolved.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load service
        service = ASRService(str(checkpoint_path_resolved), device=device)
        asr_services[model_name] = service
        
        logger.info(f"Model '{model_name}' loaded successfully from {checkpoint_path_resolved}")
        
        return {
            "status": "loaded",
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path_resolved),
            "device": device
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    model_name: str = Query("default", description="Model name to use"),
    return_timestamps: bool = Query(False, description="Return word-level timestamps"),
    language_id: Optional[int] = Query(None, description="Language ID (0=Vietnamese, 1=English)")
):
    """
    Transcribe audio file to text.
    
    Args:
        audio: Audio file (WAV, MP3, FLAC, etc.)
        model_name: Name of loaded model
        return_timestamps: If True, return word-level timestamps
        language_id: Language ID (0=Vietnamese, 1=English), None for auto-detect
        
    Returns:
        Transcription result with text and optional timestamps
    """
    start_time = time.time()
    
    if model_name not in asr_services:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Use /load_model endpoint first. "
                   f"Available models: {list(asr_services.keys())}"
        )
    
    service = asr_services[model_name]
    
    try:
        # Save uploaded file temporarily
        file_extension = Path(audio.filename).suffix if audio.filename else '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await audio.read()
            tmp_path.write_bytes(content)
        
        # Auto preprocess audio (làm sạch tự động và đảm bảo format đúng)
        # AI2Text yêu cầu: 16kHz mono WAV, sau đó convert sang mel spectrogram (n_fft=400, hop=160, n_mels=80)
        try:
            import sys
            audio_processing_path = Path(__file__).parent.parent.parent / "audio_processing"
            if audio_processing_path.exists():
                sys.path.insert(0, str(audio_processing_path.parent))
                from audio_processing import auto_preprocess_audio
                logger.info(f"[AI2Text] Auto preprocessing audio: {tmp_path}")
                # Preprocess và đảm bảo format: 16kHz mono WAV, PCM 16-bit
                tmp_path = Path(auto_preprocess_audio(
                    str(tmp_path), 
                    sample_rate=16000,
                    model_type='ai2text'  # Format cho transformer ASR
                ))
                logger.info(f"[AI2Text] Preprocessed audio: {tmp_path} (16kHz mono WAV, ready for mel spec)")
        except Exception as e:
            logger.warning(f"[AI2Text] Auto preprocessing failed: {e}, using original audio")
        
        # Transcribe
        result = service.transcribe(
            str(tmp_path),
            return_timestamps=return_timestamps,
            language_id=language_id
        )
        
        processing_time = time.time() - start_time
        
        # Cleanup
        tmp_path.unlink()
        
        return TranscriptionResponse(
            text=result['text'],
            timestamps=result.get('timestamps'),
            confidence=result.get('confidence'),
            model_name=model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe_batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    request: BatchTranscriptionRequest,
    model_name: str = Query("default", description="Model name to use")
):
    """
    Transcribe multiple audio files.
    
    Args:
        request: Batch transcription request with audio paths
        model_name: Name of loaded model
        
    Returns:
        List of transcription results
    """
    start_time = time.time()
    
    if model_name not in asr_services:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Use /load_model endpoint first."
        )
    
    service = asr_services[model_name]
    
    try:
        results = service.transcribe_batch(request.audio_paths)
        total_time = time.time() - start_time
        
        return BatchTranscriptionResponse(
            results=results,
            total_time=total_time
        )
    except Exception as e:
        logger.error(f"Batch transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models and checkpoints."""
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "checkpoints"
    
    models = []
    
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.rglob("*.pt"):
            try:
                rel_path = checkpoint_file.relative_to(checkpoint_dir)
                model_name = str(rel_path).replace("\\", "/").replace(".pt", "")
                
                if checkpoint_file.name == "best_model.pt":
                    parent_name = checkpoint_file.parent.name
                    if parent_name != "checkpoints":
                        model_name = parent_name
                
                file_size = checkpoint_file.stat().st_size / (1024 * 1024)
                models.append({
                    "name": model_name,
                    "path": str(checkpoint_file.relative_to(project_root)),
                    "size_mb": round(file_size, 2),
                    "full_path": str(checkpoint_file.absolute())
                })
            except Exception as e:
                logger.warning(f"Error reading checkpoint {checkpoint_file}: {e}")
    
    models.sort(key=lambda x: x["path"], reverse=True)
    
    return {
        "models": models,
        "loaded": list(asr_services.keys()),
        "total": len(models)
    }


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    if model_name in asr_services:
        del asr_services[model_name]
        return {"status": "unloaded", "model_name": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not in cache")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

