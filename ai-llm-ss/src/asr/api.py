from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchaudio
import json
import os
from pathlib import Path
from typing import Optional
from .model import CRNNCTC
from .features import wav_to_logmelspec, ensure_mono16k
from .decode import greedy_decode

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "vocab.json"
MODEL_PATH = PROJECT_ROOT / "data" / "results" / "asr_ctc.pt"

app = FastAPI(
    title="ASR CTC API",
    description="Automatic Speech Recognition API using CTC model",
    version="1.0.0"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocabulary
try:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        VOCAB = json.load(f)
    ITOS = {i: c for i, c in enumerate(VOCAB)}
    print(f"Loaded vocabulary with {len(VOCAB)} tokens")
except FileNotFoundError:
    raise RuntimeError(f"Vocabulary file not found at {VOCAB_PATH}")

# Initialize and load model
MODEL = None
try:
    MODEL = CRNNCTC(n_mels=80, vocab_size=len(VOCAB))
    if MODEL_PATH.exists():
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. API will start but transcription will fail.")
    MODEL.to(device).eval()
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None


class TranscriptionResponse(BaseModel):
    text: str
    duration: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    vocab_size: int


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "ASR CTC API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "transcribe": "/transcribe (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "model_not_loaded",
        device=device,
        model_loaded=MODEL is not None,
        vocab_size=len(VOCAB)
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "device": device,
        "vocab_size": len(VOCAB),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": "CRNNCTC"
    }


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    
    Accepts audio files in formats supported by torchaudio (WAV, MP3, FLAC, etc.)
    Audio will be automatically resampled to 16kHz mono if needed.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists at data/results/asr_ctc.pt"
        )
    
    # Validate file type
    if not file.content_type or not any(
        file.content_type.startswith(prefix) 
        for prefix in ["audio/", "video/"]
    ):
        # Still try to process it, torchaudio might support it
        pass
    
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            wav, sr = torchaudio.load(tmp_path)
            duration = wav.shape[-1] / sr if wav.numel() > 0 else 0.0
            
            # Preprocess audio
            wav, sr = ensure_mono16k(wav, sr)
            feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)  # (1,T,F)
            
            # Transcribe
            with torch.no_grad():
                logits, lens = MODEL(feats, torch.tensor([feats.shape[1]], device=device))
            text = greedy_decode(logits.cpu(), ITOS)[0]
            
            return TranscriptionResponse(text=text, duration=duration)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing audio file: {str(e)}"
        )
