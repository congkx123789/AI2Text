"""
REST API for ASR system.

FastAPI application providing endpoints for:
- Transcribing audio files
- Training models
- Evaluating models
- Managing models and experiments
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database is optional; API can run inference without it
try:
    from database.db_utils import ASRDatabase  # type: ignore
except Exception as e:
    ASRDatabase = None  # Fallback if module not available
    logger.warning(f"database.db_utils not available, skipping DB features: {e}")

from decoding.beam_search import BeamSearchDecoder
from decoding.lm_decoder import LMBeamSearchDecoder
from decoding.confidence import ConfidenceScorer
from utils.metrics import calculate_wer, calculate_cer

app = FastAPI(
    title="Vietnamese ASR API",
    description="REST API for Vietnamese Speech-to-Text system",
    version="1.0.0"
)

# 配置CORS以支持前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models_cache = {}
tokenizer_cache = None
processor_cache = None


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    model_name: Optional[str] = "default"
    use_beam_search: bool = True
    beam_width: int = 5
    use_lm: bool = False
    lm_path: Optional[str] = None
    min_confidence: Optional[float] = 0.5


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    confidence: float
    model_name: str
    processing_time: float


class TrainingRequest(BaseModel):
    """Request model for training."""
    config_path: str
    model_type: str = "transformer"  # Only transformer architecture supported
    num_epochs: int = 10
    batch_size: int = 16


@app.on_event("startup")
async def startup_event():
    """Initialize models and components on startup."""
    global tokenizer_cache, processor_cache
    
    logger.info("Initializing ASR API...")
    
    # Initialize tokenizer (use SentencePiece to match training)
    sp_path = Path("models/tokenizer_vi_en_3500.model")
    if not sp_path.exists():
        logger.warning(f"SentencePiece model not found at {sp_path}, falling back to char Tokenizer")
        tokenizer_cache = Tokenizer()
    else:
        tokenizer_cache = SentencePieceTokenizer(str(sp_path))
    logger.info(f"Tokenizer initialized with vocab size: {len(tokenizer_cache)}")
    
    # Initialize audio processor
    processor_cache = AudioProcessor(sample_rate=16000, n_mels=80)
    logger.info("Audio processor initialized")
    
    logger.info("API ready!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vietnamese ASR API",
        "version": "1.0.0",
        "endpoints": {
            "transcribe": "POST /transcribe",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models_cache),
        "tokenizer_ready": tokenizer_cache is not None,
        "processor_ready": processor_cache is not None
    }


def load_model(model_path: str, model_type: str = "transformer"):
    """Load ASR model from checkpoint."""
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    # Since these are user-trained models, we trust the source
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    # 从config或checkpoint中获取参数
    input_dim = config.get('n_mels', 80)
    d_model = config.get('d_model', 1024)
    num_encoder_layers = config.get('num_encoder_layers', 24)
    num_decoder_layers = config.get('num_decoder_layers', 6)
    num_heads = config.get('num_heads', 16)
    d_ff = config.get('d_ff', 4096)
    dropout = config.get('dropout', 0.1)
    use_gc = config.get('use_gradient_checkpointing', True)
    
    # 尝试从checkpoint获取vocab_size，否则使用tokenizer的大小
    if 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    elif 'vocab_size' in config:
        vocab_size = config['vocab_size']
    else:
        vocab_size = len(tokenizer_cache)
        logger.warning(f"无法从checkpoint获取vocab_size，使用tokenizer大小: {vocab_size}")
    
    # Only Transformer architecture supported
    model_type = "transformer"
    logger.info(f"加载模型: type={model_type}, input_dim={input_dim}, vocab_size={vocab_size}, d_model={d_model}")
    
    # Create Transformer model
    model = ASRModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        use_gradient_checkpointing=use_gc
    )
    
    # 加载模型权重
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("模型权重加载成功（严格模式）")
        except Exception as e:
            logger.warning(f"严格加载失败，尝试非严格模式: {e}")
            model.load_state_dict(state_dict, strict=False)
            logger.info("模型权重加载成功（非严格模式）")
    else:
        raise ValueError("Checkpoint中没有找到模型权重")
    
    model.eval()
    
    return model


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    model_name: str = "default",
    use_beam_search: bool = True,
    beam_width: int = 5,
    use_lm: bool = False,
    lm_path: Optional[str] = None,
    min_confidence: Optional[float] = None
):
    """
    Transcribe audio file to text.
    
    Args:
        audio: Audio file (WAV, MP3, FLAC)
        model_name: Model name to use
        use_beam_search: Use beam search decoding
        beam_width: Beam search width
        use_lm: Use language model (KenLM)
        lm_path: Path to language model file
        min_confidence: Minimum confidence threshold
        
    Returns:
        Transcription result with text and confidence
    """
    import time
    # 音频读取由processor_cache处理，无需额外依赖
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily (preserve original extension)
        file_extension = Path(audio.filename).suffix if audio.filename else '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await audio.read()
            tmp_path.write_bytes(content)
        
        # Use shared audio processor for consistent, fast decoding (handles mp3 via torchaudio)
        try:
            audio_data, sample_rate = processor_cache.load_audio(str(tmp_path))
        except Exception as audio_error:
            logger.error(f"Failed to load audio via processor: {audio_error}")
            raise HTTPException(
                status_code=400,
                detail=f"无法读取音频文件。支持的格式: WAV, MP3, M4A, FLAC, OGG等。错误: {str(audio_error)}"
            )
        
        # Extract mel spectrogram features
        features = processor_cache.extract_mel_spectrogram(audio_data)
        # features shape: (n_mels, time_frames)
        # Transpose to (time_frames, n_mels) for model input
        features = features.T  # (time_frames, n_mels)
        features_tensor = torch.from_numpy(features).unsqueeze(0).float()
        lengths = torch.tensor([features.shape[0]])
        
        # Load model (lazy loading)
        if model_name not in models_cache:
            # 获取项目根目录
            api_dir = Path(__file__).parent
            project_root = api_dir.parent
            checkpoint_dir = project_root / "checkpoints"
            checkpoint_path = None
            
            # 优先使用全局 best_model.pt（best checkpoint hiện tại）
            default_best = checkpoint_dir / "best_model.pt"
            if default_best.exists():
                checkpoint_path = default_best
                logger.info(f"优先使用全局 best checkpoint: {checkpoint_path}")
            
            # 如果model_name包含路径分隔符，直接使用
            if checkpoint_path is None and ("/" in model_name or "\\" in model_name):
                checkpoint_path = checkpoint_dir / model_name.replace("/", "\\")
                if not checkpoint_path.exists():
                    checkpoint_path = checkpoint_dir / model_name.replace("/", "\\") / "best_model.pt"
            
            # 如果还未找到，尝试多种路径
            if checkpoint_path is None or not checkpoint_path.exists():
                search_paths = [
                    checkpoint_dir / f"{model_name}.pt",
                    checkpoint_dir / model_name / "best_model.pt",
                    checkpoint_dir / model_name / f"{model_name}.pt",
                ]
                for path in search_paths:
                    if path is not None and path.exists():
                        checkpoint_path = path
                        break
            
            # 最后兜底：选择 checkpoints/ 下最新的 best_model.pt
            if checkpoint_path is None or not checkpoint_path.exists():
                best_models = list(checkpoint_dir.rglob("best_model.pt"))
                if best_models:
                    checkpoint_path = max(best_models, key=lambda p: p.stat().st_mtime)
                    logger.info(f"兜底使用最新的 best_model: {checkpoint_path}")
            
            if checkpoint_path and checkpoint_path.exists():
                try:
                    # 尝试从checkpoint获取模型类型
                    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
                    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                    model_type = checkpoint.get('model_type', 'transformer')
                    
                    model = load_model(str(checkpoint_path), model_type=model_type)
                    models_cache[model_name] = model
                    logger.info(f"成功加载模型: {model_name} from {checkpoint_path}")
                except Exception as e:
                    logger.error(f"加载模型失败 {checkpoint_path}: {e}", exc_info=True)
                    # 尝试使用默认transformer类型
                    try:
                        logger.info("尝试使用transformer类型重新加载...")
                        model = load_model(str(checkpoint_path), model_type='transformer')
                        models_cache[model_name] = model
                        logger.info(f"使用transformer类型成功加载模型: {model_name}")
                    except Exception as e2:
                        logger.error(f"重新加载也失败: {e2}", exc_info=True)
                        raise HTTPException(
                            status_code=500, 
                            detail=f"加载模型失败: {str(e)}. 详细信息: {str(e2)}"
                        )
            else:
                available_models = await list_models()
                model_names = [m["name"] for m in available_models["models"]]
                raise HTTPException(
                    status_code=404, 
                    detail=f"模型 '{model_name}' 未找到。可用模型: {', '.join(model_names[:5])}"
                )
        
        model = models_cache[model_name]
        
        
        # Run inference using autoregressive generation
        with torch.no_grad():
            generated = model.generate(
                features_tensor,
                lengths=lengths,
                language_ids=None,
                max_len=512,
                sos_token_id=getattr(tokenizer_cache, 'sos_token_id', 2),
                eos_token_id=getattr(tokenizer_cache, 'eos_token_id', 3),
                pad_token_id=getattr(tokenizer_cache, 'pad_token_id', 0),
                temperature=1.0
            )
        
        # Decode generated tokens
        gen_seq = generated[0].cpu().tolist()
        decoded_tokens = []
        sos_id = getattr(tokenizer_cache, 'sos_token_id', 2)
        eos_id = getattr(tokenizer_cache, 'eos_token_id', 3)
        pad_id = getattr(tokenizer_cache, 'pad_token_id', 0)
        for t in gen_seq:
            if t == eos_id:
                break
            if t not in (sos_id, pad_id):
                decoded_tokens.append(t)
        text = tokenizer_cache.decode(decoded_tokens)
        confidence = 0.0
        
        # Filter by confidence if requested
        if min_confidence is not None and confidence < min_confidence:
            text = ""
        
        processing_time = time.time() - start_time
        
        # Cleanup
        tmp_path.unlink()
        
        return TranscriptionResponse(
            text=text,
            confidence=float(confidence),
            model_name=model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    # 获取项目根目录（api目录的父目录）
    api_dir = Path(__file__).parent
    project_root = api_dir.parent
    checkpoint_dir = project_root / "checkpoints"
    
    models = []
    
    logger.info(f"搜索模型目录: {checkpoint_dir}")
    
    if checkpoint_dir.exists():
        # 递归搜索所有.pt文件
        for checkpoint_file in checkpoint_dir.rglob("*.pt"):
            # 获取相对路径作为模型名称
            try:
                rel_path = checkpoint_file.relative_to(checkpoint_dir)
                model_name = str(rel_path).replace("\\", "/").replace(".pt", "")
                
                # 如果是best_model，使用父目录名
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
                logger.info(f"找到模型: {model_name} at {checkpoint_file}")
            except Exception as e:
                logger.warning(f"无法读取模型文件 {checkpoint_file}: {e}")
    else:
        logger.warning(f"模型目录不存在: {checkpoint_dir}")
    
    # 按路径排序，最新的在前
    models.sort(key=lambda x: x["path"], reverse=True)
    
    logger.info(f"总共找到 {len(models)} 个模型")
    
    return {
        "models": models,
        "loaded": list(models_cache.keys()),
        "total": len(models)
    }


@app.post("/models/load")
async def load_model_endpoint(model_path: str, model_name: str = "default"):
    """Load a model into cache."""
    try:
        model = load_model(model_path, model_type="transformer")
        models_cache[model_name] = model
        return {"status": "loaded", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from cache."""
    if model_name in models_cache:
        del models_cache[model_name]
        return {"status": "unloaded", "model_name": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not in cache")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

