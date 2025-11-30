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
from models.lstm_asr import LSTMASRModel
from models.enhanced_asr import EnhancedASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from database.db_utils import ASRDatabase
from decoding.beam_search import BeamSearchDecoder
from decoding.lm_decoder import LMBeamSearchDecoder
from decoding.confidence import ConfidenceScorer
from utils.metrics import calculate_wer, calculate_cer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model_type: str = "transformer"  # transformer, lstm, enhanced
    num_epochs: int = 10
    batch_size: int = 16


@app.on_event("startup")
async def startup_event():
    """Initialize models and components on startup."""
    global tokenizer_cache, processor_cache
    
    logger.info("Initializing ASR API...")
    
    # Initialize tokenizer
    tokenizer_cache = Tokenizer()
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
    d_model = config.get('d_model', 256)
    
    # 尝试从checkpoint获取vocab_size，否则使用tokenizer的大小
    if 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    elif 'vocab_size' in config:
        vocab_size = config['vocab_size']
    else:
        vocab_size = len(tokenizer_cache)
        logger.warning(f"无法从checkpoint获取vocab_size，使用tokenizer大小: {vocab_size}")
    
    # 确定模型类型
    if model_type == "auto":
        # 尝试从checkpoint推断模型类型
        model_type = checkpoint.get('model_type', 'transformer')
    
    logger.info(f"加载模型: type={model_type}, input_dim={input_dim}, vocab_size={vocab_size}, d_model={d_model}")
    
    if model_type == "lstm":
        model = LSTMASRModel(input_dim=input_dim, vocab_size=vocab_size, hidden_size=d_model)
    elif model_type == "enhanced":
        model = EnhancedASRModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            use_contextual_embeddings=config.get('use_contextual_embeddings', True)
        )
    else:  # transformer
        model = ASRModel(input_dim=input_dim, vocab_size=vocab_size, d_model=d_model)
    
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
            
            # 尝试多种路径查找模型
            search_paths = [
                checkpoint_dir / f"{model_name}.pt",  # 直接路径
                checkpoint_dir / model_name / "best_model.pt",  # 训练目录下的best_model
                checkpoint_dir / model_name / f"{model_name}.pt",  # 训练目录下的同名文件
                checkpoint_dir / "test_run" / "best_model.pt" if model_name == "default" else None,  # 默认模型
                checkpoint_dir / "training_20251111-014704" / "best_model.pt" if model_name == "training_20251111-014704" else None,  # 最新训练模型
            ]
            
            # 如果model_name包含路径分隔符，直接使用
            if "/" in model_name or "\\" in model_name:
                checkpoint_path = checkpoint_dir / model_name.replace("/", "\\")
                if not checkpoint_path.exists():
                    checkpoint_path = checkpoint_dir / model_name.replace("/", "\\") / "best_model.pt"
            
            # 尝试所有搜索路径
            if checkpoint_path is None or not checkpoint_path.exists():
                for path in search_paths:
                    if path is not None and path.exists():
                        checkpoint_path = path
                        break
            
            # 如果还是找不到，尝试查找最新的best_model
            if checkpoint_path is None or not checkpoint_path.exists():
                best_models = list(checkpoint_dir.rglob("best_model.pt"))
                if best_models:
                    # 选择最新的
                    checkpoint_path = max(best_models, key=lambda p: p.stat().st_mtime)
                    logger.info(f"使用找到的最新模型: {checkpoint_path}")
            
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
        
        # Run inference
        with torch.no_grad():
            logits, output_lengths = model(features_tensor, lengths)
        
        # Decode
        if use_lm and lm_path and Path(lm_path).exists():
            # LM decoding
            vocab = [tokenizer_cache.id_to_token.get(i, "") for i in range(len(tokenizer_cache))]
            lm_decoder = LMBeamSearchDecoder(vocab=vocab, lm_path=lm_path)
            results = lm_decoder.decode(logits, output_lengths)
            if results:
                text = results[0]["text"]
                confidence = results[0].get("score", 0.0)
            else:
                text = ""
                confidence = 0.0
        elif use_beam_search:
            # Beam search decoding
            decoder = BeamSearchDecoder(
                vocab_size=len(tokenizer_cache),
                blank_token_id=tokenizer_cache.blank_token_id,
                beam_width=beam_width
            )
            results = decoder.decode_batch(logits, output_lengths, tokenizer_cache)
            if results:
                text = results[0].get("text_decoded", "")
                confidence = results[0].get("confidence", 0.0)
                logger.info(f"Beam search解码结果: '{text[:50]}...' (前50字符)")
                
                # 如果文本为空，显示token信息
                if not text or text.strip() == '':
                    # 获取预测的tokens
                    predictions = torch.argmax(logits, dim=-1)
                    pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
                    
                    # 统计token分布
                    from collections import Counter
                    token_counts = Counter(pred_tokens)
                    top_tokens = token_counts.most_common(10)
                    
                    # 尝试解码每个token ID对应的字符
                    token_chars = []
                    for token_id in pred_tokens[:50]:
                        if token_id in tokenizer_cache.idx_to_char:
                            char = tokenizer_cache.idx_to_char[token_id]
                            token_chars.append(f"{token_id}('{char}')")
                        else:
                            token_chars.append(f"{token_id}(<unk>)")
                    
                    text = f"[Beam Search输出分析]\n"
                    text += f"Token数量: {len(pred_tokens)}\n"
                    top_tokens_str = ', '.join([f'ID {tid}({tokenizer_cache.idx_to_char.get(tid, "<unk>")}) x {count}' for tid, count in top_tokens])
                    text += f"前10个最常见的token: {top_tokens_str}\n"
                    text += f"前50个token详情: {', '.join(token_chars[:50])}"
                    if len(pred_tokens) > 50:
                        text += f"\n... (共{len(pred_tokens)}个tokens)"
                    
                    logger.info(f"Beam search解码为空，显示token IDs和字符: {text[:200]}")
            else:
                # 获取预测的tokens
                predictions = torch.argmax(logits, dim=-1)
                pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
                
                # 统计token分布
                from collections import Counter
                token_counts = Counter(pred_tokens)
                top_tokens = token_counts.most_common(10)
                
                # 尝试解码每个token ID对应的字符
                token_chars = []
                for token_id in pred_tokens[:50]:
                    if token_id in tokenizer_cache.idx_to_char:
                        char = tokenizer_cache.idx_to_char[token_id]
                        token_chars.append(f"{token_id}('{char}')")
                    else:
                        token_chars.append(f"{token_id}(<unk>)")
                
                text = f"[Beam Search输出分析]\n"
                text += f"Token数量: {len(pred_tokens)}\n"
                top_tokens_str = ', '.join([f'ID {tid}({tokenizer_cache.idx_to_char.get(tid, "<unk>")}) x {count}' for tid, count in top_tokens])
                text += f"前10个最常见的token: {top_tokens_str}\n"
                text += f"前50个token详情: {', '.join(token_chars[:50])}"
                if len(pred_tokens) > 50:
                    text += f"\n... (共{len(pred_tokens)}个tokens)"
                
                confidence = 0.0
                logger.warning(f"Beam search解码结果为空，显示token IDs和字符: {text[:200]}")
        else:
            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)
            pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
            
            # 调试信息：检查token输出
            logger.info(f"预测的token数量: {len(pred_tokens)}")
            logger.info(f"所有token IDs: {pred_tokens}")
            
            # 获取logits的最大值（置信度）
            max_logits = torch.max(logits, dim=-1)[0]
            max_logits_values = max_logits[0, :output_lengths[0]].cpu().tolist()
            logger.info(f"前10个token的最大logits值: {max_logits_values[:10]}")
            
            # 解码
            text = tokenizer_cache.decode(pred_tokens)
            
            # 调试信息：检查解码结果
            logger.info(f"解码后的文本长度: {len(text)}")
            logger.info(f"解码后的文本: '{text[:50]}...' (前50字符)")
            
            # 如果文本为空或只有空白，显示token IDs作为输出
            if not text or text.strip() == '':
                logger.warning("解码结果为空，显示token IDs和详细信息")
                
                # 统计token分布
                from collections import Counter
                token_counts = Counter(pred_tokens)
                top_tokens = token_counts.most_common(10)
                
                # 尝试解码每个token ID对应的字符
                token_chars = []
                for token_id in pred_tokens[:50]:  # 只显示前50个
                    if token_id in tokenizer_cache.idx_to_char:
                        char = tokenizer_cache.idx_to_char[token_id]
                        token_chars.append(f"{token_id}('{char}')")
                    else:
                        token_chars.append(f"{token_id}(<unk>)")
                
                # 显示详细信息
                text = f"[模型输出分析]\n"
                text += f"Token数量: {len(pred_tokens)}\n"
                top_tokens_str = ', '.join([f'ID {tid}({tokenizer_cache.idx_to_char.get(tid, "<unk>")}) x {count}' for tid, count in top_tokens])
                text += f"前10个最常见的token: {top_tokens_str}\n"
                text += f"前50个token详情: {', '.join(token_chars[:50])}"
                if len(pred_tokens) > 50:
                    text += f"\n... (共{len(pred_tokens)}个tokens)"
                
                logger.info(f"显示token IDs和字符映射: {text[:200]}")
            
            # Compute confidence
            scorer = ConfidenceScorer()
            confidence = scorer.compute(logits, None, output_lengths)[0].item()
        
        # Filter by confidence if requested
        if min_confidence is not None and confidence < min_confidence:
            text = ""  # Reject low-confidence predictions
        
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

