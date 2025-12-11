from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Union
from faster_whisper import WhisperModel

# -------- Core API (safe to import from FastAPI) --------

_MODEL_CACHE: Dict[str, Any] = {}
_TRANSFORMERS_MODEL_CACHE: Dict[str, Any] = {}

def _is_local_path(model_path: str) -> bool:
    """Check if model_path is a local path (not a size string)"""
    if model_path in ["tiny", "base", "small", "medium", "large"]:
        return False
    path = Path(model_path)
    return path.exists() and path.is_dir()


def _is_ct2_model(model_path: str | Path) -> bool:
    """Check if model path is CTranslate2 format (has model.bin)"""
    path = Path(model_path)
    return (path / "model.bin").exists() or (path / "model.bin.index.json").exists()


def _is_huggingface_model(model_path: str | Path) -> bool:
    """Check if model path is HuggingFace format (has config.json and model files)"""
    path = Path(model_path)
    has_config = (path / "config.json").exists()
    has_model = (
        (path / "model.safetensors").exists() or
        (path / "pytorch_model.bin").exists() or
        (path / "model.bin").exists()
    )
    return has_config and has_model

def _get_model(size: str, device: Optional[str], compute: Optional[str]) -> Union[WhisperModel, Any]:
    """
    Get or create WhisperModel instance with caching.
    Supports both faster_whisper (for base models) and transformers (for fine-tuned models).
    
    Args:
        size: Model size ('tiny', 'base', 'small', 'medium', 'large') or local path to fine-tuned model
        device: Device ('cuda', 'cpu', 'auto', None)
        compute: Compute type ('float16', 'int8', 'int8_float16', None)
    
    Returns:
        WhisperModel instance (faster_whisper) or tuple (processor, model) from transformers
    """
    device = device or os.getenv("ASR_DEVICE", "auto")
    
    # Auto-detect device if not specified
    if device == "auto" or device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if this is a local model path
    if _is_local_path(size):
        # Check if it's CTranslate2 format (for faster-whisper)
        if _is_ct2_model(size):
            # Use faster-whisper with CTranslate2 model
            if compute is None:
                compute = os.getenv("ASR_COMPUTE")
            if compute is None:
                compute = "float16" if device == "cuda" else "int8"
            
            key = f"ct2:{size}:{device}:{compute}"
            if key not in _MODEL_CACHE:
                print(f"[Whisper] Loading CTranslate2 model from: {size}")
                try:
                    _MODEL_CACHE[key] = WhisperModel(str(size), device=device, compute_type=compute)
                    print(f"[Whisper] CTranslate2 model loaded successfully")
                except Exception as e:
                    print(f"[Whisper] Error loading CTranslate2 model: {e}")
                    raise
            return _MODEL_CACHE[key]
        
        # Check if it's HuggingFace format (use transformers)
        elif _is_huggingface_model(size):
            # Use transformers for HuggingFace models
            key = f"transformers:{size}:{device}"
            if key not in _TRANSFORMERS_MODEL_CACHE:
                print(f"[Whisper] Loading HuggingFace model from: {size}")
                try:
                    from transformers import WhisperProcessor, WhisperForConditionalGeneration
                    import torch
                    
                    processor = WhisperProcessor.from_pretrained(size)
                    model = WhisperForConditionalGeneration.from_pretrained(size)
                    model.eval()
                    
                    if device == "cuda" and torch.cuda.is_available():
                        model = model.cuda()
                        print(f"[Whisper] HuggingFace model loaded on GPU")
                    else:
                        print(f"[Whisper] HuggingFace model loaded on CPU")
                    
                    _TRANSFORMERS_MODEL_CACHE[key] = (processor, model)
                    print(f"[Whisper] HuggingFace model loaded successfully")
                except Exception as e:
                    print(f"[Whisper] Error loading HuggingFace model: {e}")
                    raise
            return _TRANSFORMERS_MODEL_CACHE[key]
        
        else:
            # Unknown format, try as CTranslate2 first, then HuggingFace
            print(f"[Whisper] Unknown model format, trying CTranslate2 first: {size}")
            try:
                if compute is None:
                    compute = os.getenv("ASR_COMPUTE")
                if compute is None:
                    compute = "float16" if device == "cuda" else "int8"
                
                key = f"ct2:{size}:{device}:{compute}"
                _MODEL_CACHE[key] = WhisperModel(str(size), device=device, compute_type=compute)
                print(f"[Whisper] Loaded as CTranslate2 model")
                return _MODEL_CACHE[key]
            except:
                print(f"[Whisper] Not CTranslate2, trying HuggingFace...")
                # Fallback to HuggingFace
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                import torch
                processor = WhisperProcessor.from_pretrained(size)
                model = WhisperForConditionalGeneration.from_pretrained(size)
                model.eval()
                if device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()
                key = f"transformers:{size}:{device}"
                _TRANSFORMERS_MODEL_CACHE[key] = (processor, model)
                return _TRANSFORMERS_MODEL_CACHE[key]
    
    # Use faster_whisper for base models (size strings)
    # Pick sane default compute type if not set
    if compute is None:
        compute = os.getenv("ASR_COMPUTE")
    if compute is None:
        # Use float16 for CUDA (faster, less memory), int8 for CPU
        compute = "float16" if device == "cuda" else "int8"
    
    key = f"{size}:{device}:{compute}"
    if key not in _MODEL_CACHE:
        print(f"[Whisper] Loading model: size={size}, device={device}, compute={compute}")
        try:
            _MODEL_CACHE[key] = WhisperModel(size, device=device, compute_type=compute)
            print(f"[Whisper] Model loaded successfully: {key}")
        except Exception as e:
            print(f"[Whisper] Error loading model: {e}")
            # Fallback to CPU if CUDA fails
            if device == "cuda":
                print(f"[Whisper] Falling back to CPU")
                key_cpu = f"{size}:cpu:{compute}"
                if key_cpu not in _MODEL_CACHE:
                    _MODEL_CACHE[key_cpu] = WhisperModel(size, device="cpu", compute_type=compute)
                return _MODEL_CACHE[key_cpu]
            raise
    return _MODEL_CACHE[key]

def transcribe(
    path: str | Path,
    size: str = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
    compute: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe a single audio file using Whisper model.
    Supports both faster_whisper (base models) and transformers (fine-tuned models).
    
    Args:
        path: Path to audio file
        size: Model size ('tiny', 'base', 'small', 'medium', 'large') or local path to fine-tuned model
        lang: Language code (e.g., 'vi', 'en', None for auto-detect)
        device: Device ('cuda', 'cpu', 'auto', None)
        compute: Compute type ('float16', 'int8', None) - only for faster_whisper
        beam_size: Beam size for decoding (default: 5)
        vad_filter: Enable VAD filtering (default: True) - only for faster_whisper
    
    Returns:
        Dict with 'text', 'segments', 'language'
    """
    size = size or os.getenv("ASR_MODEL", "small")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    
    try:
        model_or_tuple = _get_model(size, device, compute)
        
        # Check if it's a transformers model (tuple) or faster_whisper model
        if isinstance(model_or_tuple, tuple):
            # Fine-tuned model using transformers
            processor, model = model_or_tuple
            import librosa
            import torch
            
            # Load audio
            audio, sr = librosa.load(str(p), sr=16000)
            
            # Process audio
            input_features = processor.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features,
                    max_length=448,
                    language=None if lang is None else lang,
                    task="transcribe",
                    num_beams=beam_size,
                    do_sample=False
                )
            
            # Decode
            prediction = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # For transformers, we don't have segment timestamps easily
            # Return full text as single segment
            return {
                "text": prediction.strip(),
                "segments": [{
                    "start": 0.0,
                    "end": len(audio) / sr if sr > 0 else 0.0,
                    "text": prediction.strip()
                }],
                "language": lang or "unknown"
            }
        else:
            # Base model using faster_whisper
            model = model_or_tuple
        seg_iter, info = model.transcribe(
            str(p), 
            language=lang, 
            vad_filter=vad_filter, 
            beam_size=beam_size
        )
        segs, texts = [], []
        for s in seg_iter:
            segs.append({
                "start": float(s.start or 0), 
                "end": float(s.end or 0), 
                "text": s.text.strip()
            })
            texts.append(s.text.strip())
        
        return {
            "text": " ".join(texts).strip(), 
            "segments": segs, 
            "language": info.language
        }
    except Exception as e:
        print(f"[Whisper] Transcription error: {e}")
        raise

def transcribe_many(files: Iterable[str | Path], **kw) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in files:
        f = Path(f)
        out[f.stem] = transcribe(f, **kw)
    return out

# -------- CLI (runs only when called as a script) --------

def main() -> None:
    import argparse, json
    ap = argparse.ArgumentParser(description="Transcribe a JSONL manifest with faster-whisper.")
    ap.add_argument("--manifest", required=True, help="Input JSONL lines: {id, audio}")
    ap.add_argument("--out", required=True, help="Output JSONL lines: {id, text, segments, language}")
    ap.add_argument("--model", default=os.getenv("ASR_MODEL", "small"))
    ap.add_argument("--lang", default=None)
    ap.add_argument("--device", default=os.getenv("ASR_DEVICE"))
    ap.add_argument("--compute", default=os.getenv("ASR_COMPUTE"))
    args = ap.parse_args()

    in_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fi, out_path.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            ex = json.loads(line)
            res = transcribe(
                ex["audio"],
                size=args.model,
                lang=args.lang,
                device=args.device,
                compute=args.compute,
            )
            fo.write(json.dumps({"id": ex["id"], **res}, ensure_ascii=False) + "\n")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
