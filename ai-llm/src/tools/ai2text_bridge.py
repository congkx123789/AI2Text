from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
from faster_whisper import WhisperModel

# -------- Core API (safe to import from FastAPI) --------

_MODEL_CACHE: Dict[str, WhisperModel] = {}

def _get_model(size: str, device: Optional[str], compute: Optional[str]) -> WhisperModel:
    """
    Get or create WhisperModel instance with caching.
    
    Args:
        size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device ('cuda', 'cpu', 'auto', None)
        compute: Compute type ('float16', 'int8', 'int8_float16', None)
    
    Returns:
        WhisperModel instance
    """
    device = device or os.getenv("ASR_DEVICE", "auto")
    
    # Auto-detect device if not specified
    if device == "auto" or device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    Args:
        path: Path to audio file
        size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        lang: Language code (e.g., 'vi', 'en', None for auto-detect)
        device: Device ('cuda', 'cpu', 'auto', None)
        compute: Compute type ('float16', 'int8', None)
        beam_size: Beam size for decoding (default: 5)
        vad_filter: Enable VAD filtering (default: True)
    
    Returns:
        Dict with 'text', 'segments', 'language'
    """
    size = size or os.getenv("ASR_MODEL", "small")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    
    try:
        model = _get_model(size, device, compute)
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
