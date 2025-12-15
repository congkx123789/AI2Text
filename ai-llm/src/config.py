from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()


DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "./vectorstore")).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models")).resolve()


ASR_MODEL = os.getenv("ASR_MODEL", "small")
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")
ASR_COMPUTE = os.getenv("ASR_COMPUTE", "float16")

# Fine-tuned model paths (nếu có)
ASR_FINETUNED_MODEL = os.getenv("ASR_FINETUNED_MODEL", None)  # e.g., "./models/finetuned/whisper-mixed"
GEN_FINETUNED_MODEL = os.getenv("GEN_FINETUNED_MODEL", None)  # e.g., "./models/finetuned/qwen-mixed"

# Use fine-tuned models if available
if ASR_FINETUNED_MODEL and Path(ASR_FINETUNED_MODEL).exists():
    ASR_MODEL = ASR_FINETUNED_MODEL

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "Qwen2.5-0.5B-Instruct")
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "512"))

# Use fine-tuned LLM if available
if GEN_FINETUNED_MODEL and Path(GEN_FINETUNED_MODEL).exists():
    GEN_MODEL = GEN_FINETUNED_MODEL

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # gemini-2.5-flash (fast) or gemini-2.5-pro (better quality)

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")  # "qwen" (local) or "gemini" (API)