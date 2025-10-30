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


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "Qwen2.5-0.5B-Instruct")
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "512"))