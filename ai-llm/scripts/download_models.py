from __future__ import annotations
import os
from pathlib import Path
from huggingface_hub import snapshot_download

"""
Download all models needed for offline use of ai-llm (speech→text + RAG + LLM).
Run this once while online:
    python scripts/download_models.py
Then you can disconnect and run the project fully offline.
"""

def main():
    base_dir = Path("models/base")
    base_dir.mkdir(parents=True, exist_ok=True)

    MODELS = {
        "whisper-small": "openai/whisper-small",  # ASR
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",  # embeddings
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # reranker
        "generator": "Qwen/Qwen2.5-0.5B-Instruct",  # LLM
    }

    for name, repo in MODELS.items():
        target = base_dir / name
        print(f"⬇️ Downloading {repo} → {target}")
        snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False)
        print(f"✅ Saved {name} at {target}\n")

    print("🎯 All models downloaded successfully.")
    print("Now you can set these paths in your .env file:")
    print("""
ASR_MODEL=./models/base/whisper-small
EMBEDDING_MODEL=./models/base/embedder
RERANKER_MODEL=./models/base/reranker
GEN_MODEL=./models/base/generator
""")

if __name__ == "__main__":
    main()
