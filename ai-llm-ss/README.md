# ai-llm-ss — Minimal Speech-to-Text (ASR) from Scratch

This is a tiny end-to-end project: audio (.wav) + transcripts (.txt) → train a CTC model → serve a FastAPI `/transcribe` endpoint.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# put a few .wav files in data/raw/audio and matching .txt in data/raw/text
python .\scripts\prepare_data.py --in data\raw --out data\processed

# train (on CPU by default; use --device cuda if available)
python .\scripts\train_asr.py

# serve
uvicorn src.asr.api:app --host 127.0.0.1 --port 8001 --reload
```
