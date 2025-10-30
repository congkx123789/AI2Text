from fastapi import FastAPI, UploadFile, File
import torch, torchaudio, json
from .model import CRNNCTC
from .features import wav_to_logmelspec, ensure_mono16k
from .decode import greedy_decode

app = FastAPI(title="ASR CTC API")
device = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = json.load(open("data/processed/vocab.json","r",encoding="utf-8"))
ITOS = {i:c for i,c in enumerate(VOCAB)}
MODEL = CRNNCTC(n_mels=80, vocab_size=len(VOCAB))
try:
    MODEL.load_state_dict(torch.load("data/results/asr_ctc.pt", map_location=device))
except FileNotFoundError:
    pass  # allow API to start before training
MODEL.to(device).eval()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    wav, sr = torchaudio.load(file.file)
    wav, sr = ensure_mono16k(wav, sr)
    feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)  # (1,T,F)
    with torch.no_grad():
        logits, lens = MODEL(feats, torch.tensor([feats.shape[1]], device=device))
    text = greedy_decode(logits.cpu(), ITOS)[0]
    return {"text": text}
