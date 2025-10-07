# tests/test_audio.py
from src.ai2text.audio.inference import transcribe

def test_transcribe_runs_smoke(tmp_path):
    # Use a tiny WAV bundled in your repo for CI, or generate a 1s silent wav.
    import soundfile as sf, numpy as np
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(16000, dtype="float32"), 16000)
    # Vosk expects speech; this is a smoke test (should return empty or small string)
    text = transcribe(str(p), backend="vosk")
    assert isinstance(text, str)
