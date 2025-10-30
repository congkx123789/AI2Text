import torch, torchaudio

def ensure_mono16k(waveform, sr):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform, 16000

def wav_to_logmelspec(waveform, sr=16000, n_fft=400, hop=160, n_mels=80):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )(waveform)
    logmel = torch.log(mel + 1e-6)       # (n_mels, frames)
    return logmel.transpose(0,1)         # (frames, n_mels)
