import torch, torchaudio, glob, os, json, csv
try:
    import soundfile as sf
except Exception:
    sf = None
from .features import wav_to_logmelspec, ensure_mono16k


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, text_dir, vocab_path):
        self.audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        self.text_dir = text_dir
        self.vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        self.stoi = {c:i for i,c in enumerate(self.vocab)}

    def __len__(self): return len(self.audio_paths)

    def __getitem__(self, idx):
        ap = self.audio_paths[idx]
        name = os.path.splitext(os.path.basename(ap))[0]
        tp = os.path.join(self.text_dir, f"{name}.txt")
        transcript = open(tp, "r", encoding="utf-8").read().strip().lower()

        wav, sr = self._load_audio(ap)
        wav, sr = ensure_mono16k(wav, sr)
        x = wav_to_logmelspec(wav, sr)               # (T, 80)

        y = torch.tensor([self.stoi.get(c, 1) for c in transcript], dtype=torch.long)
        return x, y


class ManifestDataset(torch.utils.data.Dataset):
    """Dataset driven by manifest.csv and optional timestamps.json for trimming."""

    def __init__(self, manifest_path, vocab_path, audio_root=None, timestamps_path=None, trim_to_segments=False):
        self.manifest_path = manifest_path
        self.audio_root = audio_root or os.path.dirname(manifest_path)
        self.trim_to_segments = trim_to_segments

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = [row for row in reader]

        self.timestamps = None
        if timestamps_path and os.path.exists(timestamps_path):
            with open(timestamps_path, "r", encoding="utf-8") as f:
                self.timestamps = json.load(f)

        self.vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        transcript = row["transcript"].strip()
        audio_rel = row["audio_path"]
        ap = audio_rel if os.path.isabs(audio_rel) else os.path.join(self.audio_root, audio_rel)

        wav, sr = self._load_audio(ap)
        wav, sr = ensure_mono16k(wav, sr)

        if self.trim_to_segments and self.timestamps:
            key = os.path.basename(ap)
            meta = self.timestamps.get(key)
            if meta and meta.get("segments"):
                starts = [s["start"] for s in meta["segments"]]
                ends = [s["end"] for s in meta["segments"]]
                if starts and ends:
                    start, end = max(0.0, min(starts)), min(max(ends), wav.size(-1) / sr)
                    s_idx, e_idx = int(start * sr), int(end * sr)
                    if e_idx > s_idx:
                        wav = wav[..., s_idx:e_idx]

        x = wav_to_logmelspec(wav, sr)  # (T, 80)
        y = torch.tensor([self.stoi.get(c, 1) for c in transcript], dtype=torch.long)
        return x, y

    def _load_audio(self, path):
        """Robust audio loader: try torchaudio, fallback to soundfile."""
        try:
            return torchaudio.load(path)
        except Exception:
            if sf is None:
                raise
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav).transpose(0, 1)  # to (channels, time)
            return wav, sr

def collate_batch(batch):
    xs, ys = zip(*batch)
    x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)                # (B, T, F)
    Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return X, x_lens, Y, y_lens
