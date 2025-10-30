import torch, torchaudio, glob, os, json
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

        wav, sr = torchaudio.load(ap)
        wav, sr = ensure_mono16k(wav, sr)
        x = wav_to_logmelspec(wav, sr)               # (T, 80)

        y = torch.tensor([self.stoi.get(c, 1) for c in transcript], dtype=torch.long)
        return x, y

def collate_batch(batch):
    xs, ys = zip(*batch)
    x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)                # (B, T, F)
    Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return X, x_lens, Y, y_lens
