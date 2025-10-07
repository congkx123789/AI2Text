# scripts/train_sentiment.py
"""
Train sentiment models on IMDB:
 - LSTM (PyTorch)
 - Logistic Regression (scikit-learn, TF-IDF)
Saves to models/sentiment_lstm.pt, models/sent_vocab.txt, models/sentiment_logreg.pkl
"""
import os, re, json, random, argparse, joblib
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def clean_text(s):
    s = s.lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_imdb(imdb_dir="aclImdb"):
    """
    Expect standard IMDB layout:
    aclImdb/train/pos, aclImdb/train/neg, aclImdb/test/pos, aclImdb/test/neg
    """
    def load_split(split):
        xs, ys = [], []
        for label, labdir in [(1,"pos"), (0,"neg")]:
            d = os.path.join(imdb_dir, split, labdir)
            for fn in os.listdir(d):
                if not fn.endswith(".txt"): continue
                with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                    xs.append(clean_text(f.read()))
                    ys.append(label)
        return xs, ys
    Xtr, Ytr = load_split("train")
    Xte, Yte = load_split("test")
    return Xtr, Ytr, Xte, Yte

# ---------- LSTM ----------
class LSTMDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    def _encode(self, s):
        toks = s.split()
        ids = [self.word2idx.get(w, 0) for w in toks][:self.max_len]
        length = len(ids)
        ids += [0]*(self.max_len - length)
        return np.array(ids, dtype=np.int64), length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        ids, length = self._encode(self.texts[i])
        return torch.from_numpy(ids), torch.tensor(length), torch.tensor(self.labels[i], dtype=torch.float32)

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, emb=100, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x, lengths):
        em = self.emb(x)  # [B,T,E]
        out, _ = self.lstm(em)
        # take last valid timestep
        idx = (lengths-1).clamp(min=0)
        last = out[torch.arange(out.size(0)), idx]
        return self.sig(self.fc(last)).squeeze(1)

def build_vocab(texts, min_freq=2, max_size=40000):
    freq = {}
    for s in texts:
        for w in s.split():
            freq[w] = freq.get(w, 0)+1
    words = [w for w,c in freq.items() if c>=min_freq]
    words.sort(key=lambda w: (-freq[w], w))
    words = words[:max_size]
    word2idx = {w:i+1 for i,w in enumerate(words)}  # 0 = pad / oov
    return word2idx, words

def train_lstm(Xtr, Ytr, Xte, Yte, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    word2idx, vocab_list = build_vocab(Xtr, min_freq=2)
    with open(os.path.join(out_dir, "sent_vocab.txt"), "w", encoding="utf-8") as f:
        for w in vocab_list: f.write(w+"\n")

    train_ds = LSTMDataset(Xtr, Ytr, word2idx)
    test_ds  = LSTMDataset(Xte, Yte, word2idx)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl  = DataLoader(test_ds, batch_size=128)

    model = LSTMNet(vocab_size=len(word2idx)+1)
    crit = nn.BCELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        losses=[]
        for x, lengths, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/5"):
            opt.zero_grad()
            p = model(x, lengths)
            loss = crit(p, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1} loss={np.mean(losses):.4f}")

    # Eval
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, lengths, y in test_dl:
            p = model(x, lengths)
            preds.extend((p>=0.5).int().cpu().numpy().tolist())
            golds.extend(y.int().cpu().numpy().tolist())
    acc = accuracy_score(golds, preds)
    print("LSTM accuracy:", acc)
    torch.save(model.state_dict(), os.path.join(out_dir, "sentiment_lstm.pt"))

def train_logreg(Xtr, Ytr, Xte, Yte, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=120000, stop_words="english")
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, solver="saga")
    clf.fit(Xtrv, Ytr)
    pred = clf.predict(Xtev)
    acc = accuracy_score(Yte, pred)
    print("LogReg accuracy:", acc)
    joblib.dump({"clf": clf, "vectorizer": vec}, os.path.join(out_dir, "sentiment_logreg.pkl"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb_dir", default="aclImdb", help="path to IMDB dataset root")
    args = ap.parse_args()
    Xtr, Ytr, Xte, Yte = load_imdb(args.imdb_dir)
    print("Train LSTM...")
    train_lstm(Xtr, Ytr, Xte, Yte)
    print("Train Logistic Regression...")
    train_logreg(Xtr, Ytr, Xte, Yte)
