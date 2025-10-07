# scripts/train_seq2seq_envi.py
"""
Train a small GRU seq2seq with Bahdanau attention on IWSLT2015 En-Vi.
Saves:
  - models/en_vocab.json
  - models/vi_vocab.json
  - models/envi_encoder.pt
  - models/envi_decoder.pt
This is a minimal educational trainer (CPU-ok but slow). For real training, use GPU and more epochs.
"""
import os, re, json, argparse, random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-zà-ỹ0-9\s\-\']", " ", s)  # keep basic VN chars; adjust as needed
    s = re.sub(r"\s+", " ", s)
    return s

def read_pairs(path):
    # Expect TSV with columns: en \t vi
    pairs=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line: continue
            en, vi = line.rstrip("\n").split("\t")[:2]
            en, vi = norm(en), norm(vi)
            if en and vi:
                pairs.append((en, vi))
    return pairs

def build_vocab(texts, min_freq=2, max_size=40000, specials=("<pad>","<s>","</s>","<unk>")):
    freq={}
    for s in texts:
        for w in s.split():
            freq[w]=freq.get(w,0)+1
    words=[w for w,c in freq.items() if c>=min_freq]
    words.sort(key=lambda w:(-freq[w], w))
    words=words[:max_size]
    stoi={specials[0]:0, specials[1]:1, specials[2]:2, specials[3]:3}
    for i,w in enumerate(words, start=len(stoi)):
        stoi[w]=i
    return stoi

class MTData(Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi, max_len=64):
        self.pairs=pairs; self.src=src_stoi; self.tgt=tgt_stoi; self.max_len=max_len
        self.sos=self.tgt["<s>"]; self.eos=self.tgt["</s>"]; self.pad=self.tgt["<pad>"]
    def __len__(self): return len(self.pairs)
    def enc(self, s, table):
        ids=[table.get(w, table["<unk>"]) for w in s.split()][:self.max_len-2]
        return [table["<s>"]] + ids + [table["</s>"]]
    def pad(self, arr, pad_id):
        if len(arr) < self.max_len:
            arr = arr + [pad_id]*(self.max_len-len(arr))
        else:
            arr = arr[:self.max_len]
        return arr
    def __getitem__(self, i):
        en, vi = self.pairs[i]
        x = self.pad([self.src.get(w, self.src["<unk>"]) for w in en.split()][:self.max_len], self.src["<pad>"])
        y = self.enc(vi, self.tgt)
        y = self.pad(y, self.pad)
        return torch.tensor(x), torch.tensor(y)

class Encoder(nn.Module):
    def __init__(self, vocab, emb=256, hid=512):
        super().__init__()
        self.emb=nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn=nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.proj=nn.Linear(hid*2, hid)
    def forward(self, x):
        em=self.emb(x)
        out,h=self.rnn(em)
        ctx=self.proj(torch.cat([h[-2],h[-1]], dim=-1)).unsqueeze(0)
        return out,ctx

class Bahdanau(nn.Module):
    def __init__(self, enc_h=1024, dec_h=512):
        super().__init__()
        self.W1=nn.Linear(enc_h, dec_h)
        self.W2=nn.Linear(dec_h, dec_h)
        self.v =nn.Linear(dec_h, 1)
    def forward(self, enc_out, dec_h):
        e=torch.tanh(self.W1(enc_out)+self.W2(dec_h).unsqueeze(1))
        a=torch.softmax(self.v(e).squeeze(-1), dim=1)
        c=(enc_out*a.unsqueeze(-1)).sum(1)
        return c,a

class Decoder(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, enc_h=1024):
        super().__init__()
        self.emb=nn.Embedding(vocab, emb, padding_idx=0)
        self.att=Bahdanau(enc_h=enc_h, dec_h=hid)
        self.rnn=nn.GRU(emb+enc_h, hid, batch_first=True)
        self.fc =nn.Linear(hid, vocab)
    def forward(self, y_prev, h, enc_out):
        em=self.emb(y_prev)  # [B,1,E]
        c,_=self.att(enc_out, h.squeeze(0))  # [B,enc_h]
        x=torch.cat([em, c.unsqueeze(1)], dim=-1)
        out,h2=self.rnn(x, h)
        return self.fc(out), h2

def train(data_tsv, out_dir="models", epochs=6, batch=64, lr=2e-4):
    os.makedirs(out_dir, exist_ok=True)
    pairs=read_pairs(data_tsv)
    random.shuffle(pairs)
    n=int(0.95*len(pairs))
    train_pairs=pairs[:n]; valid_pairs=pairs[n:]

    src_stoi=build_vocab([p[0] for p in train_pairs])
    tgt_stoi=build_vocab([p[1] for p in train_pairs])

    with open(os.path.join(out_dir,"en_vocab.json"),"w",encoding="utf-8") as f:
        json.dump(src_stoi,f,ensure_ascii=False)
    with open(os.path.join(out_dir,"vi_vocab.json"),"w",encoding="utf-8") as f:
        json.dump(tgt_stoi,f,ensure_ascii=False)

    trDS=MTData(train_pairs, src_stoi, tgt_stoi)
    vaDS=MTData(valid_pairs, src_stoi, tgt_stoi)
    trDL=DataLoader(trDS, batch_size=batch, shuffle=True)
    vaDL=DataLoader(vaDS, batch_size=batch)

    enc=Encoder(vocab=len(src_stoi))
    dec=Decoder(vocab=len(tgt_stoi))
    crit=nn.CrossEntropyLoss(ignore_index=tgt_stoi["<pad>"])
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr)

    for ep in range(1, epochs+1):
        enc.train(); dec.train()
        losses=[]
        for x, y in tqdm(trDL, desc=f"Epoch {ep}/{epochs}"):
            # Teacher forcing: next input is previous gold token
            y_in = y[:, :-1]  # [B,T-1]
            y_tg = y[:, 1:]   # [B,T-1]

            opt.zero_grad()
            enc_out, h = enc(x)
            B, Tm1 = y_in.size()
            loss=0.0
            y_prev = y_in[:,0].unsqueeze(1)
            for t in range(Tm1):
                logits, h = dec(y_prev, h, enc_out)
                loss += crit(logits.squeeze(1), y_tg[:, t])
                if t+1 < Tm1:
                    y_prev = y_in[:, t+1].unsqueeze(1)
            (loss/Tm1).backward()
            opt.step()
            losses.append(float(loss.item()/Tm1))
        print(f"train loss: {sum(losses)/len(losses):.4f}")

    torch.save(enc.state_dict(), os.path.join(out_dir,"envi_encoder.pt"))
    torch.save(dec.state_dict(), os.path.join(out_dir,"envi_decoder.pt"))
    print("Saved seq2seq weights.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_tsv", required=True, help="TSV with 'en\\tvi' per line")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    train(args.data_tsv, args.out_dir, args.epochs, args.batch, args.lr)
