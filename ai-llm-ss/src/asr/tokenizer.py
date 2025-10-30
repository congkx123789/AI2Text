import json
from collections import OrderedDict
from pathlib import Path

BLANK = "<blank>"
UNK = "<unk>"

def build_char_vocab(transcripts, extra_tokens=(BLANK, UNK)):
    chars = set()
    for t in transcripts:
        t = t.lower().strip()
        chars.update(list(t))
    for ch in ["\n","\r","\t"]:
        if ch in chars:
            chars.remove(ch)
    vocab = list(extra_tokens) + sorted(chars)
    stoi = OrderedDict((c,i) for i,c in enumerate(vocab))
    itos = {i:c for c,i in stoi.items()}
    return {"vocab":vocab, "stoi":dict(stoi), "itos":itos}

def encode(text, stoi):
    return [stoi.get(c, stoi[UNK]) for c in text.lower()]

def save_vocab(path, vocab):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab["vocab"], f, ensure_ascii=False, indent=2)
