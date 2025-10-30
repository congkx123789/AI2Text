import torch

def greedy_decode(logits, itos):
    # logits: (T,B,V)
    pred = logits.argmax(dim=-1).transpose(0,1)  # (B,T)
    texts = []
    for seq in pred:
        prev = None; out=[]
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                out.append(itos[idx])
            prev = idx
        texts.append("".join(out))
    return texts
