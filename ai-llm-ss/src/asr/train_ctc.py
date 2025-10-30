import argparse, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from .dataset import ASRDataset, collate_batch
from .model import CRNNCTC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_audio", default="data/raw/audio")
    ap.add_argument("--train_text",  default="data/raw/text")
    ap.add_argument("--vocab",       default="data/processed/vocab.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/results/asr_ctc.pt")
    args = ap.parse_args()

    ds = ASRDataset(args.train_audio, args.train_text, args.vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    model = CRNNCTC(n_mels=80, vocab_size=len(ds.vocab)).to(args.device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        model.train(); total = 0.0
        for X, Xlen, Y, Ylen in dl:
            X, Xlen, Y, Ylen = X.to(args.device), Xlen.to(args.device), Y.to(args.device), Ylen.to(args.device)
            logits, out_lens = model(X, Xlen)      # (T,B,V)
            log_probs = logits.log_softmax(dim=-1)
            loss = ctc(log_probs, Y, out_lens, Ylen)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {ep} | loss {total/len(dl):.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")

if __name__ == "__main__":
    main()
