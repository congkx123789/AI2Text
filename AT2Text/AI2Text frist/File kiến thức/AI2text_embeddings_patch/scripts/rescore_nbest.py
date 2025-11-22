import argparse, json
from gensim.models import KeyedVectors
from language_model.rescoring import rescore_nbest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", required=True, help="Path to n-best JSON. Format: [{'text':..., 'am_score':..., 'lm_score':...}, ...]")
    ap.add_argument("--kv_sem", default="models/embeddings/word2vec.kv")
    ap.add_argument("--kv_ph",  default="models/embeddings/phon2vec.kv")
    ap.add_argument("--context", default="", help="Optional context text to bias.")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.5)
    args = ap.parse_args()

    nbest = json.load(open(args.nbest, "r", encoding="utf-8"))
    kv_sem = KeyedVectors.load(args.kv_sem, mmap="r") if args.kv_sem else None
    kv_ph  = KeyedVectors.load(args.kv_ph,  mmap="r") if args.kv_ph else None
    rescored = rescore_nbest(nbest, kv_sem, kv_ph, context_text=args.context,
                             alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta)
    print(json.dumps(rescored, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
