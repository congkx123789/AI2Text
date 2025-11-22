import argparse, os, yaml
from nlp.word2vec_trainer import train_word2vec, export_to_sqlite as export_w2v
from nlp.phon2vec_trainer import train_phon2vec, export_to_sqlite as export_p2v
from nlp.faiss_index import build_and_save_from_db

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="database/asr_training.db")
    ap.add_argument("--config", default="configs/embeddings.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = cfg.get("output_dir", "models/embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Train Word2Vec (semantic)
    print("[1/4] Training Word2Vec...")
    w2v_path = train_word2vec(args.db, out_dir,
                              vector_size=cfg.get("vector_size",256),
                              window=cfg.get("window",5),
                              min_count=cfg.get("min_count",2),
                              workers=cfg.get("workers",4),
                              epochs=cfg.get("epochs",10))
    print("Saved:", w2v_path)
    print("Export to SQLite...")
    export_w2v(os.path.join(out_dir,"word2vec.kv"), args.db, table="WordEmbeddings")

    # Train Phon2Vec (phonetic)
    if cfg.get("phonetic",{}).get("enabled", True):
        print("[2/4] Training Phon2Vec...")
        p2v_path = train_phon2vec(args.db, out_dir,
                                  vector_size=max(128, cfg.get("vector_size",256)//2),
                                  window=cfg.get("window",5),
                                  min_count=cfg.get("min_count",2),
                                  workers=cfg.get("workers",4),
                                  epochs=cfg.get("epochs",10),
                                  telex=cfg.get("phonetic",{}).get("use_telex",True),
                                  tone=cfg.get("phonetic",{}).get("tone_token",True))
        print("Saved:", p2v_path)
        print("Export to SQLite...")
        export_p2v(os.path.join(out_dir,"phon2vec.kv"), args.db, table="PronunciationEmbeddings")

    # Build FAISS indexes
    nlist = cfg.get("faiss",{}).get("nlist",1024)
    print("[3/4] Building FAISS (semantic)...")
    build_and_save_from_db(args.db, "WordEmbeddings", os.path.join(out_dir,"word2vec.faiss"), nlist=nlist)
    print("[4/4] Building FAISS (phonetic)...")
    build_and_save_from_db(args.db, "PronunciationEmbeddings", os.path.join(out_dir,"phon2vec.faiss"), nlist=nlist)
    print("Done.")

if __name__ == "__main__":
    main()
