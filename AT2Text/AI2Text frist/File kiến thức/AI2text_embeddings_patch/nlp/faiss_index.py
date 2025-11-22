import os
import numpy as np
import sqlite3
import faiss

def _load_vectors_from_db(db_path: str, table: str):
    con = sqlite3.connect(db_path)
    cur = con.execute(f"SELECT token, vector, dim FROM {table}")
    tokens, vecs = [], []
    for token, blob, dim in cur.fetchall():
        vec = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        tokens.append(token)
        vecs.append(vec)
    con.close()
    if not vecs:
        return [], np.zeros((0, 1), dtype=np.float32)
    return tokens, np.vstack(vecs).astype(np.float32)

def build_ivf_index(vecs: np.ndarray, nlist: int = 1024):
    d = vecs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(vecs)
    index.train(vecs)
    index.add(vecs)
    return index

def save_index(index, path: str, tokens: list[str]):
    faiss.write_index(index, path)
    with open(path + ".vocab", "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")

def load_index(path: str):
    index = faiss.read_index(path)
    with open(path + ".vocab", "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return index, tokens

def query(index, tokens, query_vecs: np.ndarray, k: int = 5, nprobe: int = 16):
    faiss.normalize_L2(query_vecs)
    index.nprobe = nprobe
    scores, idx = index.search(query_vecs, k)
    results = []
    for row_scores, row_idx in zip(scores, idx):
        hits = []
        for s, i in zip(row_scores, row_idx):
            if i == -1:
                continue
            hits.append((tokens[i], float(s)))
        results.append(hits)
    return results

def build_and_save_from_db(db_path: str, table: str, out_path: str, nlist: int = 1024):
    tokens, vecs = _load_vectors_from_db(db_path, table)
    if vecs.shape[0] == 0:
        raise RuntimeError(f"No vectors found in table {table}.")
    index = build_ivf_index(vecs, nlist=nlist)
    save_index(index, out_path, tokens)
    return out_path
