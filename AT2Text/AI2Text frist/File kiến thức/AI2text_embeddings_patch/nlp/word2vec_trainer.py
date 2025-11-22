from __future__ import annotations
import os, sqlite3
from typing import Iterable, List
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def _iter_transcripts(db_path: str) -> Iterable[List[str]]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT transcript FROM Transcripts WHERE transcript IS NOT NULL")
        for (txt,) in cur.fetchall():
            yield simple_preprocess(txt or "", deacc=False, min_len=1, max_len=50)
    finally:
        con.close()

def train_word2vec(
    db_path: str,
    out_dir: str,
    vector_size: int = 256,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sentences = list(_iter_transcripts(db_path))
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1
    )
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    out_path = os.path.join(out_dir, "word2vec.model")
    model.save(out_path)
    model.wv.save(os.path.join(out_dir, "word2vec.kv"))
    return out_path

def export_to_sqlite(kv_path: str, db_path: str, table: str = "WordEmbeddings"):
    import sqlite3
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load(kv_path, mmap="r")
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT UNIQUE, vector BLOB, dim INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        con.execute(f"DELETE FROM {table};")
        q = f"INSERT OR REPLACE INTO {table} (token, vector, dim) VALUES (?,?,?)"
        for tok in kv.index_to_key:
            vec = kv.get_vector(tok).astype('float32')
            blob = vec.tobytes()
            con.execute(q, (tok, blob, vec.shape[0]))
        con.commit()
    finally:
        con.close()
