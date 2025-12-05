"""
Word2Vec trainer for semantic embeddings from Vietnamese transcripts.

Trains Word2Vec embeddings on transcripts to capture semantic relationships
between words. Used for contextual biasing and semantic similarity.
"""

from __future__ import annotations
import os
import sqlite3
from typing import Iterable, List
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pathlib import Path


def _iter_transcripts(db_path: str) -> Iterable[List[str]]:
    """
    Iterator over transcripts from database.
    
    Yields lists of preprocessed words from each transcript.
    
    Args:
        db_path: Path to SQLite database
        
    Yields:
        List of words from each transcript
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT transcript FROM Transcripts WHERE transcript IS NOT NULL")
        for (txt,) in cur.fetchall():
            # Preprocess: lowercase, tokenize, filter
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
    """
    Train Word2Vec model on transcripts from database.
    
    Trains a skip-gram Word2Vec model on Vietnamese transcripts to create
    semantic word embeddings for contextual biasing and similarity matching.
    
    Args:
        db_path: Path to SQLite database with transcripts
        out_dir: Output directory for model files
        vector_size: Embedding dimension (default: 256)
        window: Context window size (default: 5)
        min_count: Minimum word frequency (default: 2)
        workers: Number of worker threads (default: 4)
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Path to saved model file
        
    Example:
        >>> model_path = train_word2vec(
        ...     db_path="database/asr_training.db",
        ...     out_dir="models/embeddings",
        ...     vector_size=256,
        ...     epochs=10
        ... )
        >>> print(f"Model saved to: {model_path}")
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load all transcripts
    sentences = list(_iter_transcripts(db_path))
    
    if not sentences:
        raise ValueError("No transcripts found in database")
    
    # Initialize and train Word2Vec model (Skip-gram)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1  # Skip-gram (1) vs CBOW (0)
    )
    
    # Train model
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    
    # Save model
    out_path = os.path.join(out_dir, "word2vec.model")
    model.save(out_path)
    
    # Save keyed vectors (for faster loading)
    model.wv.save(os.path.join(out_dir, "word2vec.kv"))
    
    return out_path


def export_to_sqlite(kv_path: str, db_path: str, table: str = "WordEmbeddings"):
    """
    Export Word2Vec embeddings to SQLite database.
    
    Args:
        kv_path: Path to KeyedVectors file (.kv)
        db_path: Path to SQLite database
        table: Table name to store embeddings
    """
    import sqlite3
    from gensim.models import KeyedVectors
    
    kv = KeyedVectors.load(kv_path, mmap="r")
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT UNIQUE, "
            "vector BLOB, dim INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        con.execute(f"DELETE FROM {table};")
        
        # Insert embeddings
        q = f"INSERT OR REPLACE INTO {table} (token, vector, dim) VALUES (?,?,?)"
        for tok in kv.index_to_key:
            vec = kv.get_vector(tok).astype('float32')
            blob = vec.tobytes()
            con.execute(q, (tok, blob, vec.shape[0]))
        
        con.commit()
    finally:
        con.close()

