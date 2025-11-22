"""
Phon2Vec trainer for phonetic embeddings using Vietnamese phonetic tokens.

Trains Word2Vec on phonetic tokens (Telex+tone) to capture sound-based
similarities. Helps with rare/OOV words and sound-alike confusions.
"""

from __future__ import annotations
import os
import sqlite3
from typing import Iterable, List
from gensim.models import Word2Vec
from .phonetic import phonetic_tokens
from pathlib import Path


def _iter_phonetic(db_path: str, telex: bool = True, tone: bool = True) -> Iterable[List[str]]:
    """
    Iterator over phonetic tokens from transcripts.
    
    Converts transcripts to phonetic tokens (Telex+tone) for Phon2Vec training.
    
    Args:
        db_path: Path to SQLite database
        telex: Use Telex encoding
        tone: Include tone markers
        
    Yields:
        List of phonetic tokens from each transcript
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT transcript FROM Transcripts WHERE transcript IS NOT NULL")
        for (txt,) in cur.fetchall():
            # Convert to phonetic tokens
            toks = phonetic_tokens(txt or "", telex=telex, tone_token=tone)
            if toks:
                yield toks
    finally:
        con.close()


def train_phon2vec(
    db_path: str,
    out_dir: str,
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10,
    telex: bool = True,
    tone: bool = True,
) -> str:
    """
    Train Phon2Vec model on phonetic tokens from transcripts.
    
    Trains Word2Vec on phonetic tokens (Telex+tone) to create sound-based
    embeddings. Helps with rare words, OOV handling, and sound-alike confusions.
    
    Args:
        db_path: Path to SQLite database with transcripts
        out_dir: Output directory for model files
        vector_size: Embedding dimension (default: 128)
        window: Context window size (default: 5)
        min_count: Minimum token frequency (default: 2)
        workers: Number of worker threads (default: 4)
        epochs: Number of training epochs (default: 10)
        telex: Use Telex encoding (True) or Soundex (False)
        tone: Include tone markers in tokens
        
    Returns:
        Path to saved model file
        
    Example:
        >>> model_path = train_phon2vec(
        ...     db_path="database/asr_training.db",
        ...     out_dir="models/embeddings",
        ...     vector_size=128,
        ...     telex=True,
        ...     tone=True
        ... )
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load phonetic tokens from transcripts
    sentences = list(_iter_phonetic(db_path, telex=telex, tone=tone))
    
    if not sentences:
        raise ValueError("No transcripts found in database")
    
    # Initialize and train Word2Vec on phonetic tokens
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1  # Skip-gram
    )
    
    # Train model
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    
    # Save model
    out_path = os.path.join(out_dir, "phon2vec.model")
    model.save(out_path)
    
    # Save keyed vectors
    model.wv.save(os.path.join(out_dir, "phon2vec.kv"))
    
    return out_path


def export_to_sqlite(kv_path: str, db_path: str, table: str = "PronunciationEmbeddings"):
    """
    Export Phon2Vec embeddings to SQLite database.
    
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

