-- Adds embedding tables for semantic Word2Vec and phonetic (Phon2Vec)
-- Safe to re-run due to IF NOT EXISTS
CREATE TABLE IF NOT EXISTS WordEmbeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE,
    vector BLOB,                 -- raw float32 bytes
    dim INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS PronunciationEmbeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT UNIQUE,           -- phonetic token (e.g., VnSoundex/telex-coded syllable)
    vector BLOB,                 -- raw float32 bytes
    dim INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
