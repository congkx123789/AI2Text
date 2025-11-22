# Embeddings Patch for AI2text (Vietnamese ASR)
This patch adds **Word2Vec semantic embeddings** and **Phonetic ("sound") embeddings (Phon2Vec)**,
plus FAISS indexes and **N-best rescoring** that blends acoustic scores with semantic & phonetic similarity.

## Why this matters
- **Semantic Word2Vec** helps choose hypotheses that are meaningful in context (domain/topic coherence).
- **Phon2Vec** (training Word2Vec on Vietnamese **phonetic tokens** like Telex+tone) helps with
  rare/OOV words and **sound-alike** confusions (e.g., tone or regional variants).
- Together they enable **contextual biasing** and **dynamic OOV handling** without re-training the acoustic model.

## Install deps
```bash
pip install -r requirements-embeddings.txt
```

## DB migration
Append the migration SQL to your DB (SQLite):
```bash
sqlite3 database/asr_training.db < database/migrations/001_add_embedding_tables.sql
```

## Train embeddings
```bash
python scripts/build_embeddings.py --db database/asr_training.db --config configs/embeddings.yaml
```
This will produce:
- `models/embeddings/word2vec.model` & `.kv` (semantic)
- `models/embeddings/phon2vec.model` & `.kv` (phonetic)
- `models/embeddings/*.faiss` + `.vocab`

## Rescore n-best
Provide an n-best (JSON) from your decoder (CTC/RNNT). Example:
```json
[
  {"text": "toi muon dat ban",  "am_score": -12.5, "lm_score": -1.1},
  {"text": "toi muon dat banh", "am_score": -12.8, "lm_score": -1.0}
]
```
Run:
```bash
python scripts/rescore_nbest.py --nbest nbest.json --context "đặt bánh gato sinh nhật"
```

## Integration tips
- Call `rescore_nbest` inside your `training/evaluate.py` right after decoding.
- If you already maintain a **bias list** (names, brands), compute its **phonetic tokens** and
  pull nearest neighbors from the FAISS phonetic index to nudge the score.
- Export embeddings to DB so you can **version** them alongside runs.

## Notes
- Phonetic rules here are lightweight. For production, consider a Vietnamese G2P (grapheme-to-phoneme)
  or pronunciation lexicon to get even sharper "sound" vectors.
