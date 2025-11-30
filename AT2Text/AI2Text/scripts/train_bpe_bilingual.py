"""
Train a bilingual (Vietnamese + English) BPE tokenizer on all normalized transcripts
stored in the ASR training database.

Usage:
    python scripts/train_bpe_bilingual.py [--vocab-size 4000] [--output models/bilingual_bpe_4000.json]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description='Train bilingual BPE tokenizer')
    parser.add_argument('--vocab-size', type=int, default=2000,
                       help='Vocabulary size (default: 2000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: models/bilingual_bpe.json or models/bilingual_bpe_{vocab_size}.json)')
    parser.add_argument('--db-path', type=str, default='database/asr_training.db',
                       help='Path to database (default: database/asr_training.db)')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum frequency for BPE merges (default: 2)')
    
    args = parser.parse_args()
    
    # Default paths
    db_path = args.db_path
    vocab_size = args.vocab_size
    
    # Auto-generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        if vocab_size == 2000:
            output_path = "models/bilingual_bpe.json"
        else:
            output_path = f"models/bilingual_bpe_{vocab_size}.json"
    
    min_frequency = args.min_frequency

    # Ensure models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)

    db = ASRDatabase(db_path)

    texts = []
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT normalized_transcript FROM Transcripts")
        texts = [row[0] for row in cursor.fetchall() if row[0]]

    print(f"Training BPE on {len(texts)} sentences...")
    if not texts:
        print("No transcripts found in database. Make sure you imported data first.")
        return

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.save(output_path)
    print(f"BPE Tokenizer saved to: {output_path}")


if __name__ == "__main__":
    main()


