"""
Build Word2Vec and Phon2Vec embeddings from database transcripts.

Trains semantic (Word2Vec) and phonetic (Phon2Vec) embeddings for
contextual biasing and N-best rescoring.
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from nlp.word2vec_trainer import train_word2vec, export_to_sqlite as export_word2vec
from nlp.phon2vec_trainer import train_phon2vec, export_to_sqlite as export_phon2vec
from database.db_utils import ASRDatabase


def main():
    parser = argparse.ArgumentParser(description='Build embeddings from database')
    parser.add_argument('--db', type=str, default='database/asr_training.db',
                       help='Path to database')
    parser.add_argument('--config', type=str, default='configs/embeddings.yaml',
                       help='Path to embeddings config')
    parser.add_argument('--semantic-only', action='store_true',
                       help='Train only semantic Word2Vec')
    parser.add_argument('--phonetic-only', action='store_true',
                       help='Train only phonetic Phon2Vec')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default config
        default_config = {
            'output_dir': 'models/embeddings',
            'min_count': 2,
            'vector_size': 256,
            'window': 5,
            'workers': 4,
            'epochs': 10,
            'phonetic': {
                'enabled': True,
                'tone_token': True,
                'use_telex': True
            }
        }
        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    output_dir = config.get('output_dir', 'models/embeddings')
    
    print("="*70)
    print("Building Embeddings from Database")
    print("="*70)
    print(f"Database: {args.db}")
    print(f"Output: {output_dir}")
    print()
    
    # Train semantic Word2Vec
    if not args.phonetic_only:
        print("Training semantic Word2Vec...")
        word2vec_path = train_word2vec(
            db_path=args.db,
            out_dir=output_dir,
            vector_size=config.get('vector_size', 256),
            window=config.get('window', 5),
            min_count=config.get('min_count', 2),
            workers=config.get('workers', 4),
            epochs=config.get('epochs', 10)
        )
        print(f"[OK] Word2Vec saved to: {word2vec_path}")
        
        # Export to database
        kv_path = str(Path(output_dir) / "word2vec.kv")
        if Path(kv_path).exists():
            export_word2vec(kv_path, args.db)
            print("[OK] Word2Vec exported to database")
        print()
    
    # Train phonetic Phon2Vec
    if not args.semantic_only and config.get('phonetic', {}).get('enabled', True):
        print("Training phonetic Phon2Vec...")
        phon2vec_path = train_phon2vec(
            db_path=args.db,
            out_dir=output_dir,
            vector_size=config.get('phonetic', {}).get('vector_size', 128),
            window=config.get('window', 5),
            min_count=config.get('min_count', 2),
            workers=config.get('workers', 4),
            epochs=config.get('epochs', 10),
            telex=config.get('phonetic', {}).get('use_telex', True),
            tone=config.get('phonetic', {}).get('tone_token', True)
        )
        print(f"[OK] Phon2Vec saved to: {phon2vec_path}")
        
        # Export to database
        kv_path = str(Path(output_dir) / "phon2vec.kv")
        if Path(kv_path).exists():
            export_phon2vec(kv_path, args.db)
            print("[OK] Phon2Vec exported to database")
        print()
    
    print("="*70)
    print("[OK] Embeddings training completed!")
    print("="*70)


if __name__ == '__main__':
    main()

