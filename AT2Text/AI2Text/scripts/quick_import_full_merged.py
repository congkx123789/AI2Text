#!/usr/bin/env python3
"""
Quick import script for full_merged_dataset - imports a subset for testing.
"""
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.text_cleaning import BilingualTextNormalizer

def main():
    base_dir = Path("data/processed/full_merged_dataset")
    train_manifest = base_dir / "train" / "manifest.csv"
    val_manifest = base_dir / "val" / "manifest.csv"
    split_version = "v3_full"
    database_path = "database/asr_training.db"
    
    print("=" * 60)
    print("Quick Import - Full Merged Dataset (Subset for Testing)")
    print("=" * 60)
    
    # Read manifests
    print("Reading manifests...")
    train_df = pd.read_csv(train_manifest, nrows=1000)  # Only first 1000 for testing
    val_df = pd.read_csv(val_manifest, nrows=200)  # Only first 200 for testing
    
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    
    # Initialize database
    db = ASRDatabase(database_path)
    normalizer = BilingualTextNormalizer()
    
    # Prepare data
    def prepare_row(row, split_name):
        audio_path = row['audio_path']
        # Fix path: manifest has 'audio/xxx.wav', need 'train/audio/xxx.wav' or 'val/audio/xxx.wav'
        if not audio_path.startswith(split_name):
            audio_path = f"{split_name}/{audio_path}"
        
        full_path = base_dir / audio_path
        if not full_path.exists():
            return None
        
        transcript = row['transcript']
        language = 'vi' if transcript.startswith('<|vi|>') else 'en'
        
        # Remove language tags from transcript
        if transcript.startswith('<|vi|>'):
            transcript = transcript[6:]
        elif transcript.startswith('<|en|>'):
            transcript = transcript[6:]
        
        normalized = normalizer.normalize(transcript, lang=language)
        
        return {
            'file_path': str(full_path),
            'transcript': transcript,
            'normalized_transcript': normalized,
            'language': language,
            'split': split_name,
            'split_version': split_version
        }
    
    # Import train data
    print("\nImporting train data...")
    train_count = 0
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        data = prepare_row(row, 'train')
        if data is None:
            continue
        
        try:
            # Check if exists
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?",
                    (data['file_path'],)
                )
                if cursor.fetchone():
                    continue
            
            # Insert
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO AudioFiles 
                    (file_path, filename, transcript, normalized_transcript, language, split, split_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['file_path'],
                    Path(data['file_path']).name,
                    data['transcript'],
                    data['normalized_transcript'],
                    data['language'],
                    data['split'],
                    data['split_version']
                ))
                conn.commit()
            train_count += 1
        except Exception as e:
            print(f"Error importing {row.get('id', idx)}: {e}")
    
    # Import val data
    print("\nImporting val data...")
    val_count = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
        data = prepare_row(row, 'val')
        if data is None:
            continue
        
        try:
            # Check if exists
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?",
                    (data['file_path'],)
                )
                if cursor.fetchone():
                    continue
            
            # Insert
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO AudioFiles 
                    (file_path, filename, transcript, normalized_transcript, language, split, split_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['file_path'],
                    Path(data['file_path']).name,
                    data['transcript'],
                    data['normalized_transcript'],
                    data['language'],
                    data['split'],
                    data['split_version']
                ))
                conn.commit()
            val_count += 1
        except Exception as e:
            print(f"Error importing {row.get('id', idx)}: {e}")
    
    print(f"\n✅ Import completed!")
    print(f"   Train: {train_count:,} samples")
    print(f"   Val: {val_count:,} samples")
    print(f"   Split version: {split_version}")

if __name__ == '__main__':
    main()

