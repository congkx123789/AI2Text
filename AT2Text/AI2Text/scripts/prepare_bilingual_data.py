"""
Script to prepare bilingual (English + Vietnamese) dataset for training.
Combines LibriSpeech (English) and VLSP/VIVOS (Vietnamese) datasets.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import BilingualTextNormalizer
import librosa


def load_librispeech_manifest(manifest_path: str) -> pd.DataFrame:
    """Load LibriSpeech manifest and add language column."""
    print(f"Loading LibriSpeech manifest: {manifest_path}")
    
    df = pd.read_csv(manifest_path)
    
    # Ensure required columns exist
    required_cols = ['id', 'transcript', 'audio_path']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in manifest: {missing_cols}")
    
    # Add language column
    df['language'] = 'en'
    
    # Rename columns to match expected format
    if 'id' in df.columns:
        # Keep original audio_path, it should be relative to processed directory
        pass
    
    # Select and rename columns
    result_df = pd.DataFrame({
        'file_path': df['audio_path'],
        'transcript': df['transcript'],
        'language': df['language'],
        'subset': df.get('subset', 'librispeech'),
        'sex': df.get('sex', '')
    })
    
    print(f"Loaded {len(result_df)} English samples")
    return result_df


def load_vietnamese_manifest(manifest_path: str) -> pd.DataFrame:
    """Load Vietnamese manifest and add language column."""
    print(f"Loading Vietnamese manifest: {manifest_path}")
    
    df = pd.read_csv(manifest_path)
    
    # Ensure required columns exist
    if 'file_path' not in df.columns or 'transcript' not in df.columns:
        raise ValueError("Manifest must have 'file_path' and 'transcript' columns")
    
    # Add language column if not present
    if 'language' not in df.columns:
        df['language'] = 'vi'
    
    # Ensure language is Vietnamese
    df['language'] = 'vi'
    
    print(f"Loaded {len(df)} Vietnamese samples")
    return df


def combine_datasets(english_df: pd.DataFrame, 
                    vietnamese_df: pd.DataFrame,
                    base_dir_english: str,
                    base_dir_vietnamese: str) -> pd.DataFrame:
    """Combine English and Vietnamese datasets with proper paths."""
    
    # Make paths absolute
    english_df = english_df.copy()
    vietnamese_df = vietnamese_df.copy()
    
    # Update English paths
    if not english_df['file_path'].iloc[0].startswith('/'):
        english_df['file_path'] = english_df['file_path'].apply(
            lambda x: str(Path(base_dir_english) / x)
        )
    
    # Update Vietnamese paths
    if not vietnamese_df['file_path'].iloc[0].startswith('/'):
        vietnamese_df['file_path'] = vietnamese_df['file_path'].apply(
            lambda x: str(Path(base_dir_vietnamese) / x)
        )
    
    # Combine
    combined_df = pd.concat([english_df, vietnamese_df], ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  English samples: {len(english_df)}")
    print(f"  Vietnamese samples: {len(vietnamese_df)}")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Language distribution:")
    print(combined_df['language'].value_counts())
    
    return combined_df


def normalize_transcripts(df: pd.DataFrame, normalizer: BilingualTextNormalizer) -> pd.DataFrame:
    """Normalize transcripts based on language."""
    print("\nNormalizing transcripts...")
    
    normalized_texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing"):
        lang = row.get('language', 'vi')
        transcript = str(row['transcript'])
        normalized = normalizer.normalize(transcript, lang=lang)
        normalized_texts.append(normalized)
    
    df['normalized_transcript'] = normalized_texts
    return df


def import_to_database(df: pd.DataFrame, db: ASRDatabase, audio_base_dir: str, 
                      split_version: str = "bilingual_v1", auto_split: bool = True):
    """Import combined dataset to database."""
    print(f"\nImporting {len(df)} samples to database...")
    
    audio_processor = AudioProcessor()
    normalizer = BilingualTextNormalizer()
    
    # Normalize transcripts
    df = normalize_transcripts(df, normalizer)
    
    # Process in batches
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Importing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        for _, row in batch_df.iterrows():
            try:
                file_path = Path(row['file_path'])
                if not file_path.is_absolute():
                    file_path = Path(audio_base_dir) / file_path
                
                if not file_path.exists():
                    print(f"Warning: Audio file not found: {file_path}")
                    continue
                
                # Load audio to get metadata
                audio, sr = librosa.load(str(file_path), sr=None, mono=True)
                duration = len(audio) / sr
                
                # Determine split if auto_split is enabled
                split = row.get('split', None)
                if split is None and auto_split:
                    # Simple split: use existing split if available, otherwise assign randomly
                    import random
                    rand = random.random()
                    if rand < 0.8:
                        split = 'train'
                    elif rand < 0.9:
                        split = 'val'
                    else:
                        split = 'test'
                
                # Import to database
                audio_id = db.add_audio_file(
                    file_path=str(file_path),
                    duration=duration,
                    sample_rate=sr,
                    quality_score=1.0
                )
                
                transcript_id = db.add_transcript(
                    audio_id=audio_id,
                    text=row['transcript'],
                    normalized_text=row['normalized_transcript'],
                    language=row.get('language', 'vi')
                )
                
                if split:
                    db.assign_to_split(
                        audio_id=audio_id,
                        split=split,
                        version=split_version
                    )
                
            except Exception as e:
                print(f"Error processing {row['file_path']}: {e}")
                continue
    
    print("\n✅ Import complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare bilingual dataset for training")
    parser.add_argument('--english-manifest', type=str, required=True,
                       help='Path to English (LibriSpeech) manifest CSV')
    parser.add_argument('--vietnamese-manifest', type=str, required=True,
                       help='Path to Vietnamese (VLSP/VIVOS) manifest CSV')
    parser.add_argument('--english-base', type=str, required=True,
                       help='Base directory for English audio files')
    parser.add_argument('--vietnamese-base', type=str, required=True,
                       help='Base directory for Vietnamese audio files')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Optional: Save combined CSV before importing')
    parser.add_argument('--database', type=str, default='database/asr_training.db',
                       help='Path to database file')
    parser.add_argument('--split-version', type=str, default='bilingual_v1',
                       help='Version tag for data splits')
    parser.add_argument('--auto-split', action='store_true', default=True,
                       help='Automatically create train/val/test splits')
    parser.add_argument('--no-import', action='store_true',
                       help='Only create CSV, do not import to database')
    
    args = parser.parse_args()
    
    # Load datasets
    english_df = load_librispeech_manifest(args.english_manifest)
    vietnamese_df = load_vietnamese_manifest(args.vietnamese_manifest)
    
    # Combine
    combined_df = combine_datasets(
        english_df, 
        vietnamese_df,
        args.english_base,
        args.vietnamese_base
    )
    
    # Save combined CSV if requested
    if args.output_csv:
        combined_df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Saved combined dataset to: {args.output_csv}")
    
    # Import to database
    if not args.no_import:
        db = ASRDatabase(args.database)
        import_to_database(
            combined_df,
            db,
            audio_base_dir=args.english_base,  # Will be adjusted per file
            split_version=args.split_version,
            auto_split=args.auto_split
        )
    else:
        print("\n⚠️  Skipping database import (--no-import flag set)")


if __name__ == "__main__":
    main()

