"""
Improved script to prepare and import data into the database.
Enhanced with batch processing, validation, duplicate detection, and quality checks.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml
import sys
import os
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import VietnameseTextNormalizer
import librosa
import numpy as np


def validate_csv_file(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Validate CSV file before processing.
    
    Returns:
        DataFrame and list of validation errors
    """
    errors = []
    
    # Check file exists
    if not Path(csv_path).exists():
        errors.append(f"CSV file not found: {csv_path}")
        return None, errors
    
    try:
        # Try to read with UTF-8
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Try with latin-1
            df = pd.read_csv(csv_path, encoding='latin-1')
            errors.append("Warning: Using latin-1 encoding. UTF-8 recommended for Vietnamese.")
        except Exception as e:
            errors.append(f"Error reading CSV: {str(e)}")
            return None, errors
    
    # Validate required columns
    required_columns = ['file_path', 'transcript']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    if errors:
        return None, errors
    
    # Check for empty rows
    df = df.dropna(subset=['file_path', 'transcript'])
    if len(df) == 0:
        errors.append("No valid rows found in CSV")
    
    # Check for duplicate file paths
    duplicates = df[df.duplicated(subset=['file_path'], keep=False)]
    if len(duplicates) > 0:
        errors.append(f"Warning: {len(duplicates)} duplicate file paths found")
    
    return df, errors


def validate_audio_file(file_path: Path, config: Dict) -> Tuple[bool, Dict]:
    """Validate an audio file for quality and constraints.
    
    Returns:
        (is_valid, metadata_dict)
    """
    metadata = {'valid': False}
    
    if not file_path.exists():
        metadata['error'] = 'File not found'
        return False, metadata
    
    try:
        # Load audio to check validity
        audio, sr = librosa.load(str(file_path), sr=None, mono=True)
        duration = len(audio) / sr
        
        # Check duration constraints
        min_duration = config.get('audio', {}).get('min_duration_seconds', 0.5)
        max_duration = config.get('audio', {}).get('max_duration_seconds', 30)
        
        if duration < min_duration:
            metadata['error'] = f'Too short ({duration:.2f}s < {min_duration}s)'
            return False, metadata
        
        if duration > max_duration:
            metadata['warning'] = f'Very long ({duration:.2f}s > {max_duration}s)'
        
        # Check sample rate
        if sr < 8000:
            metadata['warning'] = 'Low sample rate (<8kHz)'
        
        # Check audio format
        file_format = file_path.suffix[1:].lower()
        supported = config.get('audio', {}).get('supported_formats', ['wav', 'mp3', 'flac', 'ogg'])
        if file_format not in supported:
            metadata['warning'] = f'Format {file_format} may not be supported'
        
        # Calculate quality
        if sr >= 44100:
            quality = 'high'
        elif sr >= 16000:
            quality = 'medium'
        else:
            quality = 'low'
        
        metadata.update({
            'valid': True,
            'duration': duration,
            'sample_rate': sr,
            'channels': 1,  # librosa loads as mono
            'format': file_format,
            'quality': quality,
            'audio_length': len(audio)
        })
        
        return True, metadata
        
    except Exception as e:
        metadata['error'] = f'Error loading audio: {str(e)}'
        return False, metadata


def validate_transcript(transcript: str, config: Dict) -> Tuple[bool, Dict]:
    """Validate transcript text.
    
    Returns:
        (is_valid, validation_info)
    """
    info = {'valid': True, 'warnings': []}
    
    if not transcript or pd.isna(transcript):
        info['valid'] = False
        info['error'] = 'Empty transcript'
        return False, info
    
    transcript = str(transcript).strip()
    
    # Check length constraints
    min_length = config.get('text', {}).get('min_length', 1)
    max_length = config.get('text', {}).get('max_length', 500)
    
    if len(transcript) < min_length:
        info['valid'] = False
        info['error'] = f'Too short (<{min_length} chars)'
        return False, info
    
    if len(transcript) > max_length:
        info['warnings'].append(f'Very long ({len(transcript)} chars)')
    
    # Check for suspicious patterns
    if transcript.lower() == transcript.upper():  # No Vietnamese diacritics
        info['warnings'].append('No Vietnamese diacritics found')
    
    info['length'] = len(transcript)
    info['word_count'] = len(transcript.split())
    
    return True, info


def import_csv_data(csv_path: str, db: ASRDatabase, 
                    audio_base_path: str = None,
                    split_version: str = "v1",
                    config: Dict = None,
                    batch_size: int = 100,
                    skip_duplicates: bool = True,
                    dry_run: bool = False):
    """Import data from CSV file into database with improved validation.
    
    Args:
        csv_path: Path to CSV file
        db: Database instance
        audio_base_path: Base path for audio files
        split_version: Version identifier for data splits
        config: Configuration dictionary
        batch_size: Number of files to process before committing
        skip_duplicates: Skip files that already exist in database
        dry_run: If True, only validate without importing
    """
    config = config or {}
    
    # Validate CSV
    print("Validating CSV file...")
    df, errors = validate_csv_file(csv_path)
    if df is None:
        print("❌ CSV Validation Failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if errors:
        print("⚠️  CSV Warnings:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"✓ CSV validated: {len(df)} rows")
    
    # Initialize components
    normalizer = VietnameseTextNormalizer()
    
    # Statistics
    stats = {
        'total': len(df),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'warnings': []
    }
    
    error_log = []
    batch_data = []
    
    print(f"\nProcessing {len(df)} audio files...")
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Importing"):
        file_path = row['file_path']
        transcript = row.get('transcript', '')
        
        # Handle relative paths
        if audio_base_path:
            full_path = Path(audio_base_path) / file_path
        else:
            full_path = Path(file_path)
        
        full_path = full_path.resolve()
        
        # Validate audio file
        is_valid, audio_meta = validate_audio_file(full_path, config)
        if not is_valid:
            error_log.append({
                'file': str(full_path),
                'error': audio_meta.get('error', 'Unknown error'),
                'row': idx + 2  # +2 because CSV has header and 0-indexed
            })
            stats['errors'] += 1
            if not dry_run:
                continue
        
        # Validate transcript
        is_valid, text_meta = validate_transcript(transcript, config)
        if not is_valid:
            error_log.append({
                'file': str(full_path),
                'error': text_meta.get('error', 'Invalid transcript'),
                'row': idx + 2
            })
            stats['errors'] += 1
            if not dry_run:
                continue
        
        if dry_run:
            stats['processed'] += 1
            if audio_meta.get('warning'):
                stats['warnings'].append(f"{full_path}: {audio_meta['warning']}")
            if text_meta.get('warnings'):
                for w in text_meta['warnings']:
                    stats['warnings'].append(f"{full_path}: {w}")
            continue
        
        # Check for duplicate in database
        if skip_duplicates:
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", 
                    (str(full_path),)
                )
                if cursor.fetchone():
                    stats['skipped'] += 1
                    continue
        
        try:
            # Prepare data for batch insert
            normalized_transcript = normalizer.normalize(transcript)
            
            batch_data.append({
                'file_path': str(full_path),
                'filename': full_path.name,
                'duration': audio_meta['duration'],
                'sample_rate': audio_meta['sample_rate'],
                'channels': audio_meta['channels'],
                'format': audio_meta['format'],
                'language': 'vi',
                'speaker_id': row.get('speaker_id', None),
                'audio_quality': audio_meta['quality'],
                'transcript': transcript,
                'normalized_transcript': normalized_transcript,
                'split': row.get('split', None)
            })
            
            stats['processed'] += 1
            
            # Batch insert
            if len(batch_data) >= batch_size:
                _batch_insert(db, batch_data, split_version)
                batch_data = []
                
        except Exception as e:
            error_log.append({
                'file': str(full_path),
                'error': str(e),
                'row': idx + 2
            })
            stats['errors'] += 1
    
    # Insert remaining batch
    if batch_data and not dry_run:
        _batch_insert(db, batch_data, split_version)
    
    # Print summary
    print("\n" + "="*60)
    print("IMPORT SUMMARY")
    print("="*60)
    print(f"Total rows:           {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (duplicates): {stats['skipped']}")
    print(f"Errors:              {stats['errors']}")
    
    if error_log:
        print(f"\n⚠️  {len(error_log)} errors found. Check error log.")
        error_file = Path(csv_path).parent / 'import_errors.csv'
        pd.DataFrame(error_log).to_csv(error_file, index=False, encoding='utf-8')
        print(f"Error log saved to: {error_file}")
    
    if stats['warnings']:
        print(f"\n⚠️  {len(stats['warnings'])} warnings:")
        for warning in stats['warnings'][:10]:  # Show first 10
            print(f"  - {warning}")
        if len(stats['warnings']) > 10:
            print(f"  ... and {len(stats['warnings']) - 10} more")
    
    if not dry_run:
        # Show statistics
        stats_df = db.get_dataset_statistics(split_version)
        print("\n📊 Dataset Statistics:")
        print(stats_df.to_string(index=False))
    
    return stats['errors'] == 0


def _batch_insert(db: ASRDatabase, batch_data: List[Dict], split_version: str):
    """Insert batch of data efficiently."""
    with db.get_connection() as conn:
        for data in batch_data:
            # Insert audio file
            cursor = conn.execute("""
                INSERT INTO AudioFiles 
                (file_path, filename, duration_seconds, sample_rate, channels, 
                 format, language, speaker_id, audio_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['file_path'], data['filename'], data['duration'],
                data['sample_rate'], data['channels'], data['format'],
                data['language'], data['speaker_id'], data['audio_quality']
            ))
            audio_id = cursor.lastrowid
            
            # Insert transcript
            word_count = len(data['transcript'].split())
            char_count = len(data['transcript'])
            conn.execute("""
                INSERT INTO Transcripts 
                (audio_file_id, transcript, normalized_transcript, 
                 word_count, char_count, annotation_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                audio_id, data['transcript'], data['normalized_transcript'],
                word_count, char_count, 'manual'
            ))
            
            # Assign split if specified
            if data['split']:
                conn.execute("""
                    INSERT OR REPLACE INTO DataSplits 
                    (audio_file_id, split_type, split_version)
                    VALUES (?, ?, ?)
                """, (audio_id, data['split'], split_version))
        
        conn.commit()


def auto_split_data(db: ASRDatabase, split_version: str = "v1",
                    train_ratio: float = 0.8, val_ratio: float = 0.1,
                    strategy: str = "random"):
    """Automatically split data that hasn't been assigned to splits.
    
    Args:
        strategy: 'random' or 'speaker_balanced' (distribute speakers evenly)
    """
    import random
    
    # Get all audio files without splits
    with db.get_connection() as conn:
        if strategy == 'speaker_balanced':
            # Get files grouped by speaker for balanced split
            cursor = conn.execute("""
                SELECT af.id, af.speaker_id
                FROM AudioFiles af
                LEFT JOIN DataSplits ds ON af.id = ds.audio_file_id 
                    AND ds.split_version = ?
                WHERE ds.id IS NULL
                ORDER BY af.speaker_id, RANDOM()
            """, (split_version,))
        else:
            # Random split
            cursor = conn.execute("""
                SELECT af.id 
                FROM AudioFiles af
                LEFT JOIN DataSplits ds ON af.id = ds.audio_file_id 
                    AND ds.split_version = ?
                WHERE ds.id IS NULL
                ORDER BY RANDOM()
            """, (split_version,))
        
        audio_ids = [row['id'] for row in cursor.fetchall()]
    
    if not audio_ids:
        print("All audio files already assigned to splits")
        return
    
    # Calculate split indices
    n = len(audio_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Assign splits
    print(f"Auto-splitting {n} audio files (strategy: {strategy})...")
    
    with db.get_connection() as conn:
        for i, audio_id in enumerate(tqdm(audio_ids, desc="Assigning splits")):
            if i < train_end:
                split_type = 'train'
            elif i < val_end:
                split_type = 'val'
            else:
                split_type = 'test'
            
            conn.execute("""
                INSERT OR REPLACE INTO DataSplits 
                (audio_file_id, split_type, split_version)
                VALUES (?, ?, ?)
            """, (audio_id, split_type, split_version))
        
        conn.commit()
    
    # Show statistics
    stats = db.get_dataset_statistics(split_version)
    print("\nDataset Statistics:")
    print(stats.to_string(index=False))


def validate_training_readiness(db: ASRDatabase, split_version: str = "v1"):
    """Validate that database is ready for training."""
    print("\n" + "="*60)
    print("TRAINING READINESS VALIDATION")
    print("="*60)
    
    validation = db.validate_data_for_training(split_version)
    
    if validation['is_ready']:
        print("✅ Database is READY for training!")
    else:
        print("❌ Database is NOT ready for training:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    
    if validation['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation['recommendations']:
        print("\n💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")
    
    print("\n📊 Statistics:")
    stats = validation['statistics']
    print(f"  Splits: {stats.get('splits', {})}")
    print(f"  Quality distribution: {stats.get('quality_distribution', {})}")
    print(f"  Duration - Avg: {stats.get('duration_stats', {}).get('avg_duration', 0):.2f}s")
    print(f"  Files without transcripts: {stats.get('files_without_transcripts', 0)}")
    
    return validation['is_ready']


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for ASR training (Improved)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic import with auto-split
  python scripts/prepare_data.py --csv data.csv --audio_base data/raw --auto_split
  
  # Dry run (validate without importing)
  python scripts/prepare_data.py --csv data.csv --audio_base data/raw --dry_run
  
  # Validate training readiness
  python scripts/prepare_data.py --validate_only
  
  # Import with speaker-balanced splitting
  python scripts/prepare_data.py --csv data.csv --audio_base data/raw --auto_split --split_strategy speaker_balanced
        """
    )
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file with audio paths and transcripts')
    parser.add_argument('--audio_base', type=str, default=None,
                       help='Base directory for audio files (if CSV has relative paths)')
    parser.add_argument('--config', type=str, default='configs/db.yaml',
                       help='Path to database config')
    parser.add_argument('--auto_split', action='store_true',
                       help='Automatically split data if no splits in CSV')
    parser.add_argument('--split_strategy', type=str, default='random',
                       choices=['random', 'speaker_balanced'],
                       help='Strategy for auto-splitting')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for database inserts (default: 100)')
    parser.add_argument('--skip_duplicates', action='store_true', default=True,
                       help='Skip files that already exist in database')
    parser.add_argument('--dry_run', action='store_true',
                       help='Validate CSV without importing to database')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate training readiness, do not import')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db_path = config['database']['path']
    db = ASRDatabase(db_path)
    
    # Validate only mode
    if args.validate_only:
        validate_training_readiness(db, config['split_version'])
        return
    
    # Import data
    if args.csv:
        success = import_csv_data(
            csv_path=args.csv,
            db=db,
            audio_base_path=args.audio_base,
            split_version=config['split_version'],
            config=config,
            batch_size=args.batch_size,
            skip_duplicates=args.skip_duplicates,
            dry_run=args.dry_run
        )
        
        if not args.dry_run and success:
            # Auto split if requested
            if args.auto_split:
                auto_split_data(
                    db=db,
                    split_version=config['split_version'],
                    train_ratio=config['splits']['train'],
                    val_ratio=config['splits']['val'],
                    strategy=args.split_strategy
                )
            
            # Validate training readiness
            print("\n")
            validate_training_readiness(db, config['split_version'])
    else:
        # Just validate if no CSV provided
        validate_training_readiness(db, config['split_version'])


if __name__ == '__main__':
    main()
