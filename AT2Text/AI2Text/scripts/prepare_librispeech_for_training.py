#!/usr/bin/env python3
"""
Prepare LibriSpeech alignments data for training.
Converts the manifest format and imports to database with language='en'.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import BilingualTextNormalizer
import librosa


def prepare_librispeech_manifest(manifest_path: str, base_dir: str, output_csv: str = None):
    """Convert LibriSpeech manifest to format expected by prepare_data.py"""
    print(f"Loading LibriSpeech manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Convert to expected format
    base_path = Path(base_dir)
    prepared_data = []
    
    for idx, row in df.iterrows():
        # Construct full audio path
        audio_path = base_path / row['audio_path']
        
        # Use transcript as-is (already normalized for LibriSpeech)
        transcript = row['transcript']
        
        prepared_data.append({
            'file_path': str(audio_path),
            'transcript': transcript,
            'language': 'en'  # Mark as English
        })
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")
    
    result_df = pd.DataFrame(prepared_data)
    
    if output_csv:
        result_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Saved prepared manifest to: {output_csv}")
    
    return result_df


def import_to_database(csv_path: str, base_dir: str, split_version: str = "librispeech_v1"):
    """Import prepared data to database"""
    from scripts.prepare_data import main as prepare_main
    import sys
    
    # Prepare arguments for prepare_data.py
    sys.argv = [
        'prepare_data.py',
        '--csv', csv_path,
        '--audio_base', base_dir,
        '--auto_split',
        '--split_strategy', 'random'
    ]
    
    # We'll call the import logic directly
    from database.db_utils import ASRDatabase
    from preprocessing.audio_processing import AudioProcessor
    from preprocessing.text_cleaning import BilingualTextNormalizer
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'db.yaml'
    if not config_path.exists():
        # Use default config
        config = {
            'database': {'path': 'database/asr_training.db'},
            'split_version': split_version
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    db = ASRDatabase(config['database']['path'])
    audio_processor = AudioProcessor()
    normalizer = BilingualTextNormalizer()
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"\nImporting {len(df)} samples to database...")
    
    # Process in batches
    batch_size = 100
    imported = 0
    skipped = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            try:
                audio_path = Path(row['file_path'])
                
                # Check if file exists
                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    skipped += 1
                    continue
                
                # Load audio to get metadata
                try:
                    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                    duration = len(audio) / sr
                except Exception as e:
                    print(f"Warning: Could not load audio {audio_path}: {e}")
                    skipped += 1
                    continue
                
                # Get language
                language = row.get('language', 'en')
                transcript = row['transcript']
                
                # Normalize transcript
                normalized = normalizer.normalize(transcript, lang=language)
                
                # Add audio file
                audio_id = db.add_audio_file(
                    file_path=str(audio_path),
                    filename=audio_path.name,
                    duration=duration,
                    sample_rate=int(sr),
                    language=language,
                    dataset_name='librispeech'
                )
                
                # Add transcript
                db.add_transcript(
                    audio_file_id=audio_id,
                    transcript=transcript,
                    normalized_transcript=normalized,
                    language=language
                )
                
                imported += 1
                
            except Exception as e:
                print(f"Error processing {row.get('file_path', 'unknown')}: {e}")
                skipped += 1
                continue
        
        if (i + batch_size) % 1000 == 0:
            print(f"Imported {imported} samples, skipped {skipped}...")
    
    print(f"\nImport complete!")
    print(f"  Imported: {imported}")
    print(f"  Skipped: {skipped}")
    
    # Auto-split the data
    print(f"\nCreating data splits (version: {split_version})...")
    from scripts.prepare_data import create_splits
    
    create_splits(db, split_version, strategy='random')
    
    print("Data preparation complete!")


def main():
    parser = argparse.ArgumentParser(description='Prepare LibriSpeech data for training')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to LibriSpeech manifest CSV')
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory for audio files')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Output CSV path (optional)')
    parser.add_argument('--split-version', type=str, default='librispeech_v1',
                       help='Split version for database')
    parser.add_argument('--import-only', action='store_true',
                       help='Only import, skip manifest preparation')
    
    args = parser.parse_args()
    
    if not args.import_only:
        # Prepare manifest
        output_csv = args.output_csv or str(Path(args.manifest).parent / 'prepared_manifest.csv')
        prepare_librispeech_manifest(args.manifest, args.base_dir, output_csv)
        csv_to_import = output_csv
    else:
        csv_to_import = args.manifest
    
    # Import to database
    import_to_database(csv_to_import, args.base_dir, args.split_version)


if __name__ == '__main__':
    main()




















