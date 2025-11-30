#!/usr/bin/env python3
"""
Script to import merged_dataset into database with split_version "v2_merged".

Usage:
    python scripts/import_merged_dataset.py
"""

import sys
import os
import tempfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from scripts.prepare_data import (
    validate_csv_file,
    import_csv_data,
    validate_training_readiness
)

def main():
    """Import merged_dataset into database."""
    
    # Paths
    base_dir = Path("data/processed/merged_dataset")
    train_manifest = base_dir / "train" / "manifest.csv"
    val_manifest = base_dir / "val" / "manifest.csv"
    test_manifest = base_dir / "test" / "manifest.csv"
    
    split_version = "v2_merged"
    database_path = "database/asr_training.db"
    
    print("=" * 60)
    print("Import Merged Dataset to Database")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Split version: {split_version}")
    print(f"Database: {database_path}")
    print()
    
    # Check if manifests exist
    if not train_manifest.exists():
        print(f"❌ Error: Train manifest not found: {train_manifest}")
        return 1
    
    if not val_manifest.exists():
        print(f"❌ Error: Val manifest not found: {val_manifest}")
        return 1
    
    # Initialize database
    print("📦 Initializing database...")
    db = ASRDatabase(database_path)
    print("✓ Database initialized")
    print()
    
    # Read manifests directly (merged_dataset uses audio_path, not file_path)
    print("📋 Reading manifest files...")
    
    try:
        train_df = pd.read_csv(train_manifest, encoding='utf-8')
        print(f"✓ Train manifest read: {len(train_df):,} rows")
    except Exception as e:
        print(f"❌ Error reading train manifest: {e}")
        return 1
    
    try:
        val_df = pd.read_csv(val_manifest, encoding='utf-8')
        print(f"✓ Val manifest read: {len(val_df):,} rows")
    except Exception as e:
        print(f"❌ Error reading val manifest: {e}")
        return 1
    
    # Check if test manifest exists (optional)
    test_df = None
    if test_manifest.exists():
        try:
            test_df = pd.read_csv(test_manifest, encoding='utf-8')
            print(f"✓ Test manifest read: {len(test_df):,} rows")
        except Exception as e:
            print(f"⚠️  Error reading test manifest (continuing anyway): {e}")
    
    # Map audio_path to file_path (required by import function)
    if 'audio_path' in train_df.columns and 'file_path' not in train_df.columns:
        train_df['file_path'] = train_df['audio_path']
    if 'audio_path' in val_df.columns and 'file_path' not in val_df.columns:
        val_df['file_path'] = val_df['audio_path']
    if test_df is not None and 'audio_path' in test_df.columns and 'file_path' not in test_df.columns:
        test_df['file_path'] = test_df['audio_path']
    
    # Validate required columns
    required_cols = ['file_path', 'transcript']
    for col in required_cols:
        if col not in train_df.columns:
            print(f"❌ Train manifest missing required column: {col}")
            print(f"   Available columns: {list(train_df.columns)}")
            return 1
        if col not in val_df.columns:
            print(f"❌ Val manifest missing required column: {col}")
            print(f"   Available columns: {list(val_df.columns)}")
            return 1
    
    # Check if test manifest exists (optional)
    test_df = None
    if test_manifest.exists():
        test_df, test_errors = validate_csv_file(str(test_manifest))
        if test_df is None:
            print(f"⚠️  Test manifest validation failed (continuing anyway):")
            for error in test_errors:
                print(f"   - {error}")
    
    print(f"✓ Train samples: {len(train_df):,}")
    print(f"✓ Val samples: {len(val_df):,}")
    if test_df is not None:
        print(f"✓ Test samples: {len(test_df):,}")
    print()
    
    # Prepare dataframes for import
    print("🔄 Preparing data for import...")
    
    # Add language if not present (detect from transcript)
    if 'language' not in train_df.columns:
        def detect_language(text):
            if pd.isna(text):
                return 'vi'  # Default to Vietnamese
            text_str = str(text)
            if text_str.startswith('<|vi|>'):
                return 'vi'
            elif text_str.startswith('<|en|>'):
                return 'en'
            return 'vi'  # Default
        
        train_df['language'] = train_df['transcript'].apply(detect_language)
        val_df['language'] = val_df['transcript'].apply(detect_language)
        if test_df is not None:
            test_df['language'] = test_df['transcript'].apply(detect_language)
    
    print("✓ Data prepared")
    print()
    
    # Prepare config
    config = {
        'audio': {
            'min_duration_seconds': 0.5,
            'max_duration_seconds': 30,
            'sample_rate': 16000
        },
        'text': {
            'min_length': 1,
            'max_length': 500
        }
    }
    
    # Import train data
    print("📥 Importing train data...")
    train_df['split'] = 'train'
    # Ensure file_path exists (already handled in prepare_df, but double-check)
    if 'file_path' not in train_df.columns and 'audio_path' in train_df.columns:
        train_df['file_path'] = train_df['audio_path']
    
    # Fix file paths: manifest has 'audio/xxx.wav', but files are in 'train/audio/xxx.wav'
    train_df['file_path'] = train_df['file_path'].apply(
        lambda x: f"train/{x}" if not x.startswith('train/') and not x.startswith('val/') and not x.startswith('test/') else x
    )
    
    # Save to temporary CSV for import
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        train_df.to_csv(f.name, index=False)
        train_csv = f.name
    
    try:
        success = import_csv_data(
            csv_path=train_csv,
            db=db,
            audio_base_path=str(base_dir),
            split_version=split_version,
            config=config,
            batch_size=100,
            skip_duplicates=True,
            dry_run=False
        )
        if not success:
            print("❌ Failed to import train data")
            return 1
        print("✓ Train data imported")
    finally:
        os.unlink(train_csv)
    
    # Import val data
    print("📥 Importing val data...")
    val_df['split'] = 'val'
    # Ensure file_path exists (already handled in prepare_df, but double-check)
    if 'file_path' not in val_df.columns and 'audio_path' in val_df.columns:
        val_df['file_path'] = val_df['audio_path']
    
    # Fix file paths: manifest has 'audio/xxx.wav', but files are in 'val/audio/xxx.wav'
    val_df['file_path'] = val_df['file_path'].apply(
        lambda x: f"val/{x}" if not x.startswith('train/') and not x.startswith('val/') and not x.startswith('test/') else x
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        val_df.to_csv(f.name, index=False)
        val_csv = f.name
    
    try:
        success = import_csv_data(
            csv_path=val_csv,
            db=db,
            audio_base_path=str(base_dir),
            split_version=split_version,
            config=config,
            batch_size=100,
            skip_duplicates=True,
            dry_run=False
        )
        if not success:
            print("❌ Failed to import val data")
            return 1
        print("✓ Val data imported")
    finally:
        os.unlink(val_csv)
    
    # Import test data if available
    if test_df is not None:
        print("📥 Importing test data...")
        test_df['split'] = 'test'
        # Ensure file_path exists (already handled in prepare_df, but double-check)
        if 'file_path' not in test_df.columns and 'audio_path' in test_df.columns:
            test_df['file_path'] = test_df['audio_path']
        
        # Fix file paths: manifest has 'audio/xxx.wav', but files are in 'test/audio/xxx.wav'
        test_df['file_path'] = test_df['file_path'].apply(
            lambda x: f"test/{x}" if not x.startswith('train/') and not x.startswith('val/') and not x.startswith('test/') else x
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            test_df.to_csv(f.name, index=False)
            test_csv = f.name
        
        try:
            success = import_csv_data(
                csv_path=test_csv,
                db=db,
                audio_base_path=str(base_dir),
                split_version=split_version,
                config=config,
                batch_size=100,
                skip_duplicates=True,
                dry_run=False
            )
            if not success:
                print("⚠️  Failed to import test data (continuing anyway)")
            else:
                print("✓ Test data imported")
        finally:
            os.unlink(test_csv)
    
    print()
    
    # Validate training readiness
    print("✅ Validating training readiness...")
    try:
        validate_training_readiness(db, split_version)
        print("✓ Dataset is ready for training!")
    except Exception as e:
        print(f"⚠️  Validation warning: {e}")
    
    print()
    print("=" * 60)
    print("✅ Import completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Verify data in database:")
    print(f"     python -c \"from database.db_utils import ASRDatabase; db = ASRDatabase('{database_path}'); print(db.get_data_summary('{split_version}'))\"")
    print()
    print("  2. Start training from scratch:")
    print("     python training/train.py --config configs/merged_dataset_train.yaml")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

