"""
Script to check merged_dataset training data integrity.

Kiểm tra:
1. Số lượng files trong manifest vs số lượng files thực tế
2. Tất cả files có tồn tại không
3. File paths trong database có đúng không
"""

import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict

# Base directory
BASE_DIR = Path(__file__).parent
MERGED_DATASET_DIR = BASE_DIR / "data/processed/merged_dataset"


def check_split(split_name: str):
    """Check a single split (train/val/test)."""
    print(f"\n{'='*80}")
    print(f"📊 Checking {split_name.upper()} split")
    print(f"{'='*80}")
    
    manifest_path = MERGED_DATASET_DIR / split_name / "manifest.csv"
    audio_dir = MERGED_DATASET_DIR / split_name / "audio"
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"✅ Manifest loaded: {len(df)} entries")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Count actual audio files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"✅ Audio files found: {len(audio_files)}")
    
    # Check if counts match
    if len(df) != len(audio_files):
        print(f"⚠️  WARNING: Manifest entries ({len(df)}) != Audio files ({len(audio_files)})")
    else:
        print(f"✅ Counts match!")
    
    # Check file existence
    missing_files = []
    invalid_paths = []
    
    for idx, row in df.iterrows():
        audio_path = row.get('audio_path', '')
        
        # Handle relative paths
        if audio_path.startswith('audio/'):
            # Relative path from manifest
            full_path = MERGED_DATASET_DIR / split_name / audio_path
        elif audio_path.startswith('/'):
            # Absolute path
            full_path = Path(audio_path)
        else:
            # Assume relative to split directory
            full_path = MERGED_DATASET_DIR / split_name / audio_path
        
        if not full_path.exists():
            missing_files.append({
                'index': idx,
                'audio_path': audio_path,
                'expected_path': str(full_path),
                'id': row.get('id', 'N/A'),
                'transcript': row.get('transcript', '')[:50]
            })
        elif not full_path.is_file():
            invalid_paths.append({
                'index': idx,
                'audio_path': audio_path,
                'expected_path': str(full_path)
            })
    
    # Report results
    print(f"\n📋 File Existence Check:")
    if len(missing_files) == 0 and len(invalid_paths) == 0:
        print(f"   ✅ All {len(df)} files exist!")
    else:
        print(f"   ❌ Missing files: {len(missing_files)}")
        print(f"   ❌ Invalid paths: {len(invalid_paths)}")
        
        if len(missing_files) > 0:
            print(f"\n   First 10 missing files:")
            for item in missing_files[:10]:
                print(f"      [{item['index']}] {item['audio_path']}")
                print(f"          Expected: {item['expected_path']}")
                print(f"          ID: {item['id']}")
                print(f"          Text: {item['transcript']}")
                print()
    
    # Check transcript quality
    empty_transcripts = df[df['transcript'].isna() | (df['transcript'].str.strip() == '')]
    if len(empty_transcripts) > 0:
        print(f"⚠️  Empty transcripts: {len(empty_transcripts)}")
    
    # Check duration
    if 'duration' in df.columns:
        durations = df['duration'].dropna()
        if len(durations) > 0:
            print(f"\n📊 Duration Statistics:")
            print(f"   Min: {durations.min():.2f}s")
            print(f"   Max: {durations.max():.2f}s")
            print(f"   Mean: {durations.mean():.2f}s")
            print(f"   Median: {durations.median():.2f}s")
            
            # Check for very short or very long files
            too_short = (durations < 0.5).sum()
            too_long = (durations > 30).sum()
            if too_short > 0:
                print(f"   ⚠️  Files < 0.5s: {too_short}")
            if too_long > 0:
                print(f"   ⚠️  Files > 30s: {too_long}")
    
    return {
        'total': len(df),
        'audio_files': len(audio_files),
        'missing': len(missing_files),
        'invalid': len(invalid_paths),
        'missing_list': missing_files[:20]  # First 20 for reference
    }


def check_database_paths():
    """Check how database stores file paths."""
    print(f"\n{'='*80}")
    print(f"🔍 Checking Database File Paths")
    print(f"{'='*80}")
    
    try:
        from database.db_utils import ASRDatabase
        
        db = ASRDatabase()
        train_df = db.get_split_data('train', 'v1')
        
        if len(train_df) == 0:
            print("⚠️  No training data in database")
            return
        
        print(f"✅ Database has {len(train_df)} training samples")
        
        # Check first few file paths
        print(f"\n📋 Sample file paths from database:")
        for idx in range(min(5, len(train_df))):
            row = train_df.iloc[idx]
            file_path = row.get('file_path', 'N/A')
            exists = Path(file_path).exists() if file_path != 'N/A' else False
            status = "✅" if exists else "❌"
            print(f"   {status} [{idx}] {file_path}")
            if not exists:
                print(f"      Transcript: {row.get('transcript', '')[:50]}")
        
        # Count existing vs missing
        existing = sum(1 for _, row in train_df.iterrows() 
                      if Path(row.get('file_path', '')).exists())
        missing = len(train_df) - existing
        
        print(f"\n📊 Database File Status:")
        print(f"   ✅ Existing: {existing} ({existing/len(train_df)*100:.1f}%)")
        print(f"   ❌ Missing: {missing} ({missing/len(train_df)*100:.1f}%)")
        
        if missing > 0:
            print(f"\n   ⚠️  Some files in database don't exist!")
            print(f"   → Check if paths are relative vs absolute")
            print(f"   → Check if merged_dataset paths are correct")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to check all splits."""
    print("="*80)
    print("🔍 MERGED DATASET INTEGRITY CHECK")
    print("="*80)
    
    if not MERGED_DATASET_DIR.exists():
        print(f"❌ Merged dataset directory not found: {MERGED_DATASET_DIR}")
        return
    
    results = {}
    
    # Check each split
    for split in ['train', 'val', 'test']:
        results[split] = check_split(split)
    
    # Check database
    check_database_paths()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    
    total_samples = sum(r['total'] for r in results.values() if r)
    total_missing = sum(r['missing'] for r in results.values() if r)
    
    print(f"Total samples: {total_samples}")
    print(f"Missing files: {total_missing}")
    
    if total_missing == 0:
        print(f"\n✅ All files exist! Dataset is ready for training.")
    else:
        print(f"\n⚠️  {total_missing} files are missing. Please check paths.")
        print(f"\n💡 Tip: Update database file paths to use:")
        print(f"   data/processed/merged_dataset/{{split}}/audio/{{filename}}.wav")


if __name__ == '__main__':
    main()

