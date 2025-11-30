"""
Update database file paths to point to merged_dataset.

Script này sẽ:
1. Đọc manifest files từ merged_dataset
2. Cập nhật database để trỏ tới đúng paths trong merged_dataset
3. Tạo data splits mới nếu cần
"""

import pandas as pd
from pathlib import Path
import sys
from database.db_utils import ASRDatabase
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
MERGED_DATASET_DIR = BASE_DIR / "data/processed/merged_dataset"
FULL_MERGED_DATASET_DIR = BASE_DIR / "data/processed/full_merged_dataset"


def update_split_to_database(split_name: str, db: ASRDatabase, split_version: str = "v1",
                             dataset_dir: Path = None, dataset_name: str = "merged_dataset"):
    """Update a single split (train/val/test) to database."""
    if dataset_dir is None:
        dataset_dir = MERGED_DATASET_DIR
    
    print(f"\n{'='*80}")
    print(f"📝 Updating {split_name.upper()} split to database")
    print(f"{'='*80}")
    
    manifest_path = dataset_dir / split_name / "manifest.csv"
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return 0
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"✅ Loaded {len(df)} entries from manifest")
    
    # Process each entry
    added_count = 0
    updated_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        try:
            # Get data from manifest
            audio_id = row.get('id', '')
            transcript = row.get('transcript', '')
            audio_path_rel = row.get('audio_path', '')  # e.g., "audio/007_000000980.wav"
            duration = row.get('duration', None)
            
            # Build full path
            if audio_path_rel.startswith('audio/'):
                full_path = dataset_dir / split_name / audio_path_rel
            else:
                full_path = dataset_dir / split_name / audio_path_rel
            
            full_path_str = str(full_path.absolute())
            
            # Check if file exists
            if not full_path.exists():
                print(f"\n⚠️  File not found: {full_path_str}")
                error_count += 1
                continue
            
            # Extract language from transcript if it has language tag
            language = "vi"  # default
            if transcript.startswith("<|vi|>"):
                language = "vi"
                transcript = transcript.replace("<|vi|>", "").strip()
            elif transcript.startswith("<|en|>"):
                language = "en"
                transcript = transcript.replace("<|en|>", "").strip()
            
            # Check if audio file already exists in database by path
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", (full_path_str,)
                )
                existing_row = cursor.fetchone()
            
            if existing_row:
                audio_file_id = existing_row[0]
                updated_count += 1
            else:
                # Add new audio file
                audio_file_id = db.add_audio_file(
                    file_path=full_path_str,
                    filename=full_path.name,
                    duration=duration,
                    sample_rate=16000,  # Default, adjust if needed
                    dataset_name=dataset_name,
                    language=language,
                    skip_duplicate=True  # Skip if duplicate
                )
                
                if audio_file_id is None:
                    # File already exists, get its ID
                    with db.get_connection() as conn:
                        cursor = conn.execute(
                            "SELECT id FROM AudioFiles WHERE file_path = ?", (full_path_str,)
                        )
                        existing_row = cursor.fetchone()
                        if existing_row:
                            audio_file_id = existing_row[0]
                            updated_count += 1
                        else:
                            error_count += 1
                            continue
                else:
                    added_count += 1
            
            # Add/update transcript
            normalized_transcript = transcript.lower().strip()
            db.add_transcript(
                audio_file_id=audio_file_id,
                transcript=transcript,
                normalized_transcript=normalized_transcript,
                language=language
            )
            
            # Assign to split
            db.assign_split(audio_file_id, split_name, split_version)
            
        except Exception as e:
            print(f"\n❌ Error processing row {idx}: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Results for {split_name}:")
    print(f"   ✅ Added: {added_count}")
    print(f"   🔄 Updated: {updated_count}")
    print(f"   ❌ Errors: {error_count}")
    
    return added_count + updated_count




def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update database to merged_dataset or full_merged_dataset')
    parser.add_argument('--dataset', type=str, choices=['merged', 'full_merged'], 
                       default='merged',
                       help='Which dataset to use: merged (47k samples) or full_merged (257k samples)')
    args = parser.parse_args()
    
    # Select dataset directory
    if args.dataset == 'full_merged':
        dataset_dir = FULL_MERGED_DATASET_DIR
        dataset_name = "full_merged_dataset"
    else:
        dataset_dir = MERGED_DATASET_DIR
        dataset_name = "merged_dataset"
    
    print("="*80)
    print(f"🔄 UPDATE DATABASE TO {dataset_name.upper()}")
    print("="*80)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Initialize database
    db = ASRDatabase()
    
    # Update each split
    split_version = "v1"
    
    total_updated = 0
    for split in ['train', 'val', 'test']:
        count = update_split_to_database(split, db, split_version, dataset_dir, dataset_name)
        total_updated += count
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples updated: {total_updated}")
    
    # Verify
    print(f"\n🔍 Verification:")
    train_df = db.get_split_data('train', split_version)
    val_df = db.get_split_data('val', split_version)
    test_df = db.get_split_data('test', split_version)
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Check file existence
    if len(train_df) > 0:
        existing = sum(1 for _, row in train_df.iterrows() 
                      if Path(row.get('file_path', '')).exists())
        print(f"\n   ✅ Existing files in train: {existing}/{len(train_df)}")


if __name__ == '__main__':
    main()

