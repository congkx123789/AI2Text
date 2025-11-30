"""
Preview database changes before updating to merged_dataset or full_merged_dataset.

Script này sẽ:
1. Kiểm tra cả merged_dataset và full_merged_dataset
2. So sánh với database hiện tại
3. Hiển thị preview các thay đổi sẽ xảy ra
4. KHÔNG thực hiện cập nhật (dry-run mode)
"""

import pandas as pd
from pathlib import Path
import sys
from database.db_utils import ASRDatabase
from collections import defaultdict

BASE_DIR = Path(__file__).parent
MERGED_DATASET_DIR = BASE_DIR / "data/processed/merged_dataset"
FULL_MERGED_DATASET_DIR = BASE_DIR / "data/processed/full_merged_dataset"


def analyze_dataset(dataset_dir: Path, dataset_name: str):
    """Analyze a dataset directory."""
    print(f"\n{'='*80}")
    print(f"📊 Analyzing {dataset_name.upper()}")
    print(f"{'='*80}")
    
    if not dataset_dir.exists():
        print(f"❌ Directory not found: {dataset_dir}")
        return None
    
    results = {}
    
    for split in ['train', 'val', 'test']:
        manifest_path = dataset_dir / split / "manifest.csv"
        audio_dir = dataset_dir / split / "audio"
        
        if not manifest_path.exists():
            print(f"⚠️  {split}: No manifest.csv found")
            results[split] = {'count': 0, 'files': 0, 'exists': 0}
            continue
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        
        # Count actual files
        audio_files = list(audio_dir.glob("*.wav")) if audio_dir.exists() else []
        
        # Check file existence
        existing_count = 0
        sample_paths = []
        
        for idx, row in df.head(10).iterrows():  # Check first 10
            audio_path = row.get('audio_path', '')
            
            if audio_path.startswith('audio/'):
                full_path = dataset_dir / split / audio_path
            else:
                full_path = dataset_dir / split / audio_path
            
            if full_path.exists():
                existing_count += 1
                sample_paths.append(str(full_path.absolute()))
        
        # Check all files
        total_existing = 0
        for idx, row in df.iterrows():
            audio_path = row.get('audio_path', '')
            if audio_path.startswith('audio/'):
                full_path = dataset_dir / split / audio_path
            else:
                full_path = dataset_dir / split / audio_path
            if full_path.exists():
                total_existing += 1
        
        results[split] = {
            'count': len(df),
            'files': len(audio_files),
            'exists': total_existing,
            'sample_paths': sample_paths[:5]  # First 5 for preview
        }
        
        print(f"\n{split.upper()}:")
        print(f"   Manifest entries: {len(df)}")
        print(f"   Audio files found: {len(audio_files)}")
        print(f"   Files exist: {total_existing}/{len(df)} ({total_existing/len(df)*100:.1f}%)")
        
        if len(df) > 0:
            print(f"   Sample paths:")
            for path in sample_paths[:3]:
                print(f"      - {path}")
    
    return results


def check_current_database():
    """Check current database state."""
    print(f"\n{'='*80}")
    print(f"🔍 Current Database State")
    print(f"{'='*80}")
    
    db = ASRDatabase()
    
    # Get current splits
    train_df = db.get_split_data('train', 'v1')
    val_df = db.get_split_data('val', 'v1')
    test_df = db.get_split_data('test', 'v1')
    
    print(f"\nCurrent database entries:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    # Check file existence
    if len(train_df) > 0:
        existing_train = sum(1 for _, row in train_df.iterrows() 
                           if Path(row.get('file_path', '')).exists())
        print(f"\n   Train files existing: {existing_train}/{len(train_df)} ({existing_train/len(train_df)*100:.1f}%)")
        
        # Show sample paths
        print(f"\n   Sample file paths in database:")
        for idx in range(min(5, len(train_df))):
            row = train_df.iloc[idx]
            file_path = row.get('file_path', 'N/A')
            exists = Path(file_path).exists() if file_path != 'N/A' else False
            status = "✅" if exists else "❌"
            print(f"      {status} {file_path}")
    
    # Count by dataset
    if len(train_df) > 0 and 'dataset_name' in train_df.columns:
        dataset_counts = train_df['dataset_name'].value_counts()
        print(f"\n   By dataset:")
        for dataset, count in dataset_counts.items():
            print(f"      {dataset}: {count}")
    
    return {
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df),
        'total': len(train_df) + len(val_df) + len(test_df),
        'train_existing': existing_train if len(train_df) > 0 else 0
    }


def preview_changes(dataset_dir: Path, dataset_name: str, db_state: dict):
    """Preview what changes will happen."""
    print(f"\n{'='*80}")
    print(f"📋 PREVIEW: Changes for {dataset_name.upper()}")
    print(f"{'='*80}")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return
    
    changes = defaultdict(lambda: {'add': 0, 'update': 0, 'keep': 0})
    
    for split in ['train', 'val', 'test']:
        manifest_path = dataset_dir / split / "manifest.csv"
        if not manifest_path.exists():
            continue
        
        df = pd.read_csv(manifest_path)
        
        # Check how many will be new vs existing
        new_count = 0
        existing_count = 0
        
        for idx, row in df.iterrows():
            audio_path = row.get('audio_path', '')
            
            if audio_path.startswith('audio/'):
                full_path = dataset_dir / split / audio_path
            else:
                full_path = dataset_dir / split / audio_path
            
            full_path_str = str(full_path.absolute())
            
            # Check if exists in database
            with db_state['db'].get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM AudioFiles WHERE file_path = ?", (full_path_str,)
                )
                if cursor.fetchone():
                    existing_count += 1
                else:
                    new_count += 1
        
        changes[split] = {
            'total': len(df),
            'new': new_count,
            'existing': existing_count,
            'current_db': db_state.get(split, 0)
        }
    
    # Print summary
    print(f"\n📊 Summary of Changes:")
    print(f"{'Split':<10} {'Current DB':<12} {'Dataset':<12} {'New':<10} {'Existing':<12} {'Change':<10}")
    print(f"{'-'*80}")
    
    total_new = 0
    total_existing = 0
    total_current = db_state['train'] + db_state['val'] + db_state['test']
    
    for split in ['train', 'val', 'test']:
        if split in changes:
            ch = changes[split]
            current = ch['current_db']
            dataset = ch['total']
            new = ch['new']
            existing = ch['existing']
            change = f"+{new}" if new > 0 else "0"
            
            print(f"{split:<10} {current:<12} {dataset:<12} {new:<10} {existing:<12} {change:<10}")
            
            total_new += new
            total_existing += existing
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<10} {total_current:<12} {sum(ch['total'] for ch in changes.values()):<12} {total_new:<10} {total_existing:<12} {f'+{total_new}':<10}")
    
    print(f"\n💡 What will happen:")
    print(f"   - {total_new} new audio files will be added to database")
    print(f"   - {total_existing} files already exist (will be updated/reused)")
    print(f"   - Database will have {total_current - db_state.get('train_existing', 0) + total_new} working file paths")
    print(f"   - Old invalid paths will remain but new splits will point to correct files")
    
    return changes


def main():
    """Main function."""
    print("="*80)
    print("🔍 DATABASE CHANGES PREVIEW")
    print("="*80)
    print("\nThis script shows what will change in the database")
    print("NO CHANGES WILL BE MADE (dry-run mode)")
    print("="*80)
    
    # Check current database
    db = ASRDatabase()
    db_state = check_current_database()
    db_state['db'] = db  # Store db instance for preview
    
    # Analyze merged_dataset
    merged_results = analyze_dataset(MERGED_DATASET_DIR, "merged_dataset")
    
    # Analyze full_merged_dataset
    full_merged_results = analyze_dataset(FULL_MERGED_DATASET_DIR, "full_merged_dataset")
    
    # Preview changes for merged_dataset
    if merged_results:
        print(f"\n{'='*80}")
        print(f"📋 PREVIEW: merged_dataset Changes")
        print(f"{'='*80}")
        merged_changes = preview_changes(MERGED_DATASET_DIR, "merged_dataset", db_state)
    
    # Preview changes for full_merged_dataset
    if full_merged_results:
        print(f"\n{'='*80}")
        print(f"📋 PREVIEW: full_merged_dataset Changes")
        print(f"{'='*80}")
        full_merged_changes = preview_changes(FULL_MERGED_DATASET_DIR, "full_merged_dataset", db_state)
    
    # Comparison
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON")
    print(f"{'='*80}")
    
    if merged_results and full_merged_results:
        print(f"\nDataset Comparison:")
        print(f"{'Split':<10} {'merged_dataset':<20} {'full_merged_dataset':<20}")
        print(f"{'-'*60}")
        
        for split in ['train', 'val', 'test']:
            merged_count = merged_results.get(split, {}).get('count', 0)
            full_count = full_merged_results.get(split, {}).get('count', 0)
            print(f"{split:<10} {merged_count:<20} {full_count:<20}")
        
        merged_total = sum(r.get('count', 0) for r in merged_results.values())
        full_total = sum(r.get('count', 0) for r in full_merged_results.values())
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':<10} {merged_total:<20} {full_total:<20}")
        
        print(f"\n💡 Recommendation:")
        if full_total > merged_total * 1.5:
            print(f"   → full_merged_dataset has {full_total - merged_total} more samples")
            print(f"   → Consider using full_merged_dataset for better model performance")
        else:
            print(f"   → Both datasets are similar in size")
            print(f"   → Choose based on your training needs")
    
    print(f"\n{'='*80}")
    print(f"✅ Preview Complete")
    print(f"{'='*80}")
    print(f"\nTo apply changes, run:")
    print(f"   python update_db_to_merged_dataset.py")
    print(f"\nOr modify the script to use full_merged_dataset")


if __name__ == '__main__':
    main()

