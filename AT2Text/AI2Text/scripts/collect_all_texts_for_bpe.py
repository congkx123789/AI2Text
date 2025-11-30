"""
Collect ALL texts from all manifests to train BPE tokenizer.
This script gathers texts from both merged_dataset and full_merged_dataset
to ensure complete vocabulary coverage.

Usage:
    python scripts/collect_all_texts_for_bpe.py [--output all_texts.txt]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def collect_texts_from_manifests():
    """Collect all texts from all manifest files."""
    
    texts = set()  # Use set to avoid duplicates
    
    # List of manifest directories to check
    manifest_dirs = [
        "data/processed/merged_dataset",
        "data/processed/full_merged_dataset",
    ]
    
    print("=" * 60)
    print("Collecting ALL texts from manifests for BPE training")
    print("=" * 60)
    print()
    
    total_files = 0
    
    for base_dir in manifest_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"⚠️  Directory not found: {base_dir}")
            continue
        
        print(f"📂 Scanning: {base_dir}")
        
        # Check train, val, test splits
        for split in ['train', 'val', 'test']:
            manifest_path = base_path / split / "manifest.csv"
            
            if not manifest_path.exists():
                print(f"   ⚠️  {split}/manifest.csv not found")
                continue
            
            try:
                # Read manifest
                df = pd.read_csv(manifest_path, encoding='utf-8')
                
                # Get transcript column (could be 'transcript' or 'normalized_transcript')
                transcript_col = None
                for col in ['transcript', 'normalized_transcript', 'text']:
                    if col in df.columns:
                        transcript_col = col
                        break
                
                if transcript_col is None:
                    print(f"   ⚠️  No transcript column found in {split}/manifest.csv")
                    print(f"      Available columns: {list(df.columns)}")
                    continue
                
                # Collect texts
                split_texts = df[transcript_col].dropna().astype(str)
                split_texts = split_texts[split_texts.str.strip() != '']
                
                # Remove language prefixes if present (e.g., <|vi|>, <|en|>)
                cleaned_texts = []
                for text in split_texts:
                    # Remove language tags
                    text = text.replace('<|vi|>', '').replace('<|en|>', '').strip()
                    if text:
                        cleaned_texts.append(text)
                
                texts.update(cleaned_texts)
                count = len(cleaned_texts)
                total_files += count
                
                print(f"   ✓ {split}: {count:,} texts")
                
            except Exception as e:
                print(f"   ❌ Error reading {manifest_path}: {e}")
                continue
    
    print()
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Total unique texts collected: {len(texts):,}")
    print(f"   Total text instances: {total_files:,}")
    print("=" * 60)
    
    return list(texts)


def main():
    parser = argparse.ArgumentParser(description='Collect all texts from manifests for BPE training')
    parser.add_argument('--output', type=str, default='all_texts_for_bpe.txt',
                       help='Output text file path (default: all_texts_for_bpe.txt)')
    
    args = parser.parse_args()
    
    # Collect all texts
    texts = collect_texts_from_manifests()
    
    if not texts:
        print("❌ No texts collected. Please check manifest files.")
        return 1
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"💾 Saving texts to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print(f"✓ Saved {len(texts):,} unique texts to {output_path}")
    print()
    print("Next step: Train BPE tokenizer with:")
    print(f"   python3 scripts/train_bpe_bilingual.py --vocab-size 16000 --output models/bilingual_bpe_16k.json")
    print()
    print("Or use the collected texts directly:")
    print(f"   python3 scripts/train_bpe_from_file.py --input {output_path} --vocab-size 16000 --output models/bilingual_bpe_16k.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

