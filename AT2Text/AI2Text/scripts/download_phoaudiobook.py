"""
Download thivux/phoaudiobook dataset from HuggingFace
and save to /home/alida/datasets/phoaudiobook

Note: This is a gated dataset. You need to:
1. Accept terms at: https://huggingface.co/datasets/thivux/phoaudiobook
2. Login to HuggingFace: huggingface-cli login
   OR set token: export HF_TOKEN=your_token_here
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import sys

# Configuration
DATASET_NAME = "thivux/phoaudiobook"
OUTPUT_DIR = Path("/home/alida/datasets/phoaudiobook")
SPLIT = "train"  # Usually "train" for this dataset


def download_dataset():
    """Download the dataset and save to local directory."""
    
    print("=" * 60)
    print("Downloading phoaudiobook Dataset")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directory: {OUTPUT_DIR}")
    print()
    
    # Check for authentication
    from huggingface_hub import whoami
    try:
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
        print()
    except Exception:
        print("⚠️  Not authenticated with HuggingFace!")
        print()
        print("This dataset requires authentication. Please:")
        print("1. Accept terms at: https://huggingface.co/datasets/thivux/phoaudiobook")
        print("2. Login using one of these methods:")
        print()
        print("   Option A: CLI login")
        print("     huggingface-cli login")
        print()
        print("   Option B: Set token")
        print("     export HF_TOKEN=your_token_here")
        print("     # Get token from: https://huggingface.co/settings/tokens")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled. Please authenticate first.")
            sys.exit(0)
        print()
    
    try:
        # Load dataset (this will download if not cached)
        print("📥 Loading dataset from HuggingFace...")
        print("   (This may take a while for large datasets)")
        print()
        
        # Try to get dataset info
        try:
            print("📊 Loading dataset (this may take time)...")
            dataset_info = load_dataset(DATASET_NAME, split=SPLIT)
            total_samples = len(dataset_info)
            print(f"   ✅ Loaded {total_samples:,} samples")
            print()
            
            # Save dataset to disk
            print("💾 Saving dataset to disk...")
            print("   (This may take a while...)")
            dataset_info.save_to_disk(str(OUTPUT_DIR))
            print(f"✅ Dataset saved to: {OUTPUT_DIR}")
            print()
            
            # Show what was saved
            print("📁 Saved files:")
            total_size = 0
            for item in OUTPUT_DIR.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    total_size += size
                    rel_path = item.relative_to(OUTPUT_DIR)
                    print(f"   - {rel_path} ({size / (1024**2):.2f} MB)")
            
            print()
            print(f"📊 Total size: {total_size / (1024**3):.2f} GB")
            
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
                print(f"❌ Authentication required!")
                print()
                print("Please:")
                print("1. Accept dataset terms: https://huggingface.co/datasets/thivux/phoaudiobook")
                print("2. Login: huggingface-cli login")
                print("3. Run this script again")
            else:
                print(f"⚠️  Could not load full dataset: {e}")
                print("   Dataset might be too large for full download.")
                print("   Consider using streaming mode for processing.")
                print()
                print("💡 Tip: Use streaming=True when loading this dataset:")
                print("   dataset = load_dataset('thivux/phoaudiobook', split='train', streaming=True)")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace dataset exists: https://huggingface.co/datasets/thivux/phoaudiobook")
        print("3. Check disk space in /home/alida/datasets/")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print()
    print(f"Dataset location: {OUTPUT_DIR}")
    print()
    print("To use the dataset:")
    print("  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{OUTPUT_DIR}')")
    print()


def check_disk_space():
    """Check available disk space."""
    import shutil
    
    stat = shutil.disk_usage(OUTPUT_DIR.parent)
    free_gb = stat.free / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 10:
        print("⚠️  Warning: Low disk space! Dataset might be large.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)
    print()


if __name__ == "__main__":
    # Check disk space first
    check_disk_space()
    
    # Download dataset
    download_dataset()

