"""
Download sample audio data for testing and demonstration.

This script helps you download sample Vietnamese audio datasets
or create synthetic data for testing the ASR system.
"""

import argparse
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def download_vietnamese_sample_data(output_dir: str = "data/external"):
    """
    Download sample Vietnamese audio data (if available from public sources).
    
    This is a placeholder for downloading sample datasets. In practice,
    you would download from:
    - Common Voice Vietnamese
    - OpenSLR Vietnamese datasets
    - Vivos dataset
    
    Args:
        output_dir (str): Directory to save downloaded data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Vietnamese ASR Sample Data Download")
    print("=" * 70)
    print("\nSample data sources you can download:")
    print("\n1. Common Voice Vietnamese:")
    print("   https://commonvoice.mozilla.org/en/datasets")
    print("   - Free, open-source dataset")
    print("   - Multiple speakers, varied audio quality")
    print("   - Download via: https://commonvoice.mozilla.org/en/datasets")
    
    print("\n2. OpenSLR Vietnamese:")
    print("   https://www.openslr.org/")
    print("   - SLR47: 100 hours of Vietnamese speech")
    print("   - Download via wget or direct links")
    
    print("\n3. Vivos Dataset:")
    print("   https://ailab.hcmus.edu.vn/vivos")
    print("   - Vietnamese speech dataset")
    print("   - Requires registration")
    
    print("\n" + "=" * 70)
    print("Manual Download Instructions:")
    print("=" * 70)
    print(f"\n1. Download audio files to: {output_path.absolute()}")
    print("2. Create a CSV file with columns: file_path, transcript")
    print("3. Use prepare_data.py to import:")
    print(f"   python scripts/prepare_data.py --csv your_data.csv --audio_base {output_dir} --auto_split")
    
    # Create example CSV template
    example_csv = output_path / "sample_data_template.csv"
    csv_content = """file_path,transcript,split
sample1.wav,xin chào việt nam,train
sample2.wav,tôi là sinh viên,train
sample3.wav,hôm nay trời đẹp,val
sample4.wav,tiếng việt rất hay,test
"""
    
    with open(example_csv, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    print(f"\n[OK] Created example CSV template at: {example_csv}")
    print("\nReplace the file paths and transcripts with your actual data!")


def create_test_data_structure(base_dir: str = "data"):
    """
    Create a test data directory structure for development.
    
    Args:
        base_dir (str): Base directory for data
    """
    base_path = Path(base_dir)
    
    # Create directories
    directories = [
        base_path / "raw",
        base_path / "processed",
        base_path / "external",
        base_path / "raw" / "train",
        base_path / "raw" / "val",
        base_path / "raw" / "test",
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to track empty directories
        (dir_path / ".gitkeep").touch(exist_ok=True)
    
    # Create README in data directory
    readme_content = """# Data Directory

This directory contains audio data for ASR training.

## Structure

- `raw/`: Raw audio files (WAV, MP3, FLAC, OGG)
  - `train/`: Training audio files
  - `val/`: Validation audio files
  - `test/`: Test audio files

- `processed/`: Preprocessed audio files (mel spectrograms, features)

- `external/`: External datasets downloaded from public sources

## Data Format

Audio files should be:
- Format: WAV (preferred), MP3, FLAC, or OGG
- Sample rate: 16kHz or higher (16kHz recommended)
- Channels: Mono (1 channel) preferred
- Duration: 0.5s to 30s (recommended)

## Import Data

1. Create CSV file with columns: `file_path`, `transcript`, `split` (optional)
2. Run: `python scripts/prepare_data.py --csv your_data.csv --audio_base data/raw --auto_split`

See `DATA_PREPARATION_GUIDE.md` for detailed instructions.
"""
    
    with open(base_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n[OK] Created data directory structure at: {base_path.absolute()}")
    print(f"[OK] Created README.md in data directory")


def main():
    parser = argparse.ArgumentParser(
        description='Download sample data or create data directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show sample data download information
  python scripts/download_sample_data.py

  # Create data directory structure
  python scripts/download_sample_data.py --create_structure

  # Custom output directory
  python scripts/download_sample_data.py --output data/my_data
        """
    )
    parser.add_argument('--output', type=str, default='data/external',
                       help='Output directory for downloaded data (default: data/external)')
    parser.add_argument('--create_structure', action='store_true',
                       help='Create data directory structure for testing')
    args = parser.parse_args()
    
    if args.create_structure:
        create_test_data_structure()
    else:
        download_vietnamese_sample_data(args.output)


if __name__ == '__main__':
    main()

