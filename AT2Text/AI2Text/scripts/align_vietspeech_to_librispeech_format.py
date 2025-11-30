#!/usr/bin/env python3
"""
Align VietSpeech manifest structure to match librispeech_alignments format.
Adds missing columns (sex, subset) and creates backup files.
Does NOT change <vi> or <en> language tags.
"""

import pandas as pd
import shutil
from pathlib import Path
import argparse


def align_manifest(manifest_path: Path, split_name: str = None):
    """Add sex and subset columns to VietSpeech manifest to match librispeech format."""
    print(f"Processing {manifest_path}...")
    
    # Read the manifest
    df = pd.read_csv(manifest_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Current columns: {list(df.columns)}")
    
    # Create backup if it doesn't exist
    backup_path = manifest_path.with_suffix('.csv.backup')
    if not backup_path.exists():
        print(f"  Creating backup: {backup_path}")
        shutil.copy2(manifest_path, backup_path)
    else:
        print(f"  Backup already exists: {backup_path}")
    
    # Add missing columns if they don't exist
    if 'sex' not in df.columns:
        # Use 'U' for unknown sex (VietSpeech doesn't have speaker metadata)
        df['sex'] = 'U'
        print("  Added 'sex' column with value 'U' (unknown)")
    
    if 'subset' not in df.columns:
        # Use 'vietspeech' as subset identifier
        df['subset'] = 'vietspeech'
        print("  Added 'subset' column with value 'vietspeech'")
    
    # Ensure column order matches librispeech_alignments: id,transcript,audio_path,words_json,sex,subset
    expected_columns = ['id', 'transcript', 'audio_path', 'words_json', 'sex', 'subset']
    # Only reorder if all expected columns exist
    if all(col in df.columns for col in expected_columns):
        df = df[expected_columns]
        print(f"  Reordered columns to: {list(df.columns)}")
    
    # Save the updated manifest
    df.to_csv(manifest_path, index=False)
    print(f"  ✅ Updated manifest saved: {manifest_path}")
    
    return df


def create_readme(vietspeech_dir: Path):
    """Create README.md for VietSpeech directory similar to librispeech_alignments."""
    readme_content = """# VietSpeech (processed)

Vietnamese VietSpeech audio with word-level timestamps lives here after being consolidated into the canonical `train` / `val` / `test` splits. All files come from the VietSpeech dataset.

## Folder layout

```
VietSpeech/
├── train/
│   ├── audio/                    # WAV files for this split
│   ├── manifest.csv              # Rich metadata (id, transcript, audio_path, words_json, sex, subset)
│   ├── manifest.csv.backup       # Backup of original manifest
│   └── timestamps.json           # Word alignments filtered to this split
├── val/
│   ├── audio/
│   ├── manifest.csv
│   ├── manifest.csv.backup
│   └── timestamps.json
└── test/
    ├── audio/
    ├── manifest.csv
    ├── manifest.csv.backup
    └── timestamps.json
```

Key file details:

- `manifest.csv` — one row per utterance with columns `id`, `transcript`, `audio_path` (relative to the split directory), `words_json`, `sex`, and `subset`.
- `timestamps.json` — dictionary of `{filename: {duration, text, segments}}` objects, scoped to the current split for quick lookup.
- `manifest.csv.backup` — backup of the original manifest before adding metadata columns.

## Notes

- Audio is WAV format with word timestamps in seconds.
- Transcripts use `<|vi|>` language tags for Vietnamese content.
- The `sex` column is set to 'U' (unknown) as VietSpeech doesn't provide speaker gender metadata.
- The `subset` column is set to 'vietspeech' to identify the dataset source.
- Keep the folder structure intact; code under `training/` assumes `audio/` lives next to the manifest inside each split directory.

"""
    
    readme_path = vietspeech_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"✅ Created README.md: {readme_path}")
    else:
        print(f"⚠️  README.md already exists: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Align VietSpeech manifest structure to match librispeech_alignments format"
    )
    parser.add_argument(
        '--vietspeech-dir',
        type=str,
        default='data/processed/VietSpeech',
        help='Path to VietSpeech processed directory'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to process (default: train val test)'
    )
    
    args = parser.parse_args()
    
    vietspeech_dir = Path(args.vietspeech_dir)
    if not vietspeech_dir.exists():
        raise SystemExit(f"Error: VietSpeech directory not found: {vietspeech_dir}")
    
    print(f"Aligning VietSpeech manifests in: {vietspeech_dir}")
    print(f"Processing splits: {args.splits}\n")
    
    # Process each split
    for split in args.splits:
        split_dir = vietspeech_dir / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}, skipping...")
            continue
        
        manifest_path = split_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"⚠️  Manifest not found: {manifest_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        align_manifest(manifest_path, split)
    
    # Create README.md
    print(f"\n{'='*60}")
    print("Creating README.md")
    print(f"{'='*60}")
    create_readme(vietspeech_dir)
    
    print(f"\n✅ Done! VietSpeech structure now matches librispeech_alignments format.")


if __name__ == "__main__":
    main()

