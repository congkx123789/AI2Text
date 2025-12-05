"""
Utility to load data from manifest.csv files.
Used for file-based datasets like merged_dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import json


def load_manifest_data(manifest_path: str, base_audio_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from a manifest.csv file.
    
    Args:
        manifest_path: Path to manifest.csv file
        base_audio_dir: Base directory for audio files. If None, uses manifest's parent/audio
    
    Returns:
        DataFrame with columns: file_path, transcript, duration_seconds, words_json, language
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    # Determine base audio directory
    if base_audio_dir is None:
        # Default: audio files are in manifest's parent directory / audio/
        # But audio_path in manifest is like "audio/filename.wav", so base is manifest's parent
        base_audio_dir = manifest_path.parent
    else:
        base_audio_dir = Path(base_audio_dir)
    
    # Build full file paths
    if 'audio_path' in df.columns:
        # audio_path is relative to base_audio_dir (e.g., "audio/007_000000980.wav")
        def build_path(audio_path: str) -> str:
            if pd.isna(audio_path):
                return None
            audio_path_str = str(audio_path)
            if Path(audio_path_str).is_absolute():
                return audio_path_str
            # audio_path is like "audio/filename.wav", so join with base_audio_dir
            return str(base_audio_dir / audio_path_str)
        
        df['file_path'] = df['audio_path'].apply(build_path)
    elif 'file_path' not in df.columns:
        # Try to construct from id
        if 'id' in df.columns:
            df['file_path'] = df['id'].apply(lambda x: str(base_audio_dir / f"{x}.wav"))
        else:
            raise ValueError("Manifest must have 'audio_path' or 'id' column")
    
    # Extract language from transcript if it has language tags
    if 'transcript' in df.columns:
        def extract_language(transcript: str) -> str:
            if pd.isna(transcript):
                return 'vi'  # Default
            transcript_str = str(transcript)
            if transcript_str.startswith('<|vi|>'):
                return 'vi'
            elif transcript_str.startswith('<|en|>'):
                return 'en'
            else:
                return 'vi'  # Default to Vietnamese
        
        df['language'] = df['transcript'].apply(extract_language)
        
        # Clean transcript (remove language tags)
        def clean_transcript(transcript: str) -> str:
            if pd.isna(transcript):
                return ''
            transcript_str = str(transcript)
            # Remove language tags
            transcript_str = transcript_str.replace('<|vi|>', '').replace('<|en|>', '').strip()
            return transcript_str
        
        df['transcript'] = df['transcript'].apply(clean_transcript)
    
    # Rename duration to duration_seconds if needed
    if 'duration' in df.columns and 'duration_seconds' not in df.columns:
        df['duration_seconds'] = df['duration']
    
    # Ensure words_json is available
    if 'words_json' in df.columns:
        # Keep words_json as is (it's already JSON string)
        pass
    else:
        df['words_json'] = None
    
    # Select and rename columns to match expected format
    result_df = pd.DataFrame({
        'file_path': df['file_path'],
        'transcript': df['transcript'],
        'duration_seconds': df.get('duration_seconds', None),
        'words_json': df.get('words_json', None),
        'language': df.get('language', 'vi')
    })
    
    # Filter out rows with missing file paths or transcripts
    result_df = result_df.dropna(subset=['file_path', 'transcript'])
    
    # Verify files exist
    def file_exists(file_path: str) -> bool:
        return Path(file_path).exists()
    
    existing_mask = result_df['file_path'].apply(file_exists)
    missing_count = (~existing_mask).sum()
    
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count} audio files not found. Filtering them out.")
        result_df = result_df[existing_mask].reset_index(drop=True)
    
    print(f"✅ Loaded {len(result_df)} samples from {manifest_path}")
    
    return result_df


def load_merged_dataset(split: str = 'train', 
                       dataset_root: str = 'data/processed/merged_dataset',
                       language: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from merged_dataset directory.
    
    Args:
        split: Split to load ('train', 'val', 'test')
        dataset_root: Root directory of merged_dataset
        language: Optional language filter ('en' or 'vi')
    
    Returns:
        DataFrame with training data
    """
    dataset_root = Path(dataset_root)
    base_dir = dataset_root / split
    manifest_path = base_dir / "manifest.csv"
    sorted_manifest_path = base_dir / "manifest_sorted.csv"
    sliced_manifest_path = base_dir / "manifest_sliced.csv"
    
    # Prefer sliced manifest if it exists (short segments for small models)
    if sliced_manifest_path.exists():
        print(f"✂️ Using sliced manifest (short segments): {sliced_manifest_path}")
        df = load_manifest_data(sliced_manifest_path)
    elif sorted_manifest_path.exists():
        print(f"📑 Using sorted manifest: {sorted_manifest_path}")
        df = load_manifest_data(sorted_manifest_path)
    else:
        df = load_manifest_data(manifest_path)
    
    # Filter by language if specified
    if language:
        df = df[df['language'] == language].reset_index(drop=True)
        print(f"   Filtered to {language}: {len(df)} samples")
    
    return df

