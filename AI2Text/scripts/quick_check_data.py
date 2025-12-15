"""
Quick script to check if audio-text pairs are correctly aligned.
Run this BEFORE training to catch data loading bugs.
"""

import sys
from pathlib import Path
import pandas as pd
import librosa
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))

from utils.manifest_loader import load_merged_dataset


def check_sample(idx: int, df: pd.DataFrame, play_audio: bool = False):
    """Check a single sample."""
    row = df.iloc[idx]
    audio_path = Path(row['file_path'])
    transcript = row['transcript']
    language = row.get('language', 'unknown')
    
    print(f"\n{'='*80}")
    print(f"Sample {idx}")
    print(f"{'='*80}")
    print(f"Audio: {audio_path}")
    print(f"Language: {language}")
    print(f"Transcript: {transcript}")
    
    # Check if file exists
    if not audio_path.exists():
        print(f"❌ ERROR: Audio file not found!")
        return False
    
    # Load and check audio
    try:
        audio, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio) / sr
        print(f"Duration: {duration:.2f}s")
        
        if duration < 0.5:
            print(f"⚠️  WARNING: Audio too short (< 0.5s)")
        if duration > 30:
            print(f"⚠️  WARNING: Audio too long (> 30s)")
        
        if play_audio:
            print(f"🔊 Playing audio...")
            import sounddevice as sd
            sd.play(audio, sr)
            sd.wait()
        
        print(f"✅ Audio file is valid")
        return True
    except Exception as e:
        print(f"❌ ERROR loading audio: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick check data alignment')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2, 10, 100],
                       help='Sample indices to check')
    parser.add_argument('--play', action='store_true',
                       help='Play audio files (requires sounddevice)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 QUICK DATA ALIGNMENT CHECK")
    print("="*80)
    
    # Load dataset
    df = load_merged_dataset(args.split, 'data/processed/full_merged_dataset')
    print(f"Dataset: {args.split}")
    print(f"Total samples: {len(df):,}")
    print(f"Checking indices: {args.indices}")
    
    # Check samples
    all_valid = True
    for idx in args.indices:
        if idx >= len(df):
            print(f"\n⚠️  Index {idx} out of range (max: {len(df)-1})")
            continue
        
        valid = check_sample(idx, df, play_audio=args.play)
        if not valid:
            all_valid = False
    
    print("\n" + "="*80)
    if all_valid:
        print("✅ All checked samples are valid!")
    else:
        print("❌ Some samples have errors - FIX BEFORE TRAINING!")
    print("="*80)

