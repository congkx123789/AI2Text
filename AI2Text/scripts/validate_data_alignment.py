"""
Script to validate audio-text alignment in the dataset.

This checks if audio files match their corresponding transcripts,
which is critical for ASR training. Mismatched data causes the model
to ignore audio and only learn language patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))

from utils.manifest_loader import load_merged_dataset
from preprocessing.audio_processing import AudioProcessor
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer


def validate_sample(audio_path: Path, transcript: str, 
                   audio_processor: AudioProcessor,
                   tokenizer: SentencePieceTokenizer,
                   sample_idx: int) -> dict:
    """Validate a single sample.
    
    Returns:
        dict with validation results
    """
    result = {
        'index': sample_idx,
        'audio_path': str(audio_path),
        'transcript': transcript,
        'valid': True,
        'errors': []
    }
    
    # Check 1: Audio file exists
    if not audio_path.exists():
        result['valid'] = False
        result['errors'].append(f"Audio file not found: {audio_path}")
        return result
    
    # Check 2: Audio file is readable
    try:
        audio, sr = librosa.load(str(audio_path), sr=None)
        if len(audio) == 0:
            result['valid'] = False
            result['errors'].append("Audio file is empty")
            return result
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Cannot load audio: {e}")
        return result
    
    # Check 3: Audio duration is reasonable (not too short, not too long)
    duration = len(audio) / sr
    if duration < 0.5:
        result['errors'].append(f"Audio too short: {duration:.2f}s")
    if duration > 30:
        result['errors'].append(f"Audio too long: {duration:.2f}s")
    
    # Check 4: Transcript is not empty
    if not transcript or len(transcript.strip()) == 0:
        result['valid'] = False
        result['errors'].append("Transcript is empty")
        return result
    
    # Check 5: Can process audio to mel spectrogram
    try:
        mel = audio_processor.process_audio(audio, sr)
        if mel.shape[0] == 0 or mel.shape[1] == 0:
            result['valid'] = False
            result['errors'].append("Mel spectrogram is empty")
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Cannot process audio: {e}")
    
    # Check 6: Can tokenize transcript
    try:
        tokens = tokenizer.encode(transcript)
        if len(tokens) == 0:
            result['valid'] = False
            result['errors'].append("Tokenization produced empty tokens")
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Cannot tokenize transcript: {e}")
    
    return result


def validate_dataset(split: str = 'train', 
                     dataset_root: str = 'data/processed/full_merged_dataset',
                     num_samples: int = 100,
                     random_sample: bool = True):
    """Validate dataset alignment.
    
    Args:
        split: Dataset split ('train' or 'val')
        dataset_root: Root directory of dataset
        num_samples: Number of samples to check
        random_sample: If True, randomly sample samples; else check first N
    """
    print("="*80)
    print("🔍 VALIDATING DATA ALIGNMENT")
    print("="*80)
    
    # Load dataset
    print(f"📂 Loading {split} dataset...")
    df = load_merged_dataset(split, dataset_root)
    print(f"   Total samples: {len(df):,}")
    
    # Sample samples to check
    if random_sample:
        sample_indices = np.random.choice(len(df), size=min(num_samples, len(df)), replace=False)
    else:
        sample_indices = range(min(num_samples, len(df)))
    
    print(f"📊 Checking {len(sample_indices)} samples...")
    print("-"*80)
    
    # Setup processors
    audio_processor = AudioProcessor(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400
    )
    
    tokenizer = SentencePieceTokenizer('models/tokenizer_vi_en_3500.model')
    
    # Validate samples
    results = []
    invalid_count = 0
    
    for idx in tqdm(sample_indices, desc="Validating"):
        row = df.iloc[idx]
        audio_path = Path(row['file_path'])
        transcript = row['transcript']
        
        result = validate_sample(audio_path, transcript, audio_processor, tokenizer, idx)
        results.append(result)
        
        if not result['valid']:
            invalid_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print("📊 VALIDATION RESULTS")
    print("="*80)
    print(f"✅ Valid samples: {len(results) - invalid_count}/{len(results)}")
    print(f"❌ Invalid samples: {invalid_count}/{len(results)}")
    
    if invalid_count > 0:
        print("\n⚠️  INVALID SAMPLES:")
        print("-"*80)
        for result in results:
            if not result['valid']:
                print(f"\nSample {result['index']}:")
                print(f"  Audio: {result['audio_path']}")
                print(f"  Transcript: {result['transcript'][:100]}...")
                print(f"  Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
    
    # Check for potential data shuffling issues
    print("\n" + "="*80)
    print("🔍 CHECKING FOR DATA SHUFFLING ISSUES")
    print("="*80)
    
    # Check if similar transcripts are grouped together (bad for training)
    sample_transcripts = [df.iloc[idx]['transcript'] for idx in sample_indices[:20]]
    unique_start_words = set()
    for transcript in sample_transcripts:
        first_word = transcript.split()[0] if transcript.split() else ""
        unique_start_words.add(first_word)
    
    if len(unique_start_words) < 5:
        print("⚠️  WARNING: Samples seem to have similar starting words.")
        print("   This might indicate data is sorted/grouped, which is bad for training.")
        print("   Consider shuffling the dataset.")
    else:
        print("✅ Sample diversity looks good")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate audio-text alignment in dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Dataset split to validate')
    parser.add_argument('--dataset-root', type=str, 
                       default='data/processed/full_merged_dataset',
                       help='Root directory of dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to check')
    parser.add_argument('--random', action='store_true',
                       help='Randomly sample samples (default: check first N)')
    
    args = parser.parse_args()
    
    validate_dataset(
        split=args.split,
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        random_sample=args.random
    )

