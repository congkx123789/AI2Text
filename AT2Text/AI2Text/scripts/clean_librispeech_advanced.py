#!/usr/bin/env python3
"""
Advanced 3-step cleaning strategy for LibriSpeech alignments dataset.
Step 1: Text & Alignment Heuristics (Sanity Check)
Step 2: Audio Signal Filtering
Step 3: Generate Clean Manifest
"""

import pandas as pd
import soundfile as sf
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# CONFIGURATION
DEFAULT_BASE_DIR = "data/raw/librispeech_alignments"
MIN_DURATION = 1.0  # Seconds
MAX_DURATION = 20.0  # Seconds
MIN_TEXT_LEN = 2     # Characters
MAX_CHARS_PER_SEC = 25  # Threshold for "too fast" (likely bad alignment)
MIN_CHARS_PER_SEC = 2   # Threshold for "too slow" (too much silence)

# Audio signal filtering thresholds
CLIPPING_THRESHOLD = 0.99  # If >99% of samples are at max amplitude, it's clipped
MIN_SILENCE_RATIO = 0.3    # If >30% of audio is silence, reject
SILENCE_THRESHOLD = 0.01   # Amplitude threshold for silence detection


def detect_clipping(audio_data):
    """
    Detect if audio is clipped (distorted).
    
    Args:
        audio_data: Audio samples (numpy array)
        
    Returns:
        (is_clipped, clipping_ratio): Boolean and ratio of clipped samples
    """
    if len(audio_data) == 0:
        return True, 1.0
    
    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(audio_data))
    if max_abs == 0:
        return False, 0.0
    
    normalized = audio_data / max_abs
    clipped_samples = np.sum(np.abs(normalized) >= CLIPPING_THRESHOLD)
    clipping_ratio = clipped_samples / len(normalized)
    
    # If more than 5% of samples are at max amplitude, consider it clipped
    is_clipped = clipping_ratio > 0.05
    return is_clipped, clipping_ratio


def estimate_silence_ratio(audio_data, sample_rate):
    """
    Estimate the ratio of silence in the audio.
    
    Args:
        audio_data: Audio samples (numpy array)
        sample_rate: Sample rate in Hz
        
    Returns:
        silence_ratio: Ratio of audio that is silence (0.0 to 1.0)
    """
    if len(audio_data) == 0:
        return 1.0
    
    # Calculate RMS in windows
    window_size = int(0.1 * sample_rate)  # 100ms windows
    if window_size == 0:
        window_size = 1
    
    num_windows = len(audio_data) // window_size
    if num_windows == 0:
        return 1.0
    
    silence_windows = 0
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio_data[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        
        if rms < SILENCE_THRESHOLD:
            silence_windows += 1
    
    silence_ratio = silence_windows / num_windows
    return silence_ratio


def trim_silence_from_alignments(words, phonemes, duration):
    """
    Trim silence at start and end based on alignments.
    
    Args:
        words: List of word timestamps
        phonemes: List of phoneme timestamps
        duration: Total audio duration
        
    Returns:
        (start_trim, end_trim): Trim times in seconds
    """
    start_trim = 0.0
    end_trim = duration
    
    # Find first non-silence phoneme
    if phonemes:
        first_phoneme = phonemes[0]
        start_trim = max(0.0, first_phoneme.get('start', 0.0) - 0.1)  # 100ms buffer
    
    # Find last non-silence phoneme
    if phonemes:
        last_phoneme = phonemes[-1]
        end_trim = min(duration, last_phoneme.get('end', duration) + 0.1)  # 100ms buffer
    
    return start_trim, end_trim


def is_valid_sample(row, split_dir, check_audio_signal=True,
                    min_duration=MIN_DURATION, max_duration=MAX_DURATION,
                    min_text_len=MIN_TEXT_LEN, max_cps=MAX_CHARS_PER_SEC,
                    min_cps=MIN_CHARS_PER_SEC):
    """
    Step 1 & 2: Validate sample using text/alignment heuristics and audio signal.
    
    Returns:
        (is_valid, reason, metadata): Tuple of validation result, reason, and metadata
    """
    metadata = {}
    
    # ========== STEP 1: Text & Alignment Heuristics ==========
    
    # 1. Check Text
    text = row['transcript']
    if not isinstance(text, str) or len(text.strip()) < min_text_len:
        return False, "text_too_short", metadata
    
    # Normalize text (remove extra whitespace, lowercase)
    text_clean = ' '.join(text.strip().split())
    metadata['text_length'] = len(text_clean)
    
    # 2. Check Audio File existence and duration
    audio_path = split_dir / row['audio_path']
    if not audio_path.exists():
        return False, "file_missing", metadata
    
    try:
        # soundfile.info is faster than reading the whole file
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        channels = info.channels
    except Exception as e:
        return False, f"audio_corrupt: {str(e)}", metadata
    
    metadata['duration'] = duration
    metadata['sample_rate'] = sample_rate
    metadata['channels'] = channels
    
    # Check duration
    if duration < min_duration:
        return False, "audio_too_short", metadata
    if duration > max_duration:
        return False, "audio_too_long", metadata
    
    # Check sample rate (should be 16kHz)
    if sample_rate < 16000:
        return False, "low_sample_rate", metadata
    
    # Check channels (should be mono)
    if channels > 1:
        return False, "not_mono", metadata
    
    # 3. Check Alignment Logic (Characters Per Second - CPS)
    cps = len(text_clean) / duration
    metadata['chars_per_sec'] = cps
    
    if cps > max_cps:
        return False, "speaking_too_fast_alignment_error", metadata
    if cps < min_cps:
        return False, "speaking_too_slow_excess_silence", metadata
    
    # 4. JSON Validation
    try:
        words = json.loads(row['words_json']) if isinstance(row['words_json'], str) else row['words_json']
        phonemes = json.loads(row['phonemes_json']) if isinstance(row['phonemes_json'], str) else row['phonemes_json']
        
        if not words or len(words) == 0:
            return False, "empty_alignment", metadata
        
        metadata['num_words'] = len(words)
        metadata['num_phonemes'] = len(phonemes) if phonemes else 0
        
        # Check alignment consistency
        if words:
            first_word_start = words[0].get('start', 0)
            last_word_end = words[-1].get('end', duration)
            
            # Alignment should be within audio duration
            if first_word_start < 0 or last_word_end > duration + 0.5:
                return False, "alignment_out_of_bounds", metadata
            
            # Calculate trim times
            start_trim, end_trim = trim_silence_from_alignments(words, phonemes, duration)
            metadata['start_trim'] = start_trim
            metadata['end_trim'] = end_trim
            metadata['effective_duration'] = end_trim - start_trim
            
    except Exception as e:
        return False, f"corrupt_json: {str(e)}", metadata
    
    # ========== STEP 2: Audio Signal Filtering ==========
    
    if check_audio_signal:
        try:
            # Read audio data
            audio_data, sr = sf.read(audio_path)
            
            # Handle stereo by converting to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Check clipping
            is_clipped, clipping_ratio = detect_clipping(audio_data)
            metadata['clipping_ratio'] = clipping_ratio
            
            if is_clipped:
                return False, "audio_clipped", metadata
            
            # Check silence ratio
            silence_ratio = estimate_silence_ratio(audio_data, sample_rate)
            metadata['silence_ratio'] = silence_ratio
            
            if silence_ratio > MIN_SILENCE_RATIO:
                return False, "too_much_silence", metadata
            
            # Check if audio is too quiet (RMS too low)
            rms = np.sqrt(np.mean(audio_data ** 2))
            metadata['rms'] = rms
            
            if rms < 0.001:  # Very quiet audio
                return False, "audio_too_quiet", metadata
            
        except Exception as e:
            return False, f"audio_read_error: {str(e)}", metadata
    
    # All checks passed
    return True, "OK", metadata


def process_dataset(base_dir, output_file="clean_train_manifest.csv", 
                   check_audio_signal=True,
                   min_duration=MIN_DURATION, max_duration=MAX_DURATION,
                   min_text_len=MIN_TEXT_LEN, max_cps=MAX_CHARS_PER_SEC,
                   min_cps=MIN_CHARS_PER_SEC):
    """
    Process all splits and generate clean manifest.
    
    Args:
        base_dir: Base directory containing splits
        output_file: Output CSV file path
        check_audio_signal: Whether to perform audio signal analysis
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    clean_rows = []
    rejected_stats = {}
    
    # Get all split directories
    splits = sorted([d for d in base_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("split_")])
    
    print(f"Scanning {len(splits)} splits...")
    print(f"Audio signal checking: {'Enabled' if check_audio_signal else 'Disabled'}")
    print()
    
    for split_dir in tqdm(splits, desc="Processing splits"):
        csv_path = split_dir / "train.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), 
                          desc=f"  {split_dir.name}", leave=False):
            is_valid, reason, metadata = is_valid_sample(
                row, split_dir, check_audio_signal=check_audio_signal,
                min_duration=min_duration, max_duration=max_duration,
                min_text_len=min_text_len, max_cps=max_cps, min_cps=min_cps
            )
            
            if is_valid:
                # Add absolute path and metadata
                row_dict = row.to_dict()
                row_dict['abs_audio_path'] = str(split_dir / row['audio_path'])
                row_dict['split_name'] = split_dir.name
                
                # Add metadata
                for key, value in metadata.items():
                    row_dict[f'meta_{key}'] = value
                
                clean_rows.append(row_dict)
            else:
                rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
    
    # SAVE RESULTS
    if clean_rows:
        clean_df = pd.DataFrame(clean_rows)
        clean_df.to_csv(output_file, index=False)
        print(f"\n✅ Clean manifest saved to: {output_file}")
        print(f"   Total accepted samples: {len(clean_df):,}")
    else:
        print("\n❌ No valid samples found!")
        return
    
    # Print report
    print("\n" + "="*60)
    print("CLEANING REPORT")
    print("="*60)
    print(f"Total Accepted: {len(clean_df):,}")
    print(f"Total Rejected: {sum(rejected_stats.values()):,}")
    print(f"Acceptance Rate: {100*len(clean_df)/(len(clean_df)+sum(rejected_stats.values())):.1f}%")
    print("\nRejection Reasons:")
    for reason, count in sorted(rejected_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {reason}: {count:,}")
    
    # Print statistics
    if len(clean_df) > 0:
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        if 'meta_duration' in clean_df.columns:
            print(f"Duration: {clean_df['meta_duration'].min():.2f}s - {clean_df['meta_duration'].max():.2f}s")
            print(f"         Mean: {clean_df['meta_duration'].mean():.2f}s")
        if 'meta_chars_per_sec' in clean_df.columns:
            print(f"CPS: {clean_df['meta_chars_per_sec'].min():.1f} - {clean_df['meta_chars_per_sec'].max():.1f}")
            print(f"     Mean: {clean_df['meta_chars_per_sec'].mean():.1f}")
        if 'meta_silence_ratio' in clean_df.columns:
            print(f"Silence Ratio: {clean_df['meta_silence_ratio'].min():.3f} - {clean_df['meta_silence_ratio'].max():.3f}")
            print(f"              Mean: {clean_df['meta_silence_ratio'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced 3-step cleaning for LibriSpeech alignments dataset'
    )
    parser.add_argument(
        '--input', type=str, default=DEFAULT_BASE_DIR,
        help='Input directory containing splits'
    )
    parser.add_argument(
        '--output', type=str, default='clean_train_manifest.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--min-duration', type=float, default=MIN_DURATION,
        help='Minimum audio duration in seconds'
    )
    parser.add_argument(
        '--max-duration', type=float, default=MAX_DURATION,
        help='Maximum audio duration in seconds'
    )
    parser.add_argument(
        '--min-text-len', type=int, default=MIN_TEXT_LEN,
        help='Minimum transcript length in characters'
    )
    parser.add_argument(
        '--max-cps', type=float, default=MAX_CHARS_PER_SEC,
        help='Maximum characters per second (alignment check)'
    )
    parser.add_argument(
        '--min-cps', type=float, default=MIN_CHARS_PER_SEC,
        help='Minimum characters per second (silence check)'
    )
    parser.add_argument(
        '--skip-audio-signal', action='store_true',
        help='Skip audio signal analysis (faster but less thorough)'
    )
    
    args = parser.parse_args()
    
    process_dataset(
        args.input,
        args.output,
        check_audio_signal=not args.skip_audio_signal,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_len=args.min_text_len,
        max_cps=args.max_cps,
        min_cps=args.min_cps
    )


if __name__ == '__main__':
    main()

