#!/usr/bin/env python3
"""
Data cleaning script for LibriSpeech alignments dataset.
Cleans transcripts, validates audio files, and removes invalid entries.
"""

import sys
import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import soundfile as sf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.text_cleaning import VietnameseTextNormalizer, BilingualTextNormalizer


class LibriSpeechDataCleaner:
    """Cleaner for LibriSpeech alignments dataset."""
    
    def __init__(self, 
                 min_audio_duration: float = 0.5,
                 max_audio_duration: float = 30.0,
                 min_transcript_length: int = 3,
                 max_transcript_length: int = 500,
                 clean_text: bool = True,
                 language: str = 'en'):
        """Initialize data cleaner.
        
        Args:
            min_audio_duration: Minimum audio duration in seconds
            max_audio_duration: Maximum audio duration in seconds
            min_transcript_length: Minimum transcript length in characters
            max_transcript_length: Maximum transcript length in characters
            clean_text: Whether to clean/normalize text
            language: Language for text cleaning ('en' or 'vi')
        """
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.min_transcript_length = min_transcript_length
        self.max_transcript_length = max_transcript_length
        self.clean_text = clean_text
        self.language = language
        
        # Initialize text normalizer
        if clean_text:
            if language == 'vi':
                self.normalizer = VietnameseTextNormalizer(
                    lowercase=True,
                    remove_punctuation=True,
                    normalize_unicode=True
                )
            else:
                # For English, use bilingual normalizer
                self.normalizer = BilingualTextNormalizer(
                    lowercase=True,
                    remove_punctuation=True
                )
        else:
            self.normalizer = None
        
        # Statistics
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid_audio': 0,
            'invalid_duration': 0,
            'invalid_transcript': 0,
            'duplicates': 0,
            'cleaned': 0
        }
    
    def validate_audio(self, audio_path: str) -> Tuple[bool, Dict]:
        """Validate audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            (is_valid, info): Tuple of validation result and audio info
        """
        info = {}
        
        try:
            if not os.path.exists(audio_path):
                return False, {'error': 'File not found'}
            
            # Get audio info
            audio_info = sf.info(audio_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            channels = audio_info.channels
            
            info = {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'format': audio_info.format
            }
            
            # Check duration
            if duration < self.min_audio_duration:
                return False, {**info, 'error': f'Too short: {duration:.2f}s'}
            
            if duration > self.max_audio_duration:
                return False, {**info, 'error': f'Too long: {duration:.2f}s'}
            
            # Check sample rate (should be 16kHz)
            if sample_rate < 16000:
                return False, {**info, 'error': f'Low sample rate: {sample_rate}Hz'}
            
            # Check channels (should be mono)
            if channels > 1:
                return False, {**info, 'error': f'Not mono: {channels} channels'}
            
            return True, info
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def clean_transcript(self, transcript: str) -> str:
        """Clean transcript text.
        
        Args:
            transcript: Raw transcript
            
        Returns:
            cleaned_transcript: Cleaned transcript
        """
        if not self.clean_text or not self.normalizer:
            return transcript.strip()
        
        # Clean using normalizer
        if self.language == 'vi':
            cleaned = self.normalizer.clean_transcript(transcript)
        else:
            cleaned = self.normalizer.normalize(transcript, lang='en')
        
        return cleaned
    
    def validate_transcript(self, transcript: str) -> Tuple[bool, str]:
        """Validate transcript.
        
        Args:
            transcript: Transcript text
            
        Returns:
            (is_valid, cleaned_transcript): Validation result and cleaned text
        """
        if not transcript:
            return False, ""
        
        # Clean transcript
        cleaned = self.clean_transcript(transcript)
        
        # Check length
        if len(cleaned) < self.min_transcript_length:
            return False, cleaned
        
        if len(cleaned) > self.max_transcript_length:
            return False, cleaned
        
        # Check if transcript is empty after cleaning
        if not cleaned.strip():
            return False, cleaned
        
        return True, cleaned
    
    def clean_split(self, split_dir: Path, output_dir: Path = None) -> Dict:
        """Clean a single split.
        
        Args:
            split_dir: Directory containing split data
            output_dir: Output directory (if None, overwrites original)
            
        Returns:
            stats: Cleaning statistics
        """
        if output_dir is None:
            output_dir = split_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'audio').mkdir(parents=True, exist_ok=True)
        
        csv_path = split_dir / 'train.csv'
        timestamps_path = split_dir / 'timestamps.json'
        audio_dir = split_dir / 'audio'
        
        # Load timestamps
        timestamps_data = []
        if timestamps_path.exists():
            with open(timestamps_path, 'r', encoding='utf-8') as f:
                timestamps_data = json.load(f)
        
        # Create timestamps lookup
        timestamps_dict = {entry['id']: entry for entry in timestamps_data}
        
        # Process CSV
        valid_rows = []
        seen_ids = set()
        duplicate_ids = set()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in tqdm(reader, desc=f"Cleaning {split_dir.name}"):
                self.stats['total'] += 1
                
                # Check for duplicates
                if row['id'] in seen_ids:
                    duplicate_ids.add(row['id'])
                    self.stats['duplicates'] += 1
                    continue
                seen_ids.add(row['id'])
                
                # Validate audio
                audio_path = split_dir / row['audio_path']
                audio_valid, audio_info = self.validate_audio(str(audio_path))
                
                if not audio_valid:
                    self.stats['invalid_audio'] += 1
                    if 'duration' in audio_info:
                        if audio_info['duration'] < self.min_audio_duration or \
                           audio_info['duration'] > self.max_audio_duration:
                            self.stats['invalid_duration'] += 1
                    continue
                
                # Validate transcript
                transcript_valid, cleaned_transcript = self.validate_transcript(row['transcript'])
                
                if not transcript_valid:
                    self.stats['invalid_transcript'] += 1
                    continue
                
                # Update row with cleaned transcript
                row['transcript'] = cleaned_transcript
                if cleaned_transcript != row['transcript']:
                    self.stats['cleaned'] += 1
                
                # Update words_json and phonemes_json if transcript changed significantly
                # (For now, we keep original timestamps)
                
                valid_rows.append(row)
                self.stats['valid'] += 1
        
        # Write cleaned CSV
        if valid_rows:
            output_csv = output_dir / 'train.csv'
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'transcript', 'audio_path', 'words_json', 
                            'phonemes_json', 'sex', 'subset']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
        
        # Write cleaned timestamps (only for valid entries)
        if timestamps_dict and valid_rows:
            valid_ids = {row['id'] for row in valid_rows}
            cleaned_timestamps = [
                timestamps_dict[id] for id in valid_ids 
                if id in timestamps_dict
            ]
            
            output_timestamps = output_dir / 'timestamps.json'
            with open(output_timestamps, 'w', encoding='utf-8') as f:
                json.dump(cleaned_timestamps, f, ensure_ascii=False, indent=2)
        
        # Copy valid audio files if output_dir is different
        if output_dir != split_dir:
            output_audio_dir = output_dir / 'audio'
            for row in valid_rows:
                src_audio = split_dir / row['audio_path']
                dst_audio = output_audio_dir / os.path.basename(row['audio_path'])
                if src_audio.exists():
                    import shutil
                    shutil.copy2(src_audio, dst_audio)
        
        return {
            'split': split_dir.name,
            'total': self.stats['total'],
            'valid': self.stats['valid'],
            'invalid_audio': self.stats['invalid_audio'],
            'invalid_duration': self.stats['invalid_duration'],
            'invalid_transcript': self.stats['invalid_transcript'],
            'duplicates': self.stats['duplicates'],
            'cleaned': self.stats['cleaned']
        }
    
    def clean_all_splits(self, base_dir: Path, output_base: Path = None):
        """Clean all splits in the dataset.
        
        Args:
            base_dir: Base directory containing all splits
            output_base: Output base directory (if None, overwrites original)
        """
        splits = sorted([d for d in base_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('split_')])
        
        print(f"Found {len(splits)} splits to clean")
        print(f"Settings:")
        print(f"  Min audio duration: {self.min_audio_duration}s")
        print(f"  Max audio duration: {self.max_audio_duration}s")
        print(f"  Min transcript length: {self.min_transcript_length} chars")
        print(f"  Max transcript length: {self.max_transcript_length} chars")
        print(f"  Clean text: {self.clean_text}")
        print(f"  Language: {self.language}")
        print()
        
        all_stats = []
        
        for split_dir in tqdm(splits, desc="Processing splits"):
            if output_base:
                output_dir = output_base / split_dir.name
            else:
                output_dir = None
            
            stats = self.clean_split(split_dir, output_dir)
            all_stats.append(stats)
            
            # Reset per-split stats
            self.stats = {k: 0 for k in self.stats.keys()}
        
        # Print summary
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        
        total_stats = {
            'total': sum(s['total'] for s in all_stats),
            'valid': sum(s['valid'] for s in all_stats),
            'invalid_audio': sum(s['invalid_audio'] for s in all_stats),
            'invalid_duration': sum(s['invalid_duration'] for s in all_stats),
            'invalid_transcript': sum(s['invalid_transcript'] for s in all_stats),
            'duplicates': sum(s['duplicates'] for s in all_stats),
            'cleaned': sum(s['cleaned'] for s in all_stats)
        }
        
        print(f"Total samples: {total_stats['total']:,}")
        print(f"Valid samples: {total_stats['valid']:,} ({100*total_stats['valid']/total_stats['total']:.1f}%)")
        print(f"Invalid audio: {total_stats['invalid_audio']:,}")
        print(f"  - Invalid duration: {total_stats['invalid_duration']:,}")
        print(f"Invalid transcript: {total_stats['invalid_transcript']:,}")
        print(f"Duplicates: {total_stats['duplicates']:,}")
        print(f"Text cleaned: {total_stats['cleaned']:,}")
        print()
        
        # Save summary
        if output_base:
            summary_path = output_base / 'cleaning_summary.json'
        else:
            summary_path = base_dir / 'cleaning_summary.json'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'settings': {
                    'min_audio_duration': self.min_audio_duration,
                    'max_audio_duration': self.max_audio_duration,
                    'min_transcript_length': self.min_transcript_length,
                    'max_transcript_length': self.max_transcript_length,
                    'clean_text': self.clean_text,
                    'language': self.language
                },
                'summary': total_stats,
                'per_split': all_stats
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Clean LibriSpeech alignments dataset')
    parser.add_argument('--input', type=str, 
                       default='data/raw/librispeech_alignments',
                       help='Input directory containing splits')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (if None, overwrites input)')
    parser.add_argument('--min-duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max-duration', type=float, default=30.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--min-transcript', type=int, default=3,
                       help='Minimum transcript length in characters')
    parser.add_argument('--max-transcript', type=int, default=500,
                       help='Maximum transcript length in characters')
    parser.add_argument('--no-clean-text', action='store_true',
                       help='Skip text cleaning/normalization')
    parser.add_argument('--language', type=str, default='en',
                       choices=['en', 'vi'],
                       help='Language for text cleaning')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    cleaner = LibriSpeechDataCleaner(
        min_audio_duration=args.min_duration,
        max_audio_duration=args.max_duration,
        min_transcript_length=args.min_transcript,
        max_transcript_length=args.max_transcript,
        clean_text=not args.no_clean_text,
        language=args.language
    )
    
    cleaner.clean_all_splits(input_dir, output_dir)


if __name__ == '__main__':
    main()

