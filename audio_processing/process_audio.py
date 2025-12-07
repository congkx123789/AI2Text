#!/usr/bin/env python3
"""
Audio Processing CLI Tool
Công cụ dòng lệnh để xử lý audio: giảm nhiễu, lọc, cải thiện chất lượng
"""

import argparse
import sys
from pathlib import Path
from .noise_reduction import NoiseReducer
from .audio_filters import AudioFilter
from .audio_enhancer import AudioEnhancer
import librosa
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description='Audio Processing Tool - Giảm nhiễu, lọc, cải thiện chất lượng audio'
    )
    
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', help='Output audio file')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (default: 16000)')
    
    # Noise reduction
    parser.add_argument('--noise-reduce', action='store_true', help='Giảm nhiễu')
    parser.add_argument('--noise-method', choices=['spectral_gating', 'wiener', 'stationary', 'nonstationary'],
                       default='spectral_gating', help='Phương pháp giảm nhiễu')
    parser.add_argument('--noise-prop', type=float, default=0.8, help='Tỷ lệ giảm nhiễu (0-1)')
    
    # Filters
    parser.add_argument('--high-pass', type=float, help='High-pass filter cutoff (Hz)')
    parser.add_argument('--low-pass', type=float, help='Low-pass filter cutoff (Hz)')
    parser.add_argument('--band-pass', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                       help='Band-pass filter (low high)')
    parser.add_argument('--remove-hum', action='store_true', help='Loại bỏ hum (50/60Hz)')
    parser.add_argument('--notch', type=float, help='Notch filter frequency (Hz)')
    
    # Enhancement
    parser.add_argument('--enhance-speech', action='store_true', help='Tự động cải thiện speech')
    parser.add_argument('--normalize', action='store_true', help='Normalize audio')
    parser.add_argument('--normalize-db', type=float, default=-3.0, help='Target dB for normalization')
    
    # EQ
    parser.add_argument('--bass-gain', type=float, default=0.0, help='Bass gain (dB)')
    parser.add_argument('--mid-gain', type=float, default=0.0, help='Mid gain (dB)')
    parser.add_argument('--treble-gain', type=float, default=0.0, help='Treble gain (dB)')
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Loading audio: {args.input}")
    audio, sr = librosa.load(args.input, sr=args.sr, mono=True)
    print(f"Sample rate: {sr} Hz, Duration: {len(audio)/sr:.2f} seconds")
    
    processed = audio.copy()
    
    # Noise reduction
    if args.noise_reduce:
        print(f"Reducing noise using {args.noise_method}...")
        reducer = NoiseReducer(method=args.noise_method)
        reducer.load_audio(processed, sr=sr)
        processed = reducer.reduce_noise(
            prop_decrease=args.noise_prop,
            stationary=(args.noise_method == 'stationary' or args.noise_method == 'wiener')
        )
    
    # Filters
    if args.high_pass or args.low_pass or args.band_pass or args.remove_hum or args.notch:
        print("Applying filters...")
        audio_filter = AudioFilter(sample_rate=sr)
        
        if args.high_pass:
            print(f"  High-pass filter: {args.high_pass} Hz")
            processed = audio_filter.high_pass_filter(processed, cutoff=args.high_pass)
        
        if args.low_pass:
            print(f"  Low-pass filter: {args.low_pass} Hz")
            processed = audio_filter.low_pass_filter(processed, cutoff=args.low_pass)
        
        if args.band_pass:
            print(f"  Band-pass filter: {args.band_pass[0]}-{args.band_pass[1]} Hz")
            processed = audio_filter.band_pass_filter(
                processed,
                low_cutoff=args.band_pass[0],
                high_cutoff=args.band_pass[1]
            )
        
        if args.remove_hum:
            print("  Removing hum (50/60/100/120 Hz)...")
            processed = audio_filter.remove_hum(processed)
        
        if args.notch:
            print(f"  Notch filter: {args.notch} Hz")
            processed = audio_filter.notch_filter(processed, freq=args.notch)
    
    # Enhancement
    if args.enhance_speech:
        print("Enhancing speech...")
        enhancer = AudioEnhancer(sample_rate=sr)
        processed = enhancer.enhance_speech(processed)
    
    # EQ
    if args.bass_gain != 0.0 or args.mid_gain != 0.0 or args.treble_gain != 0.0:
        print(f"Applying EQ (bass: {args.bass_gain}dB, mid: {args.mid_gain}dB, treble: {args.treble_gain}dB)...")
        enhancer = AudioEnhancer(sample_rate=sr)
        processed = enhancer.equalize(
            processed,
            bass_gain=args.bass_gain,
            mid_gain=args.mid_gain,
            treble_gain=args.treble_gain
        )
    
    # Normalize
    if args.normalize:
        print(f"Normalizing to {args.normalize_db} dB...")
        audio_filter = AudioFilter(sample_rate=sr)
        processed = audio_filter.normalize(processed, target_db=args.normalize_db)
    
    # Save
    print(f"Saving to: {args.output}")
    sf.write(args.output, processed, sr)
    print("Done!")


if __name__ == '__main__':
    main()

