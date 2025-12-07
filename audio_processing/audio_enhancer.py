"""
Audio Enhancement Module
Cải thiện chất lượng audio: EQ, compression, de-essing
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Optional, Union
from pathlib import Path


class AudioEnhancer:
    """Class để cải thiện chất lượng audio"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: Sample rate của audio (Hz)
        """
        self.sample_rate = sample_rate
    
    def equalize(self, audio: np.ndarray,
                bass_gain: float = 0.0,
                mid_gain: float = 0.0,
                treble_gain: float = 0.0) -> np.ndarray:
        """
        Equalizer - Điều chỉnh bass, mid, treble
        
        Args:
            audio: Audio data
            bass_gain: Gain cho bass (dB)
            mid_gain: Gain cho mid (dB)
            treble_gain: Gain cho treble (dB)
        
        Returns:
            Audio đã được EQ
        """
        # Bass: 20-250 Hz
        bass_filter = signal.butter(4, 250 / (self.sample_rate / 2), btype='low')
        bass = signal.filtfilt(bass_filter[0], bass_filter[1], audio)
        bass_gain_linear = 10 ** (bass_gain / 20)
        
        # Mid: 250-4000 Hz
        mid_filter = signal.butter(4, [250, 4000], btype='band', fs=self.sample_rate)
        mid = signal.filtfilt(mid_filter[0], mid_filter[1], audio)
        mid_gain_linear = 10 ** (mid_gain / 20)
        
        # Treble: 4000+ Hz
        treble_filter = signal.butter(4, 4000 / (self.sample_rate / 2), btype='high')
        treble = signal.filtfilt(treble_filter[0], treble_filter[1], audio)
        treble_gain_linear = 10 ** (treble_gain / 20)
        
        # Combine
        enhanced = (bass * bass_gain_linear + 
                   mid * mid_gain_linear + 
                   treble * treble_gain_linear)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val * 0.95
        
        return enhanced
    
    def compress(self, audio: np.ndarray,
                threshold: float = -12.0,
                ratio: float = 4.0,
                attack: float = 0.003,
                release: float = 0.1) -> np.ndarray:
        """
        Compressor - Giảm dynamic range
        
        Args:
            audio: Audio data
            threshold: Ngưỡng (dB)
            ratio: Tỷ lệ compression (4:1, 8:1, etc.)
            attack: Thời gian attack (seconds)
            release: Thời gian release (seconds)
        
        Returns:
            Audio đã được compress
        """
        # Convert to dB
        threshold_linear = 10 ** (threshold / 20)
        
        # Calculate gain reduction
        abs_audio = np.abs(audio)
        over_threshold = abs_audio > threshold_linear
        
        # Calculate compression
        compressed = audio.copy()
        for i in range(len(audio)):
            if abs_audio[i] > threshold_linear:
                # Calculate how much over threshold
                over = abs_audio[i] - threshold_linear
                # Apply ratio
                reduced_over = over / ratio
                # New level
                new_level = threshold_linear + reduced_over
                # Apply gain reduction
                gain_reduction = new_level / abs_audio[i] if abs_audio[i] > 0 else 1.0
                compressed[i] = audio[i] * gain_reduction
        
        return compressed
    
    def de_ess(self, audio: np.ndarray,
              threshold: float = -6.0,
              freq_range: tuple = (4000, 10000)) -> np.ndarray:
        """
        De-esser - Giảm sibilance (s, sh sounds)
        
        Args:
            audio: Audio data
            threshold: Ngưỡng (dB)
            freq_range: Dải tần số sibilance (Hz)
        
        Returns:
            Audio đã được de-ess
        """
        # Band-pass filter for sibilance range
        nyquist = self.sample_rate / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        sibilance = signal.filtfilt(b, a, audio)
        
        # Calculate envelope
        envelope = np.abs(signal.hilbert(sibilance))
        
        # Threshold
        threshold_linear = 10 ** (threshold / 20)
        over_threshold = envelope > threshold_linear
        
        # Apply gain reduction
        gain_reduction = np.ones_like(audio)
        for i in range(len(audio)):
            if over_threshold[i]:
                # Reduce gain in sibilance range
                reduction = 0.5  # 50% reduction
                gain_reduction[i] = 1.0 - (reduction * (envelope[i] / threshold_linear - 1))
                gain_reduction[i] = np.clip(gain_reduction[i], 0.3, 1.0)
        
        # Apply to sibilance only
        de_essed = audio.copy()
        sibilance_reduced = sibilance * gain_reduction
        de_essed = de_essed - sibilance + sibilance_reduced
        
        return de_essed
    
    def enhance_speech(self, audio: np.ndarray,
                      noise_reduction: bool = True,
                      eq: bool = True,
                      normalize: bool = True) -> np.ndarray:
        """
        Tự động cải thiện speech audio
        
        Args:
            audio: Audio data
            noise_reduction: Có giảm nhiễu không
            eq: Có áp dụng EQ không
            normalize: Có normalize không
        
        Returns:
            Audio đã được cải thiện
        """
        enhanced = audio.copy()
        
        if noise_reduction:
            # Simple noise reduction using spectral subtraction
            stft = librosa.stft(enhanced, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise
            noise_frames = int(0.5 * self.sample_rate / 512)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - 2.0 * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.01 * magnitude)
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced = librosa.istft(enhanced_stft, hop_length=512)
        
        if eq:
            # Speech-friendly EQ: boost mid frequencies
            enhanced = self.equalize(enhanced, bass_gain=-2.0, mid_gain=3.0, treble_gain=1.0)
        
        if normalize:
            # Normalize to -3dB
            rms = np.sqrt(np.mean(enhanced**2))
            if rms > 0:
                target_db = -3.0
                current_db = 20 * np.log10(rms)
                gain_db = target_db - current_db
                gain_linear = 10 ** (gain_db / 20)
                enhanced = enhanced * gain_linear
                
                # Prevent clipping
                max_val = np.max(np.abs(enhanced))
                if max_val > 1.0:
                    enhanced = enhanced / max_val * 0.95
        
        return enhanced


def enhance_audio_file(input_path: str, output_path: str,
                      sample_rate: int = 16000,
                      enhance_speech: bool = True,
                      **kwargs) -> np.ndarray:
    """
    Helper function để enhance file audio
    
    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output
        sample_rate: Sample rate
        enhance_speech: Sử dụng auto speech enhancement
        **kwargs: Các tham số khác
    
    Returns:
        Audio đã được xử lý
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    
    # Create enhancer
    enhancer = AudioEnhancer(sample_rate=sr)
    
    # Enhance
    if enhance_speech:
        enhanced = enhancer.enhance_speech(audio, **kwargs)
    else:
        enhanced = audio
    
    # Save
    sf.write(output_path, enhanced, sr)
    
    return enhanced


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        print(f"Enhancing {input_file}...")
        enhance_audio_file(input_file, output_file)
        print(f"Saved to {output_file}")

