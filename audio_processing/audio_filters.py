"""
Audio Filters Module
Các bộ lọc audio: high-pass, low-pass, band-pass, notch filter
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Optional, Union, Tuple
from pathlib import Path


class AudioFilter:
    """Class để áp dụng các bộ lọc audio"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: Sample rate của audio (Hz)
        """
        self.sample_rate = sample_rate
    
    def high_pass_filter(self, audio: np.ndarray, 
                        cutoff: float = 80.0,
                        order: int = 5) -> np.ndarray:
        """
        High-pass filter - Loại bỏ tần số thấp (bass, rumble)
        
        Args:
            audio: Audio data
            cutoff: Tần số cắt (Hz)
            order: Bậc của filter
        
        Returns:
            Audio đã được lọc
        """
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def low_pass_filter(self, audio: np.ndarray,
                       cutoff: float = 8000.0,
                       order: int = 5) -> np.ndarray:
        """
        Low-pass filter - Loại bỏ tần số cao (hiss, noise)
        
        Args:
            audio: Audio data
            cutoff: Tần số cắt (Hz)
            order: Bậc của filter
        
        Returns:
            Audio đã được lọc
        """
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def band_pass_filter(self, audio: np.ndarray,
                        low_cutoff: float = 80.0,
                        high_cutoff: float = 8000.0,
                        order: int = 5) -> np.ndarray:
        """
        Band-pass filter - Chỉ giữ lại tần số trong dải
        
        Args:
            audio: Audio data
            low_cutoff: Tần số cắt thấp (Hz)
            high_cutoff: Tần số cắt cao (Hz)
            order: Bậc của filter
        
        Returns:
            Audio đã được lọc
        """
        nyquist = self.sample_rate / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def notch_filter(self, audio: np.ndarray,
                    freq: float = 60.0,
                    Q: float = 30.0) -> np.ndarray:
        """
        Notch filter - Loại bỏ tần số cụ thể (ví dụ: 60Hz hum)
        
        Args:
            audio: Audio data
            freq: Tần số cần loại bỏ (Hz)
            Q: Quality factor (càng cao càng hẹp)
        
        Returns:
            Audio đã được lọc
        """
        b, a = signal.iirnotch(freq, Q, self.sample_rate)
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def remove_hum(self, audio: np.ndarray, 
                  frequencies: list = [50, 60, 100, 120]) -> np.ndarray:
        """
        Loại bỏ hum (50Hz, 60Hz và harmonics)
        
        Args:
            audio: Audio data
            frequencies: Danh sách tần số cần loại bỏ
        
        Returns:
            Audio đã được lọc
        """
        filtered = audio.copy()
        for freq in frequencies:
            filtered = self.notch_filter(filtered, freq=freq, Q=30.0)
        return filtered
    
    def normalize(self, audio: np.ndarray, 
                 target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio về mức target dB
        
        Args:
            audio: Audio data
            target_db: Mức dB mong muốn
        
        Returns:
            Audio đã được normalize
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
        
        # Calculate current dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.95
        
        return normalized
    
    def apply_filters(self, audio: np.ndarray,
                     high_pass: Optional[float] = None,
                     low_pass: Optional[float] = None,
                     band_pass: Optional[Tuple[float, float]] = None,
                     remove_hum_freqs: Optional[list] = None,
                     normalize_db: Optional[float] = None) -> np.ndarray:
        """
        Áp dụng nhiều filters cùng lúc
        
        Args:
            audio: Audio data
            high_pass: Cutoff cho high-pass filter
            low_pass: Cutoff cho low-pass filter
            band_pass: (low, high) cho band-pass filter
            remove_hum_freqs: List tần số để loại bỏ hum
            normalize_db: Target dB để normalize
        
        Returns:
            Audio đã được xử lý
        """
        filtered = audio.copy()
        
        # Apply filters in order
        if high_pass:
            filtered = self.high_pass_filter(filtered, cutoff=high_pass)
        
        if low_pass:
            filtered = self.low_pass_filter(filtered, cutoff=low_pass)
        
        if band_pass:
            filtered = self.band_pass_filter(
                filtered, 
                low_cutoff=band_pass[0],
                high_cutoff=band_pass[1]
            )
        
        if remove_hum_freqs:
            filtered = self.remove_hum(filtered, frequencies=remove_hum_freqs)
        
        if normalize_db is not None:
            filtered = self.normalize(filtered, target_db=normalize_db)
        
        return filtered


def filter_audio_file(input_path: str, output_path: str,
                     sample_rate: int = 16000,
                     high_pass: Optional[float] = None,
                     low_pass: Optional[float] = None,
                     band_pass: Optional[Tuple[float, float]] = None,
                     remove_hum: bool = False,
                     normalize: bool = False) -> np.ndarray:
    """
    Helper function để filter file audio
    
    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output
        sample_rate: Sample rate
        high_pass: High-pass cutoff
        low_pass: Low-pass cutoff
        band_pass: (low, high) cho band-pass
        remove_hum: Có loại bỏ hum không
        normalize: Có normalize không
    
    Returns:
        Audio đã được xử lý
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    
    # Create filter
    audio_filter = AudioFilter(sample_rate=sr)
    
    # Apply filters
    hum_freqs = [50, 60, 100, 120] if remove_hum else None
    normalize_db = -3.0 if normalize else None
    
    filtered = audio_filter.apply_filters(
        audio,
        high_pass=high_pass,
        low_pass=low_pass,
        band_pass=band_pass,
        remove_hum_freqs=hum_freqs,
        normalize_db=normalize_db
    )
    
    # Save
    sf.write(output_path, filtered, sr)
    
    return filtered


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        print(f"Filtering {input_file}...")
        filter_audio_file(
            input_file, 
            output_file,
            high_pass=80.0,
            low_pass=8000.0,
            remove_hum=True,
            normalize=True
        )
        print(f"Saved to {output_file}")

