"""
Noise Reduction Module
Giảm nhiễu và ồn trong audio sử dụng các thuật toán khác nhau
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not installed. Using basic noise reduction.")


class NoiseReducer:
    """Class để giảm nhiễu trong audio"""
    
    def __init__(self, method: str = 'spectral_gating'):
        """
        Args:
            method: Phương pháp giảm nhiễu
                - 'spectral_gating': Spectral gating (nhanh, hiệu quả)
                - 'wiener': Wiener filter (tốt cho stationary noise)
                - 'stationary': Stationary noise reduction
                - 'nonstationary': Non-stationary noise reduction
        """
        self.method = method
        self.sample_rate = None
        self.audio_data = None
    
    def load_audio(self, audio_path: Union[str, Path, np.ndarray], 
                   sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio file hoặc nhận audio array
        
        Args:
            audio_path: Đường dẫn file hoặc audio array
            sr: Sample rate (nếu None sẽ tự detect)
        
        Returns:
            Audio data as numpy array
        """
        if isinstance(audio_path, np.ndarray):
            self.audio_data = audio_path
            if sr is None:
                raise ValueError("Sample rate required when passing audio array")
            self.sample_rate = sr
        else:
            self.audio_data, self.sample_rate = librosa.load(
                str(audio_path), 
                sr=sr,
                mono=True
            )
        
        return self.audio_data
    
    def reduce_noise(self, 
                    audio: Optional[np.ndarray] = None,
                    stationary: bool = False,
                    prop_decrease: float = 0.8,
                    n_std_thresh: float = 2.0) -> np.ndarray:
        """
        Giảm nhiễu trong audio
        
        Args:
            audio: Audio data (nếu None sẽ dùng self.audio_data)
            stationary: Nếu True, giả định noise là stationary
            prop_decrease: Tỷ lệ giảm noise (0.0 - 1.0)
            n_std_thresh: Ngưỡng standard deviation
        
        Returns:
            Audio đã được xử lý
        """
        if audio is not None:
            audio_data = audio
            sr = self.sample_rate
        else:
            if self.audio_data is None:
                raise ValueError("No audio data loaded")
            audio_data = self.audio_data
            sr = self.sample_rate
        
        if NOISEREDUCE_AVAILABLE:
            if self.method == 'spectral_gating':
                reduced = nr.reduce_noise(
                    y=audio_data,
                    sr=sr,
                    stationary=stationary,
                    prop_decrease=prop_decrease,
                    n_std_thresh_stationary=n_std_thresh
                )
            elif self.method == 'wiener':
                reduced = nr.reduce_noise(
                    y=audio_data,
                    sr=sr,
                    stationary=True,
                    prop_decrease=prop_decrease
                )
            elif self.method == 'stationary':
                reduced = nr.reduce_noise(
                    y=audio_data,
                    sr=sr,
                    stationary=True,
                    prop_decrease=prop_decrease
                )
            elif self.method == 'nonstationary':
                reduced = nr.reduce_noise(
                    y=audio_data,
                    sr=sr,
                    stationary=False,
                    prop_decrease=prop_decrease
                )
            else:
                reduced = self._basic_noise_reduction(audio_data, sr)
        else:
            reduced = self._basic_noise_reduction(audio_data, sr)
        
        return reduced
    
    def _basic_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Basic noise reduction sử dụng spectral subtraction"""
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * sr / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_audio
    
    def save_audio(self, audio: np.ndarray, output_path: Union[str, Path], 
                   format: str = 'wav'):
        """Lưu audio đã xử lý"""
        sf.write(str(output_path), audio, self.sample_rate, format=format)
    
    def process_file(self, input_path: Union[str, Path], 
                    output_path: Union[str, Path],
                    **kwargs) -> np.ndarray:
        """
        Xử lý file audio từ đầu đến cuối
        
        Args:
            input_path: Đường dẫn file input
            output_path: Đường dẫn file output
            **kwargs: Các tham số cho reduce_noise
        
        Returns:
            Audio đã được xử lý
        """
        self.load_audio(input_path)
        processed = self.reduce_noise(**kwargs)
        self.save_audio(processed, output_path)
        return processed


def reduce_noise_file(input_path: str, output_path: str, 
                     method: str = 'spectral_gating',
                     **kwargs) -> np.ndarray:
    """
    Helper function để giảm nhiễu file audio
    
    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output
        method: Phương pháp giảm nhiễu
        **kwargs: Các tham số khác
    
    Returns:
        Audio đã được xử lý
    """
    reducer = NoiseReducer(method=method)
    return reducer.process_file(input_path, output_path, **kwargs)


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else 'spectral_gating'
        
        print(f"Processing {input_file}...")
        reduce_noise_file(input_file, output_file, method=method)
        print(f"Saved to {output_file}")

