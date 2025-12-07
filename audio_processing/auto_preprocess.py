"""
Auto Preprocessing Module
Tự động xử lý audio để chuẩn bị cho ASR models
Đây là preprocessing mặc định, không phải tính năng người dùng chọn
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

from .noise_reduction import NoiseReducer
from .audio_filters import AudioFilter
from .audio_enhancer import AudioEnhancer


def auto_preprocess_audio(audio_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          sample_rate: int = 16000,
                          return_array: bool = False,
                          model_type: Optional[str] = None) -> Union[np.ndarray, str]:
    """
    Tự động xử lý audio để chuẩn bị cho ASR models
    
    Quy trình:
    1. Load và convert về 16kHz mono (chuẩn cho tất cả models)
    2. Giảm nhiễu (spectral gating)
    3. High-pass filter (80Hz) - loại bỏ rumble
    4. Low-pass filter (8000Hz) - loại bỏ hiss
    5. Remove hum (50/60Hz)
    6. Enhance speech (nếu cần)
    7. Normalize về -3dB
    8. Đảm bảo format đúng cho model (16kHz mono WAV)
    
    Args:
        audio_path: Đường dẫn file audio input
        output_path: Đường dẫn file output (nếu None sẽ tạo temp file)
        sample_rate: Sample rate mục tiêu (mặc định: 16000)
        return_array: Nếu True, trả về numpy array thay vì file path
        model_type: Loại model ('ai-llm-ss', 'ai-llm', 'ai2text', None=auto)
    
    Returns:
        Nếu return_array=True: numpy array (16kHz mono, float32, normalized)
        Nếu return_array=False: đường dẫn file WAV (16kHz mono, PCM 16-bit)
    """
    # Load audio - đảm bảo 16kHz mono
    audio, sr = librosa.load(
        str(audio_path), 
        sr=sample_rate, 
        mono=True,
        res_type='kaiser_fast'  # Fast resampling
    )
    
    # Đảm bảo là mono (nếu có nhiều channels, lấy channel đầu)
    if len(audio.shape) > 1:
        audio = audio[0] if audio.shape[0] > 1 else audio.flatten()
    
    # Đảm bảo sample rate đúng
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type='kaiser_fast')
        sr = sample_rate
    
    print(f"[Auto Preprocess] Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz, mono: True")
    
    processed = audio.copy()
    
    # 1. Noise Reduction
    try:
        if NOISEREDUCE_AVAILABLE:
            print("[Auto Preprocess] Applying noise reduction...")
            reducer = NoiseReducer(method='spectral_gating')
            reducer.load_audio(processed, sr=sr)
            processed = reducer.reduce_noise(prop_decrease=0.7, stationary=False)
        else:
            # Basic noise reduction
            print("[Auto Preprocess] Applying basic noise reduction...")
            stft = librosa.stft(processed, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first 0.5 seconds
            noise_frames = int(0.5 * sr / 512)
            if noise_frames > 0:
                noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
                enhanced_magnitude = magnitude - 1.5 * noise_spectrum
                enhanced_magnitude = np.maximum(enhanced_magnitude, 0.01 * magnitude)
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                processed = librosa.istft(enhanced_stft, hop_length=512)
    except Exception as e:
        print(f"[Auto Preprocess] Warning: Noise reduction failed: {e}")
    
    # 2. Audio Filters
    try:
        print("[Auto Preprocess] Applying filters...")
        audio_filter = AudioFilter(sample_rate=sr)
        
        # High-pass: loại bỏ rumble, bass noise
        processed = audio_filter.high_pass_filter(processed, cutoff=80.0)
        
        # Low-pass: loại bỏ hiss, high-frequency noise
        processed = audio_filter.low_pass_filter(processed, cutoff=8000.0)
        
        # Remove hum: 50Hz, 60Hz và harmonics
        processed = audio_filter.remove_hum(processed, frequencies=[50, 60, 100, 120])
    except Exception as e:
        print(f"[Auto Preprocess] Warning: Filtering failed: {e}")
    
    # 3. Speech Enhancement (light)
    try:
        print("[Auto Preprocess] Enhancing speech...")
        enhancer = AudioEnhancer(sample_rate=sr)
        # Light enhancement - chỉ boost mid frequencies một chút
        processed = enhancer.equalize(processed, bass_gain=-1.0, mid_gain=2.0, treble_gain=0.5)
    except Exception as e:
        print(f"[Auto Preprocess] Warning: Enhancement failed: {e}")
    
    # 4. Normalize
    try:
        print("[Auto Preprocess] Normalizing...")
        audio_filter = AudioFilter(sample_rate=sr)
        processed = audio_filter.normalize(processed, target_db=-3.0)
    except Exception as e:
        print(f"[Auto Preprocess] Warning: Normalization failed: {e}")
    
    # Return
    if return_array:
        return processed
    
    # Save to file - đảm bảo format đúng cho ASR models
    if output_path is None:
        import tempfile
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    # Đảm bảo audio là float32 và normalized trước khi save
    if processed.dtype != np.float32:
        processed = processed.astype(np.float32)
    
    # Normalize để tránh clipping khi convert sang int16
    max_val = np.max(np.abs(processed))
    if max_val > 1.0:
        processed = processed / max_val * 0.95
    
    # Save as 16kHz mono WAV, PCM 16-bit (chuẩn cho ASR)
    # Models yêu cầu:
    # - ai-llm-ss: 16kHz mono WAV
    # - ai-llm (faster-whisper): 16kHz mono (tự động xử lý)
    # - AI2Text: 16kHz mono WAV
    sf.write(
        str(output_path), 
        processed, 
        sr,
        subtype='PCM_16'  # 16-bit PCM - chuẩn cho ASR
    )
    print(f"[Auto Preprocess] Saved processed audio to: {output_path}")
    print(f"[Auto Preprocess] Format: 16kHz mono WAV, PCM 16-bit (compatible with all ASR models)")
    
    return str(output_path)


def preprocess_audio_array(audio: np.ndarray,
                           sample_rate: int = 16000,
                           model_type: Optional[str] = None) -> np.ndarray:
    """
    Preprocess audio array (không cần file)
    
    Args:
        audio: Audio array (có thể là mono hoặc stereo)
        sample_rate: Sample rate hiện tại
        model_type: Loại model ('ai-llm-ss', 'ai-llm', 'ai2text')
    
    Returns:
        Processed audio array (16kHz mono, float32, normalized)
    """
    import tempfile
    import os
    
    # Đảm bảo là numpy array
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    
    # Convert stereo to mono nếu cần
    if len(audio.shape) > 1:
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        else:
            audio = audio.flatten()
    
    # Resample nếu cần
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
        sample_rate = 16000
    
    # Save to temp file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_input.name, audio, sample_rate)
    temp_input.close()
    
    try:
        # Process
        processed = auto_preprocess_audio(
            temp_input.name,
            return_array=True,
            sample_rate=16000,
            model_type=model_type
        )
        return processed
    finally:
        # Cleanup
        try:
            os.unlink(temp_input.name)
        except:
            pass


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'output.wav'
        
        print(f"Auto preprocessing {input_file}...")
        result = auto_preprocess_audio(input_file, output_file)
        print(f"Done! Output: {result}")

