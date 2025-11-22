"""
Audio preprocessing and feature extraction for Vietnamese ASR.
Includes noise reduction, augmentation, and feature extraction (spectrograms, mel features).
"""

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """Handles all audio preprocessing operations."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 win_length: int = 400,
                 fmin: float = 0.0,
                 fmax: Optional[float] = 8000.0):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate (16kHz standard for ASR)
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window length
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def load_audio(self, audio_path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio amplitude
            
        Returns:
            audio: Audio waveform as numpy array
            sr: Sample rate
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if normalize:
            audio = librosa.util.normalize(audio)
        
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
        """
        sf.write(output_path, audio, self.sample_rate)
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features.
        
        Args:
            audio: Audio waveform
            
        Returns:
            mel_spec: Mel spectrogram (n_mels, time)
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db.numpy()
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features.
        
        Args:
            audio: Audio waveform
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            mfcc: MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """Compute frame-wise energy."""
        return librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
    
    def trim_silence(self, audio: np.ndarray, 
                     top_db: float = 20.0) -> np.ndarray:
        """Trim leading and trailing silence.
        
        Args:
            audio: Audio waveform
            top_db: Threshold in dB below reference
            
        Returns:
            trimmed_audio: Audio with silence removed
        """
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio
    
    def pad_or_truncate(self, audio: np.ndarray, 
                        target_length: int) -> np.ndarray:
        """Pad or truncate audio to target length.
        
        Args:
            audio: Audio waveform
            target_length: Target length in samples
            
        Returns:
            processed_audio: Padded or truncated audio
        """
        if len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]
        
        return audio


class AudioAugmenter:
    """Audio augmentation for robust ASR training."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio: np.ndarray, 
                  noise_factor: float = 0.005) -> np.ndarray:
        """Add random Gaussian noise.
        
        Args:
            audio: Audio waveform
            noise_factor: Standard deviation of noise
            
        Returns:
            noisy_audio: Audio with added noise
        """
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, 
                   shift_max: float = 0.2) -> np.ndarray:
        """Randomly shift audio in time.
        
        Args:
            audio: Audio waveform
            shift_max: Maximum shift as fraction of length
            
        Returns:
            shifted_audio: Time-shifted audio
        """
        shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        return np.roll(audio, shift)
    
    def time_stretch(self, audio: np.ndarray, 
                     rate_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Randomly stretch or compress audio in time.
        
        Args:
            audio: Audio waveform
            rate_range: (min_rate, max_rate) for stretching
            
        Returns:
            stretched_audio: Time-stretched audio
        """
        rate = np.random.uniform(rate_range[0], rate_range[1])
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, 
                    n_steps_range: Tuple[int, int] = (-2, 2)) -> np.ndarray:
        """Randomly shift pitch.
        
        Args:
            audio: Audio waveform
            n_steps_range: (min_steps, max_steps) in semitones
            
        Returns:
            pitch_shifted_audio: Pitch-shifted audio
        """
        n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )
    
    def change_volume(self, audio: np.ndarray, 
                      gain_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
        """Randomly change volume.
        
        Args:
            audio: Audio waveform
            gain_range: (min_gain, max_gain) multipliers
            
        Returns:
            volume_changed_audio: Audio with changed volume
        """
        gain = np.random.uniform(gain_range[0], gain_range[1])
        return audio * gain
    
    def add_background_noise(self, audio: np.ndarray, 
                             noise_audio: np.ndarray,
                             snr_db: float = 10.0) -> np.ndarray:
        """Add background noise at specific SNR.
        
        Args:
            audio: Clean audio waveform
            noise_audio: Noise waveform
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            noisy_audio: Audio with background noise
        """
        # Match noise length to audio
        if len(noise_audio) < len(audio):
            # Repeat noise
            repeats = int(np.ceil(len(audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)[:len(audio)]
        else:
            # Random crop
            start = np.random.randint(0, len(noise_audio) - len(audio))
            noise_audio = noise_audio[start:start + len(audio)]
        
        # Calculate current power
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # Calculate required noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = audio_power / snr_linear
        
        # Scale noise
        if noise_power > 0:
            noise_audio = noise_audio * np.sqrt(target_noise_power / noise_power)
        
        return audio + noise_audio
    
    def spec_augment(self, mel_spec: np.ndarray, 
                     freq_mask_param: int = 15,
                     time_mask_param: int = 35,
                     num_freq_masks: int = 2,
                     num_time_masks: int = 2) -> np.ndarray:
        """Apply SpecAugment (frequency and time masking).
        
        Args:
            mel_spec: Mel spectrogram (n_mels, time)
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            
        Returns:
            augmented_spec: Augmented spectrogram
        """
        spec = mel_spec.copy()
        n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, n_mels - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, n_frames - t)
            spec[:, t0:t0 + t] = 0
        
        return spec
    
    def augment(self, audio: np.ndarray, 
                augmentation_types: list = None) -> np.ndarray:
        """Apply random augmentations.
        
        Args:
            audio: Audio waveform
            augmentation_types: List of augmentation types to apply
            
        Returns:
            augmented_audio: Augmented audio
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'volume', 'shift']
        
        augmented = audio.copy()
        
        for aug_type in augmentation_types:
            if np.random.random() < 0.5:  # 50% chance to apply each
                if aug_type == 'noise':
                    augmented = self.add_noise(augmented)
                elif aug_type == 'volume':
                    augmented = self.change_volume(augmented)
                elif aug_type == 'shift':
                    augmented = self.time_shift(augmented)
                elif aug_type == 'stretch':
                    augmented = self.time_stretch(augmented)
                elif aug_type == 'pitch':
                    augmented = self.pitch_shift(augmented)
        
        return augmented


def preprocess_audio_file(file_path: str,
                          output_dir: Optional[str] = None,
                          processor: Optional[AudioProcessor] = None,
                          augmenter: Optional[AudioAugmenter] = None,
                          apply_augmentation: bool = False,
                          extract_features: bool = True) -> Dict[str, Any]:
    """Complete preprocessing pipeline for a single audio file.
    
    Args:
        file_path: Path to audio file
        output_dir: Directory to save processed files
        processor: AudioProcessor instance
        augmenter: AudioAugmenter instance
        apply_augmentation: Whether to apply augmentation
        extract_features: Whether to extract features
        
    Returns:
        result: Dictionary with processed data and metadata
    """
    if processor is None:
        processor = AudioProcessor()
    
    if augmenter is None and apply_augmentation:
        augmenter = AudioAugmenter()
    
    # Load audio
    audio, sr = processor.load_audio(file_path)
    
    # Trim silence
    audio = processor.trim_silence(audio)
    
    # Apply augmentation if requested
    if apply_augmentation and augmenter:
        audio = augmenter.augment(audio)
    
    result = {
        'file_path': file_path,
        'audio': audio,
        'sample_rate': sr,
        'duration': len(audio) / sr
    }
    
    # Extract features
    if extract_features:
        mel_spec = processor.extract_mel_spectrogram(audio)
        result['mel_spectrogram'] = mel_spec
        result['feature_shape'] = mel_spec.shape
    
    # Save processed audio if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = Path(file_path).stem + '_processed.wav'
        output_file = output_path / filename
        processor.save_audio(audio, str(output_file))
        result['processed_path'] = str(output_file)
    
    return result


if __name__ == "__main__":
    # Test audio processing
    processor = AudioProcessor()
    print("Audio processor initialized")
    print(f"Sample rate: {processor.sample_rate} Hz")
    print(f"Mel bands: {processor.n_mels}")
    print(f"FFT size: {processor.n_fft}")

