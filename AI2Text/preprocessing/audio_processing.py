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
        
        Prioritizes torchaudio for faster mp3 decoding, falls back to librosa
        for rare formats not supported by torchaudio on the current system.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio amplitude
            
        Returns:
            audio: Audio waveform as numpy array
            sr: Sample rate
        """
        audio: Optional[np.ndarray] = None
        sr: int = self.sample_rate
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert stereo -> mono by averaging channels
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate
            
            audio = waveform.squeeze(0).cpu().numpy()
        except Exception:
            # Fallback to librosa for exotic codecs; use faster resampling
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                res_type='kaiser_fast'
            )
        
        if normalize and audio is not None:
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
        # Convert to tensor with explicit channel dimension
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db.squeeze(0).cpu().numpy()
    
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
    
    def extract_features(self, audio: np.ndarray, feature_type: str = "mel",
                         **kwargs) -> np.ndarray:
        """High-level feature extraction helper used by tests and integration.
        
        Args:
            audio: Audio waveform array
            feature_type: Type of feature to return ('mel' or 'mfcc')
            **kwargs: Extra keyword arguments passed to lower-level extractors
        
        Returns:
            features: Feature matrix shaped (time, feature_dim)
        """
        if feature_type == "mel":
            mel_spec = self.extract_mel_spectrogram(audio)
            return mel_spec.T
        if feature_type == "mfcc":
            n_mfcc = kwargs.get("n_mfcc", 13)
            mfcc = self.extract_mfcc(audio, n_mfcc=n_mfcc)
            return mfcc.T
        raise ValueError(f"Unsupported feature_type '{feature_type}'.")
    
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
    
    def __init__(self, sample_rate: int = 16000, 
                 aggressive: bool = True):
        """Initialize audio augmenter.
        
        Args:
            sample_rate: Audio sample rate
            aggressive: If True, use more aggressive augmentation for harder training
        """
        self.sample_rate = sample_rate
        self.aggressive = aggressive
    
    def add_noise(self, audio: np.ndarray, 
                  noise_factor: Optional[float] = None) -> np.ndarray:
        """Add random Gaussian noise.
        
        Args:
            audio: Audio waveform
            noise_factor: Standard deviation of noise (auto if None based on aggressive mode)
            
        Returns:
            noisy_audio: Audio with added noise
        """
        if noise_factor is None:
            # More aggressive noise for harder training
            noise_factor = 0.02 if self.aggressive else 0.005
        noise = np.random.randn(len(audio)) * noise_factor
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, 
                   shift_ms: Optional[float] = None,
                   shift_max: float = 0.2) -> np.ndarray:
        """Randomly shift audio in time.
        
        Args:
            audio: Audio waveform
            shift_ms: Specific shift in milliseconds (positive values shift forward)
            shift_max: Maximum shift as fraction of length when shift_ms is None
            
        Returns:
            shifted_audio: Time-shifted audio
        """
        if shift_ms is not None:
            max_shift_samples = len(audio)
            shift = int((shift_ms / 1000.0) * self.sample_rate)
            shift = max(-max_shift_samples, min(max_shift_samples, shift))
        else:
            shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        return np.roll(audio, shift)
    
    def time_stretch(self, audio: np.ndarray, 
                     rate: Optional[float] = None,
                     rate_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Randomly stretch or compress audio in time.
        
        Args:
            audio: Audio waveform
            rate: Specific stretch rate. If None, choose randomly within rate_range
            rate_range: (min_rate, max_rate) for stretching when rate is None (auto if None)
            
        Returns:
            stretched_audio: Time-stretched audio
        """
        if rate_range is None:
            # More aggressive time stretching for harder training
            rate_range = (0.7, 1.3) if self.aggressive else (0.8, 1.2)
        rate = rate if rate is not None else np.random.uniform(rate_range[0], rate_range[1])
        rate = max(rate, 1e-3)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, 
                    n_steps: Optional[int] = None,
                    n_steps_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Randomly shift pitch.
        
        Args:
            audio: Audio waveform
            n_steps: Specific number of semitones to shift. If None use random range.
            n_steps_range: (min_steps, max_steps) in semitones (auto if None)
            
        Returns:
            pitch_shifted_audio: Pitch-shifted audio
        """
        if n_steps_range is None:
            # More aggressive pitch shifting for harder training
            n_steps_range = (-4, 4) if self.aggressive else (-2, 2)
        if n_steps is None:
            n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )
    
    def volume_change(self, audio: np.ndarray, 
                      factor: Optional[float] = None,
                      gain_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Randomly change volume.
        
        Args:
            audio: Audio waveform
            factor: Explicit multiplier. If None, draw randomly from gain_range.
            gain_range: (min_gain, max_gain) multipliers when factor is None (auto if None)
            
        Returns:
            volume_changed_audio: Audio with changed volume
        """
        if gain_range is None:
            # More aggressive volume changes for harder training
            gain_range = (0.3, 1.8) if self.aggressive else (0.5, 1.5)
        gain = factor if factor is not None else np.random.uniform(gain_range[0], gain_range[1])
        return audio * gain
    
    def change_volume(self, audio: np.ndarray, 
                      gain_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
        """Backward compatible alias for volume_change."""
        return self.volume_change(audio, gain_range=gain_range)
    
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
                     freq_mask_param: Optional[int] = None,
                     time_mask_param: Optional[int] = None,
                     num_freq_masks: Optional[int] = None,
                     num_time_masks: Optional[int] = None) -> np.ndarray:
        """Apply SpecAugment (frequency and time masking).
        
        Args:
            mel_spec: Mel spectrogram (n_mels, time)
            freq_mask_param: Maximum frequency mask size (auto if None)
            time_mask_param: Maximum time mask size (auto if None)
            num_freq_masks: Number of frequency masks (auto if None)
            num_time_masks: Number of time masks (auto if None)
            
        Returns:
            augmented_spec: Augmented spectrogram
        """
        # More aggressive SpecAugment for harder training
        if freq_mask_param is None:
            freq_mask_param = 25 if self.aggressive else 15
        if time_mask_param is None:
            time_mask_param = 50 if self.aggressive else 35
        if num_freq_masks is None:
            num_freq_masks = 3 if self.aggressive else 2
        if num_time_masks is None:
            num_time_masks = 3 if self.aggressive else 2
        
        spec = mel_spec.copy()
        n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, n_frames - t))
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
            # More aggressive augmentation for harder training
            if self.aggressive:
                # Apply all augmentations for maximum difficulty
                augmentation_types = ['noise', 'volume', 'shift', 'stretch', 'pitch']
            else:
                augmentation_types = ['noise', 'volume', 'shift']
        
        augmented = audio.copy()
        
        # Apply augmentations randomly (not all, but more in aggressive mode)
        if self.aggressive:
            # In aggressive mode, apply 3-5 random augmentations
            num_augs = np.random.randint(3, 6)
            selected_augs = np.random.choice(augmentation_types, size=min(num_augs, len(augmentation_types)), replace=False)
        else:
            # In normal mode, apply 2-3 random augmentations with 50% chance each
            selected_augs = [aug for aug in augmentation_types if np.random.random() < 0.5]
            if len(selected_augs) == 0:
                selected_augs = [np.random.choice(augmentation_types)]  # At least one
        
        for aug_type in selected_augs:
            if aug_type == 'noise':
                augmented = self.add_noise(augmented)
            elif aug_type == 'volume':
                augmented = self.volume_change(augmented)
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

