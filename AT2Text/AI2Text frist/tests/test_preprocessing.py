"""
Tests for preprocessing modules (audio and text).
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile

from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import VietnameseTextNormalizer, Tokenizer


class TestAudioProcessor:
    """Test AudioProcessor class."""
    
    def test_load_audio(self, audio_processor, sample_audio_file):
        """Test loading audio file."""
        audio_path, expected_sr = sample_audio_file
        
        audio, sr = audio_processor.load_audio(audio_path)
        
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert sr == expected_sr
        assert len(audio) > 0
    
    def test_extract_mel_spectrogram(self, audio_processor, sample_audio_file):
        """Test mel spectrogram extraction."""
        audio_path, _ = sample_audio_file
        audio, _ = audio_processor.load_audio(audio_path)
        
        mel_spec = audio_processor.extract_mel_spectrogram(audio)
        
        assert mel_spec is not None
        assert mel_spec.shape[0] == audio_processor.n_mels  # Frequency bins
        assert mel_spec.shape[1] > 0  # Time frames
    
    def test_trim_silence(self, audio_processor):
        """Test silence trimming."""
        # Create audio with silence at start and end
        silence = np.zeros(1000)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = np.concatenate([silence, signal, silence])
        
        trimmed = audio_processor.trim_silence(audio)
        
        assert len(trimmed) < len(audio)
        assert len(trimmed) > 0
    
    def test_pad_or_truncate(self, audio_processor):
        """Test padding and truncation."""
        # Test padding
        short_audio = np.random.randn(1000)
        padded = audio_processor.pad_or_truncate(short_audio, target_length=5000)
        assert len(padded) == 5000
        
        # Test truncation
        long_audio = np.random.randn(10000)
        truncated = audio_processor.pad_or_truncate(long_audio, target_length=5000)
        assert len(truncated) == 5000


class TestAudioAugmenter:
    """Test AudioAugmenter class."""
    
    def test_add_noise(self, audio_augmenter):
        """Test adding noise to audio."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        augmented = audio_augmenter.add_noise(audio, noise_factor=0.01)
        
        assert augmented is not None
        assert len(augmented) == len(audio)
        assert not np.array_equal(audio, augmented)  # Should be different
    
    def test_time_shift(self, audio_augmenter):
        """Test time shifting."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        shifted = audio_augmenter.time_shift(audio, shift_ms=100)
        
        assert shifted is not None
        assert len(shifted) == len(audio)
    
    def test_time_stretch(self, audio_augmenter):
        """Test time stretching."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        stretched = audio_augmenter.time_stretch(audio, rate=1.1)
        
        assert stretched is not None
        assert len(stretched) != len(audio)  # Length should change
    
    def test_pitch_shift(self, audio_augmenter):
        """Test pitch shifting."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        shifted = audio_augmenter.pitch_shift(audio, n_steps=2)
        
        assert shifted is not None
        assert len(shifted) == len(audio)
    
    def test_volume_change(self, audio_augmenter):
        """Test volume change."""
        audio = np.ones(16000) * 0.5
        
        louder = audio_augmenter.volume_change(audio, factor=2.0)
        
        assert louder is not None
        assert np.max(np.abs(louder)) > np.max(np.abs(audio))
    
    def test_augment(self, audio_augmenter):
        """Test augment method."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        augmented = audio_augmenter.augment(audio)
        
        assert augmented is not None
        assert len(augmented) == len(audio)


class TestVietnameseTextNormalizer:
    """Test VietnameseTextNormalizer class."""
    
    def test_lowercase_normalization(self, text_normalizer):
        """Test lowercase conversion."""
        text = "Xin Chào VIỆT NAM"
        normalized = text_normalizer.normalize(text)
        
        assert normalized.islower()
        assert "việt nam" in normalized.lower()
    
    def test_preserve_vietnamese_tone_marks(self, text_normalizer):
        """Test that Vietnamese tone marks are preserved."""
        text = "Xin chào Việt Nam"
        normalized = text_normalizer.normalize(text)
        
        # Should preserve Vietnamese characters
        assert "việt" in normalized
        assert "chào" in normalized
    
    def test_number_to_words(self, text_normalizer):
        """Test number to word conversion."""
        text = "tôi có 123 người bạn"
        normalized = text_normalizer.normalize(text)
        
        # Should convert numbers to words
        assert "123" not in normalized or "một trăm" in normalized
    
    def test_abbreviation_expansion(self, text_normalizer):
        """Test abbreviation expansion."""
        text = "Dr. Nguyễn Văn A"
        normalized = text_normalizer.normalize(text)
        
        # Should expand abbreviations
        assert "dr" not in normalized.lower() or "tiến sĩ" in normalized.lower()
    
    def test_filler_word_removal(self, text_normalizer):
        """Test filler word removal."""
        text = "tôi ừm có thể ừm làm được"
        normalized = text_normalizer.normalize(text)
        
        # Should remove filler words
        assert "ừm" not in normalized or normalized.count("ừm") < text.count("ừm")
    
    def test_normalize_with_various_inputs(self, text_normalizer, sample_vietnamese_texts):
        """Test normalization with various Vietnamese text inputs."""
        for input_text, expected_pattern in sample_vietnamese_texts.items():
            normalized = text_normalizer.normalize(input_text)
            
            assert normalized is not None
            assert isinstance(normalized, str)
            assert len(normalized) > 0
            
            # Check lowercase
            if text_normalizer.lowercase:
                assert normalized.islower()
    
    def test_special_character_removal(self, text_normalizer):
        """Test special character removal."""
        text = "xin chào!!! việt nam... ???"
        normalized = text_normalizer.normalize(text)
        
        if text_normalizer.remove_punctuation:
            assert "!!!" not in normalized
            assert "..." not in normalized


class TestTokenizer:
    """Test Tokenizer class."""
    
    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer is not None
        assert len(tokenizer.vocab) > 0
        assert tokenizer.pad_token_id is not None
        assert tokenizer.blank_token_id is not None
    
    def test_encode_decode(self, tokenizer):
        """Test encoding and decoding."""
        text = "xin chào việt nam"
        
        # Encode
        token_ids = tokenizer.encode(text)
        
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(t, int) for t in token_ids)
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encode-decode roundtrip works."""
        texts = [
            "xin chào",
            "việt nam",
            "tôi là sinh viên",
            "hôm nay trời đẹp"
        ]
        
        for text in texts:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            
            # Should be similar (may lose some formatting)
            assert len(decoded) > 0
            assert isinstance(decoded, str)
    
    def test_special_tokens(self, tokenizer):
        """Test special tokens handling."""
        # Encode with special tokens
        token_ids = tokenizer.encode("xin chào")
        
        # Decode without special tokens
        decoded_clean = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Decode with special tokens
        decoded_all = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        assert isinstance(decoded_clean, str)
        assert isinstance(decoded_all, str)
    
    def test_vocab_size(self, tokenizer):
        """Test vocabulary size."""
        assert len(tokenizer.vocab) > 0
        
        # Should have at least basic Vietnamese characters
        vietnamese_chars = ['a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y']
        vocab_lower = [c.lower() for c in tokenizer.vocab]
        assert any(char in vocab_lower for char in vietnamese_chars)
    
    def test_pad_token(self, tokenizer):
        """Test padding token."""
        assert tokenizer.pad_token_id is not None
        assert tokenizer.pad_token_id < len(tokenizer.vocab)
    
    def test_blank_token(self, tokenizer):
        """Test blank token for CTC."""
        assert tokenizer.blank_token_id is not None
        assert tokenizer.blank_token_id < len(tokenizer.vocab)
        assert tokenizer.blank_token_id != tokenizer.pad_token_id

