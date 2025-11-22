"""
Pytest configuration and fixtures for ASR tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
import sqlite3

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import VietnameseTextNormalizer, Tokenizer
from models.asr_base import ASRModel
from utils.metrics import calculate_wer, calculate_cer, calculate_accuracy


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    db = ASRDatabase(db_path)
    yield db
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    # Generate 1 second of audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    audio_path = temp_dir / "test_audio.wav"
    sf.write(str(audio_path), audio, sample_rate)
    return str(audio_path), sample_rate


@pytest.fixture
def sample_audio_file_stereo(temp_dir):
    """Create a sample stereo audio file for testing."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    stereo_audio = np.stack([audio, audio])  # 2 channels
    
    audio_path = temp_dir / "test_audio_stereo.wav"
    sf.write(str(audio_path), stereo_audio.T, sample_rate)
    return str(audio_path), sample_rate


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_data = {
        'file_path': [
            'audio1.wav',
            'audio2.wav',
            'audio3.wav'
        ],
        'transcript': [
            'xin chào việt nam',
            'tôi là sinh viên',
            'hôm nay trời đẹp'
        ],
        'split': ['train', 'train', 'val'],
        'speaker_id': ['speaker_01', 'speaker_02', 'speaker_01']
    }
    
    csv_path = temp_dir / "test_data.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return str(csv_path)


@pytest.fixture
def audio_processor():
    """Create an AudioProcessor instance."""
    return AudioProcessor(sample_rate=16000, n_mels=80)


@pytest.fixture
def audio_augmenter():
    """Create an AudioAugmenter instance."""
    return AudioAugmenter(sample_rate=16000)


@pytest.fixture
def text_normalizer():
    """Create a VietnameseTextNormalizer instance."""
    return VietnameseTextNormalizer()


@pytest.fixture
def tokenizer():
    """Create a Tokenizer instance."""
    return Tokenizer()


@pytest.fixture
def asr_model():
    """Create an ASR model for testing."""
    return ASRModel(
        input_dim=80,
        vocab_size=100,
        d_model=128,
        num_encoder_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1
    )


@pytest.fixture
def sample_transcripts():
    """Sample Vietnamese transcripts for testing."""
    return [
        "xin chào việt nam",
        "tôi là sinh viên",
        "hôm nay trời đẹp",
        "tiếng việt rất hay",
        "chúc bạn một ngày tốt lành"
    ]


@pytest.fixture
def sample_vietnamese_texts():
    """Sample Vietnamese texts for normalization testing."""
    return {
        "Xin chào VIỆT NAM": "xin chào việt nam",
        "tôi có 123 người bạn": "tôi có một trăm hai mươi ba người bạn",
        "Hôm nay là 25/12/2023": "hôm nay là hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi ba",
        "Dr. Nguyễn Văn A": "tiến sĩ nguyễn văn a",
        "Tôi... ừm... có thể...": "tôi có thể",
    }

