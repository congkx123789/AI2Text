"""
Tests for database utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd

from database.db_utils import ASRDatabase


class TestASRDatabase:
    """Test ASRDatabase class."""
    
    def test_database_initialization(self, temp_db):
        """Test database initialization."""
        assert temp_db.db_path.exists()
        assert temp_db.db_path.suffix == '.db'
    
    def test_add_audio_file(self, temp_db):
        """Test adding audio file to database."""
        audio_id = temp_db.add_audio_file(
            file_path="test/path/audio.wav",
            filename="audio.wav",
            duration=5.5,
            sample_rate=16000,
            channels=1,
            format="wav",
            language="vi",
            speaker_id="speaker_01",
            audio_quality="high"
        )
        
        assert audio_id is not None
        assert isinstance(audio_id, int)
        
        # Retrieve and verify
        audio_file = temp_db.get_audio_file(audio_id)
        assert audio_file is not None
        assert audio_file['file_path'] == "test/path/audio.wav"
        assert audio_file['duration_seconds'] == 5.5
        assert audio_file['sample_rate'] == 16000
        assert audio_file['speaker_id'] == "speaker_01"
    
    def test_add_audio_file_duplicate_skip(self, temp_db):
        """Test duplicate detection in add_audio_file."""
        audio_id1 = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000,
            skip_duplicate=True
        )
        
        # Try to add same file again
        audio_id2 = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000,
            skip_duplicate=True
        )
        
        assert audio_id1 is not None
        assert audio_id2 is None  # Should be skipped
    
    def test_add_transcript(self, temp_db):
        """Test adding transcript."""
        # Add audio file first
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        # Add transcript
        transcript_id = temp_db.add_transcript(
            audio_file_id=audio_id,
            transcript="xin chào việt nam",
            normalized_transcript="xin chào việt nam",
            annotation_type="manual"
        )
        
        assert transcript_id is not None
        
        # Verify
        with temp_db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM Transcripts WHERE id = ?", (transcript_id,)
            )
            transcript = cursor.fetchone()
            assert transcript is not None
            assert transcript['transcript'] == "xin chào việt nam"
            assert transcript['word_count'] == 3
    
    def test_assign_split(self, temp_db):
        """Test assigning data split."""
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        
        split_id = temp_db.assign_split(audio_id, "train", "v1")
        assert split_id is not None
        
        # Verify split assignment
        df = temp_db.get_split_data("train", "v1")
        assert len(df) == 1
        assert df.iloc[0]['audio_file_id'] == audio_id
    
    def test_get_split_data(self, temp_db):
        """Test retrieving split data."""
        # Add multiple files with different splits
        audio_ids = []
        for i in range(5):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/audio{i}.wav",
                filename=f"audio{i}.wav",
                duration=5.0,
                sample_rate=16000
            )
            audio_ids.append(audio_id)
            
            # Add transcript
            temp_db.add_transcript(
                audio_file_id=audio_id,
                transcript=f"test transcript {i}",
                normalized_transcript=f"test transcript {i}"
            )
            
            # Assign split
            split_type = "train" if i < 3 else "val"
            temp_db.assign_split(audio_id, split_type, "v1")
        
        # Get train split
        train_df = temp_db.get_split_data("train", "v1")
        assert len(train_df) == 3
        
        # Get val split
        val_df = temp_db.get_split_data("val", "v1")
        assert len(val_df) == 2
    
    def test_validate_data_for_training_empty(self, temp_db):
        """Test validation with empty database."""
        validation = temp_db.validate_data_for_training("v1")
        
        assert validation['is_ready'] == False
        assert "No training data found" in validation['issues'][0]
        assert len(validation['statistics']['splits']) == 0
    
    def test_validate_data_for_training_with_data(self, temp_db):
        """Test validation with proper data."""
        # Add training data
        for i in range(60):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/train{i}.wav",
                filename=f"train{i}.wav",
                duration=5.0,
                sample_rate=16000
            )
            temp_db.add_transcript(
                audio_file_id=audio_id,
                transcript=f"transcript {i}",
                normalized_transcript=f"transcript {i}"
            )
            temp_db.assign_split(audio_id, "train", "v1")
        
        # Add validation data
        for i in range(10):
            audio_id = temp_db.add_audio_file(
                file_path=f"test/val{i}.wav",
                filename=f"val{i}.wav",
                duration=5.0,
                sample_rate=16000
            )
            temp_db.add_transcript(
                audio_file_id=audio_id,
                transcript=f"transcript {i}",
                normalized_transcript=f"transcript {i}"
            )
            temp_db.assign_split(audio_id, "val", "v1")
        
        validation = temp_db.validate_data_for_training("v1")
        
        assert validation['is_ready'] == True
        assert validation['statistics']['splits']['train'] == 60
        assert validation['statistics']['splits']['val'] == 10
    
    def test_get_dataset_statistics(self, temp_db):
        """Test getting dataset statistics."""
        # Add some data
        for split_type in ['train', 'val', 'test']:
            for i in range(5):
                audio_id = temp_db.add_audio_file(
                    file_path=f"test/{split_type}{i}.wav",
                    filename=f"{split_type}{i}.wav",
                    duration=5.0 * (i + 1),
                    sample_rate=16000,
                    speaker_id=f"speaker_{i % 3}"
                )
                temp_db.add_transcript(
                    audio_file_id=audio_id,
                    transcript=f"transcript {i}",
                    normalized_transcript=f"transcript {i}"
                )
                temp_db.assign_split(audio_id, split_type, "v1")
        
        stats = temp_db.get_dataset_statistics("v1")
        
        assert len(stats) == 3
        assert all(stats['split_type'].values == ['test', 'train', 'val'])
        assert all(stats['num_files'].values == [5, 5, 5])
    
    def test_batch_add_audio_files(self, temp_db):
        """Test batch adding audio files."""
        audio_data = [
            {
                'file_path': f'test/audio{i}.wav',
                'filename': f'audio{i}.wav',
                'duration': 5.0,
                'sample_rate': 16000,
                'channels': 1,
                'format': 'wav',
                'language': 'vi',
                'speaker_id': f'speaker_{i % 2}',
                'audio_quality': 'medium'
            }
            for i in range(10)
        ]
        
        ids = temp_db.batch_add_audio_files(audio_data)
        
        assert len(ids) == 10
        assert all(id is not None for id in ids)
        
        # Verify files were added
        with temp_db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM AudioFiles")
            count = cursor.fetchone()['count']
            assert count == 10
    
    def test_get_data_summary(self, temp_db):
        """Test getting comprehensive data summary."""
        # Add minimal data
        audio_id = temp_db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=10.0,
            sample_rate=16000,
            speaker_id="speaker_01"
        )
        temp_db.add_transcript(
            audio_file_id=audio_id,
            transcript="xin chào việt nam",
            normalized_transcript="xin chào việt nam"
        )
        temp_db.assign_split(audio_id, "train", "v1")
        
        summary = temp_db.get_data_summary("v1")
        
        assert 'overall' in summary
        assert 'splits' in summary
        assert 'validation' in summary
        assert summary['overall']['total_files'] == 1
        assert summary['overall']['total_speakers'] == 1

