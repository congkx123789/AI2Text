"""
Tests for training pipeline components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from training.dataset import ASRDataset, collate_fn, create_data_loaders
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
import pandas as pd


class TestASRDataset:
    """Test ASRDataset class."""
    
    def test_dataset_initialization(self, audio_processor, tokenizer, sample_audio_file, temp_dir):
        """Test dataset initialization."""
        # Create a simple CSV with actual audio file
        audio_path, _ = sample_audio_file
        
        data = {
            'file_path': [audio_path],
            'transcript': ['xin chào việt nam']
        }
        df = pd.DataFrame(data)
        
        dataset = ASRDataset(
            data_df=df,
            audio_processor=audio_processor,
            tokenizer=tokenizer
        )
        
        assert len(dataset) == 1
    
    def test_dataset_getitem(self, audio_processor, tokenizer, sample_audio_file):
        """Test getting item from dataset."""
        audio_path, _ = sample_audio_file
        
        data = {
            'file_path': [audio_path],
            'transcript': ['xin chào']
        }
        df = pd.DataFrame(data)
        
        dataset = ASRDataset(
            data_df=df,
            audio_processor=audio_processor,
            tokenizer=tokenizer
        )
        
        item = dataset[0]
        
        assert 'audio_features' in item
        assert 'audio_length' in item
        assert 'text_tokens' in item
        assert 'text_length' in item
        assert 'transcript' in item
        
        assert isinstance(item['audio_features'], torch.Tensor)
        assert isinstance(item['text_tokens'], torch.Tensor)
        assert isinstance(item['transcript'], str)
    
    def test_dataset_with_augmentation(self, audio_processor, tokenizer, audio_augmenter, sample_audio_file):
        """Test dataset with augmentation enabled."""
        audio_path, _ = sample_audio_file
        
        data = {
            'file_path': [audio_path],
            'transcript': ['xin chào']
        }
        df = pd.DataFrame(data)
        
        dataset = ASRDataset(
            data_df=df,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            augmenter=audio_augmenter,
            apply_augmentation=True
        )
        
        item = dataset[0]
        assert item['audio_features'] is not None
    
    def test_dataset_max_lengths(self, audio_processor, tokenizer, sample_audio_file):
        """Test dataset with max length constraints."""
        audio_path, _ = sample_audio_file
        
        data = {
            'file_path': [audio_path],
            'transcript': ['xin chào việt nam']
        }
        df = pd.DataFrame(data)
        
        dataset = ASRDataset(
            data_df=df,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            max_audio_len=50,
            max_text_len=10
        )
        
        item = dataset[0]
        assert item['audio_length'] <= 50
        assert item['text_length'] <= 10


class TestCollateFunction:
    """Test collate function for DataLoader."""
    
    def test_collate_fn_basic(self, tokenizer):
        """Test basic collate function."""
        # Create dummy batch
        batch = [
            {
                'audio_features': torch.randn(100, 80),
                'audio_length': 100,
                'text_tokens': torch.tensor([1, 2, 3, 4]),
                'text_length': 4,
                'transcript': 'xin chào'
            },
            {
                'audio_features': torch.randn(80, 80),
                'audio_length': 80,
                'text_tokens': torch.tensor([1, 2, 3]),
                'text_length': 3,
                'transcript': 'việt nam'
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'audio_features' in collated
        assert 'audio_lengths' in collated
        assert 'text_tokens' in collated
        assert 'text_lengths' in collated
        assert 'transcripts' in collated
        
        # Check shapes
        assert collated['audio_features'].shape[0] == 2  # Batch size
        assert collated['audio_features'].shape[1] == 100  # Max length
        assert collated['text_tokens'].shape[0] == 2
        assert collated['text_tokens'].shape[1] == 4  # Max length
    
    def test_collate_fn_padding(self, tokenizer):
        """Test that collate function pads correctly."""
        batch = [
            {
                'audio_features': torch.randn(50, 80),
                'audio_length': 50,
                'text_tokens': torch.tensor([1, 2]),
                'text_length': 2,
                'transcript': 'ab'
            },
            {
                'audio_features': torch.randn(100, 80),
                'audio_length': 100,
                'text_tokens': torch.tensor([1, 2, 3, 4, 5]),
                'text_length': 5,
                'transcript': 'abcde'
            }
        ]
        
        collated = collate_fn(batch)
        
        # Should pad to max length
        assert collated['audio_features'].shape[1] == 100
        assert collated['text_tokens'].shape[1] == 5
        
        # First item should be padded with zeros
        assert torch.allclose(collated['audio_features'][0, 50:], torch.zeros(50, 80))
        assert torch.allclose(collated['text_tokens'][0, 2:], torch.zeros(3))


class TestTrainingComponents:
    """Test training-related components."""
    
    def test_dataset_normalization(self, audio_processor, tokenizer, sample_audio_file):
        """Test that transcripts are normalized."""
        audio_path, _ = sample_audio_file
        
        data = {
            'file_path': [audio_path],
            'transcript': ['Xin Chào VIỆT NAM']
        }
        df = pd.DataFrame(data)
        
        normalizer = VietnameseTextNormalizer()
        dataset = ASRDataset(
            data_df=df,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            normalizer=normalizer
        )
        
        item = dataset[0]
        # Should be normalized (lowercase)
        assert item['transcript'].islower()

