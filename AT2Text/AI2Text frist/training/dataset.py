"""
Dataset classes for ASR training.
Handles loading audio and text data efficiently.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import VietnameseTextNormalizer, Tokenizer


class ASRDataset(Dataset):
    """Dataset for ASR training."""
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 audio_processor: AudioProcessor,
                 tokenizer: Tokenizer,
                 normalizer: Optional[VietnameseTextNormalizer] = None,
                 augmenter: Optional[AudioAugmenter] = None,
                 max_audio_len: Optional[int] = None,
                 max_text_len: Optional[int] = None,
                 apply_augmentation: bool = False):
        """Initialize ASR dataset.
        
        Args:
            data_df: DataFrame with columns ['file_path', 'transcript']
            audio_processor: Audio processor instance
            tokenizer: Text tokenizer instance
            normalizer: Text normalizer instance
            augmenter: Audio augmenter instance
            max_audio_len: Maximum audio length in samples
            max_text_len: Maximum text length in tokens
            apply_augmentation: Whether to apply augmentation
        """
        self.data_df = data_df.reset_index(drop=True)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.normalizer = normalizer or VietnameseTextNormalizer()
        self.augmenter = augmenter
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.apply_augmentation = apply_augmentation
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset.
        
        Returns:
            item: Dictionary with 'audio_features', 'audio_length', 
                  'text_tokens', 'text_length'
        """
        row = self.data_df.iloc[idx]
        
        # Load and process audio
        audio, sr = self.audio_processor.load_audio(row['file_path'])
        
        # Apply augmentation if enabled (only during training)
        if self.apply_augmentation and self.augmenter:
            audio = self.augmenter.augment(audio)
        
        # Trim silence
        audio = self.audio_processor.trim_silence(audio)
        
        # Extract features
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        
        # Transpose to (time, freq) for model input
        mel_spec = mel_spec.T
        
        # Normalize and tokenize text
        transcript = row['transcript']
        if 'normalized_transcript' in row and pd.notna(row['normalized_transcript']):
            normalized_text = row['normalized_transcript']
        else:
            normalized_text = self.normalizer.normalize(transcript)
        
        text_tokens = self.tokenizer.encode(normalized_text)
        
        # Convert to tensors
        audio_features = torch.from_numpy(mel_spec).float()
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        # Get lengths
        audio_length = audio_features.size(0)
        text_length = text_tokens.size(0)
        
        # Truncate if needed
        if self.max_audio_len and audio_length > self.max_audio_len:
            audio_features = audio_features[:self.max_audio_len]
            audio_length = self.max_audio_len
        
        if self.max_text_len and text_length > self.max_text_len:
            text_tokens = text_tokens[:self.max_text_len]
            text_length = self.max_text_len
        
        return {
            'audio_features': audio_features,
            'audio_length': audio_length,
            'text_tokens': text_tokens,
            'text_length': text_length,
            'transcript': normalized_text
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with padding.
    
    Args:
        batch: List of dataset items
        
    Returns:
        collated_batch: Dictionary with padded tensors
    """
    # Find max lengths in batch
    max_audio_len = max(item['audio_length'] for item in batch)
    max_text_len = max(item['text_length'] for item in batch)
    
    # Get feature dimension
    freq_dim = batch[0]['audio_features'].size(1)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    audio_features = torch.zeros(batch_size, max_audio_len, freq_dim)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    transcripts = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        audio_len = item['audio_length']
        text_len = item['text_length']
        
        audio_features[i, :audio_len] = item['audio_features']
        text_tokens[i, :text_len] = item['text_tokens']
        audio_lengths[i] = audio_len
        text_lengths[i] = text_len
        transcripts.append(item['transcript'])
    
    return {
        'audio_features': audio_features,
        'audio_lengths': audio_lengths,
        'text_tokens': text_tokens,
        'text_lengths': text_lengths,
        'transcripts': transcripts
    }


def create_data_loaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       audio_processor: AudioProcessor,
                       tokenizer: Tokenizer,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       augmenter: Optional[AudioAugmenter] = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        audio_processor: Audio processor instance
        tokenizer: Tokenizer instance
        batch_size: Batch size
        num_workers: Number of worker processes
        augmenter: Optional audio augmenter for training
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    normalizer = VietnameseTextNormalizer()
    
    # Create datasets
    train_dataset = ASRDataset(
        data_df=train_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        augmenter=augmenter,
        apply_augmentation=True
    )
    
    val_dataset = ASRDataset(
        data_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        apply_augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from preprocessing.audio_processing import AudioProcessor
    from preprocessing.text_cleaning import Tokenizer
    
    # Create dummy data
    data = {
        'file_path': ['dummy1.wav', 'dummy2.wav'],
        'transcript': ['xin chào', 'tạm biệt']
    }
    df = pd.DataFrame(data)
    
    processor = AudioProcessor()
    tokenizer = Tokenizer()
    
    print("Dataset test complete!")

