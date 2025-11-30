"""
Tests for ASR model architecture.
"""

import pytest
import torch
import numpy as np

from models.asr_base import ASRModel


class TestASRModel:
    """Test ASR model architecture."""
    
    def test_model_initialization(self, asr_model):
        """Test model initialization."""
        assert asr_model is not None
        assert hasattr(asr_model, 'encoder')
        assert hasattr(asr_model, 'decoder')
    
    def test_model_forward(self, asr_model):
        """Test model forward pass."""
        batch_size = 2
        seq_length = 100
        input_dim = 80
        
        # Create dummy input (batch, time, features)
        x = torch.randn(batch_size, seq_length, input_dim)
        lengths = torch.tensor([seq_length, seq_length - 20])
        
        # Forward pass
        logits, output_lengths = asr_model(x, lengths)
        
        assert logits is not None
        assert isinstance(logits, torch.Tensor)
        assert logits.shape[0] == batch_size  # Batch dimension
        assert logits.shape[1] == seq_length or logits.shape[1] <= seq_length  # Time dimension
        assert logits.shape[2] == asr_model.vocab_size  # Vocabulary dimension
        
        assert output_lengths is not None
        assert len(output_lengths) == batch_size
    
    def test_model_parameters(self, asr_model):
        """Test model parameter counting."""
        num_params = asr_model.get_num_params()
        num_trainable = asr_model.get_num_trainable_params()
        
        assert num_params > 0
        assert num_trainable > 0
        assert num_trainable <= num_params
        
        # Should have reasonable number of parameters
        assert num_params > 1000  # At least some parameters
        assert num_params < 100000000  # Not unreasonably large
    
    def test_model_gradient_flow(self, asr_model):
        """Test that gradients can flow through model."""
        batch_size = 1
        seq_length = 50
        input_dim = 80
        
        x = torch.randn(batch_size, seq_length, input_dim, requires_grad=True)
        lengths = torch.tensor([seq_length])
        
        logits, _ = asr_model(x, lengths)
        
        # Compute loss (dummy)
        loss = logits.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
    
    def test_model_different_batch_sizes(self, asr_model):
        """Test model with different batch sizes."""
        input_dim = 80
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 100, input_dim)
            lengths = torch.tensor([100] * batch_size)
            
            logits, output_lengths = asr_model(x, lengths)
            
            assert logits.shape[0] == batch_size
            assert len(output_lengths) == batch_size
    
    def test_model_different_sequence_lengths(self, asr_model):
        """Test model with different sequence lengths."""
        batch_size = 2
        input_dim = 80
        
        seq_lengths = [50, 100, 150]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, input_dim)
            lengths = torch.tensor([seq_len, seq_len - 10])
            
            logits, output_lengths = asr_model(x, lengths)
            
            assert logits is not None
            assert output_lengths is not None
    
    def test_model_eval_mode(self, asr_model):
        """Test model in eval mode."""
        asr_model.eval()
        
        x = torch.randn(1, 50, 80)
        lengths = torch.tensor([50])
        
        with torch.no_grad():
            logits, _ = asr_model(x, lengths)
        
        assert logits is not None
        assert not asr_model.training
    
    def test_model_train_mode(self, asr_model):
        """Test model in train mode."""
        asr_model.train()
        
        assert asr_model.training
        
        x = torch.randn(1, 50, 80)
        lengths = torch.tensor([50])
        
        logits, _ = asr_model(x, lengths)
        
        assert logits is not None
    
    def test_model_device_movement(self, asr_model):
        """Test moving model to different devices."""
        # CPU
        asr_model_cpu = asr_model
        x_cpu = torch.randn(1, 50, 80)
        lengths_cpu = torch.tensor([50])
        
        logits_cpu, _ = asr_model_cpu(x_cpu, lengths_cpu)
        assert logits_cpu.device.type == 'cpu'
        
        # If CUDA available, test GPU
        if torch.cuda.is_available():
            asr_model_gpu = asr_model.cuda()
            x_gpu = x_cpu.cuda()
            lengths_gpu = lengths_cpu.cuda()
            
            logits_gpu, _ = asr_model_gpu(x_gpu, lengths_gpu)
            assert logits_gpu.device.type == 'cuda'

