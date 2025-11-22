"""
Performance benchmarking script for ASR models.

Measures and compares performance metrics across different model architectures.
"""

import argparse
import torch
import time
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from models.lstm_asr import LSTMASRModel
from models.enhanced_asr import EnhancedASRModel
from preprocessing.text_cleaning import Tokenizer
from decoding.beam_search import BeamSearchDecoder
from utils.metrics import calculate_wer, calculate_cer


class Benchmark:
    """Performance benchmarker for ASR models."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize benchmarker.
        
        Args:
            device: Device to run benchmarks on
        """
        self.device = torch.device(device)
        self.results = []
    
    def benchmark_model(self,
                       model: torch.nn.Module,
                       model_name: str,
                       input_shape: tuple,
                       num_warmup: int = 10,
                       num_runs: int = 100) -> Dict:
        """
        Benchmark a single model.
        
        Args:
            model: Model to benchmark
            model_name: Name of model
            input_shape: Input shape (batch, time, features)
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        model.eval()
        model.to(self.device)
        
        batch_size, seq_len, features = input_shape
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, features).to(self.device)
        dummy_lengths = torch.tensor([seq_len] * batch_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input, dummy_lengths)
        
        # Synchronize CUDA if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input, dummy_lengths)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs
        
        # Memory usage
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Calculate throughput
        samples_per_sec = batch_size / avg_time
        
        # Model size
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        result = {
            'model': model_name,
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'features': features,
            'device': str(self.device),
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': samples_per_sec,
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved
        }
        
        self.results.append(result)
        return result
    
    def benchmark_decoding(self,
                          logits: torch.Tensor,
                          decoder: BeamSearchDecoder,
                          beam_width: int,
                          num_runs: int = 100) -> Dict:
        """Benchmark decoding performance."""
        decoder_name = f"beam_search_w{beam_width}"
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = decoder.decode(logits)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs
        
        return {
            'decoder': decoder_name,
            'beam_width': beam_width,
            'avg_decode_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': logits.size(0) / avg_time
        }
    
    def save_results(self, output_path: str):
        """Save benchmark results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[OK] Results saved to {output_path}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("Benchmark Results Summary")
        print("="*70)
        
        for result in self.results:
            print(f"\n{result['model']}:")
            print(f"  Parameters: {result['num_parameters']:,}")
            print(f"  Inference time: {result['avg_inference_time_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput_samples_per_sec']:.2f} samples/sec")
            if result['memory_allocated_gb'] > 0:
                print(f"  GPU Memory: {result['memory_allocated_gb']:.2f} GB")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark ASR models')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--features', type=int, default=80, help='Feature dimension')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cuda, cpu)')
    parser.add_argument('--output', type=str, default='benchmarks/results.json',
                      help='Output JSON file')
    parser.add_argument('--models', nargs='+', 
                      choices=['lstm', 'transformer', 'enhanced', 'all'],
                      default=['all'],
                      help='Models to benchmark')
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("ASR Model Benchmarking")
    print("="*70)
    print(f"Device: {device}")
    print(f"Input shape: ({args.batch_size}, {args.seq_len}, {args.features})")
    print()
    
    benchmark = Benchmark(device=device)
    
    input_shape = (args.batch_size, args.seq_len, args.features)
    
    # LSTM model
    if 'all' in args.models or 'lstm' in args.models:
        print("Benchmarking LSTM model...")
        lstm_model = LSTMASRModel(
            input_dim=args.features,
            vocab_size=args.vocab_size,
            hidden_size=256
        )
        benchmark.benchmark_model(lstm_model, "LSTM", input_shape)
    
    # Transformer model
    if 'all' in args.models or 'transformer' in args.models:
        print("Benchmarking Transformer model...")
        transformer_model = ASRModel(
            input_dim=args.features,
            vocab_size=args.vocab_size,
            d_model=256
        )
        benchmark.benchmark_model(transformer_model, "Transformer", input_shape)
    
    # Enhanced model
    if 'all' in args.models or 'enhanced' in args.models:
        print("Benchmarking Enhanced model...")
        enhanced_model = EnhancedASRModel(
            input_dim=args.features,
            vocab_size=args.vocab_size,
            d_model=256,
            use_contextual_embeddings=True
        )
        benchmark.benchmark_model(enhanced_model, "Enhanced", input_shape)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.save_results(str(output_path))


if __name__ == "__main__":
    main()

