#!/usr/bin/env python3
"""Estimate training time based on dataset size and configuration."""
import json
import csv
from pathlib import Path

def count_samples(manifest_path):
    """Count number of samples in manifest (excluding header)."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)

def estimate_training_time(config_path):
    """Estimate training time based on config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    # Get dataset size
    manifest_path = cfg['manifest']
    num_samples = count_samples(manifest_path)
    
    # Training parameters
    batch_size = cfg['batch_size']
    gradient_accumulation = cfg.get('gradient_accumulation_steps', 1)
    epochs = cfg['epochs']
    num_workers = cfg.get('num_workers', 0)
    
    # Calculate batches per epoch
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    
    # Estimate time per batch (seconds)
    # Conservative estimates based on typical ASR training:
    # - With GPU: ~0.1-0.5s per batch depending on audio length
    # - With 8 workers and batch_size=32: ~0.2-0.4s per batch
    # We'll use 0.3s as average (can be adjusted based on actual performance)
    time_per_batch = 0.3
    
    # Total time
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * time_per_batch
    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_secs = int(total_seconds % 60)
    
    print("=" * 60)
    print("ESTIMATED TRAINING TIME")
    print("=" * 60)
    print(f"Dataset size:        {num_samples:,} samples")
    print(f"Batch size:          {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation}")
    print(f"Effective batch size: {batch_size * gradient_accumulation}")
    print(f"Epochs:              {epochs}")
    print(f"Workers:             {num_workers}")
    print(f"Batches per epoch:   {batches_per_epoch:,}")
    print(f"Total batches:       {total_batches:,}")
    print("-" * 60)
    print(f"Estimated time/batch: {time_per_batch:.2f}s")
    if total_hours > 0:
        print(f"Total estimated time: {total_hours}h {total_minutes}m {total_secs}s ({total_seconds/3600:.2f} hours)")
    else:
        print(f"Total estimated time: {total_minutes}m {total_secs}s ({total_seconds/60:.2f} minutes)")
    print("=" * 60)
    print("\nNote: Actual time may vary based on:")
    print("  - Audio length and complexity")
    print("  - GPU performance")
    print("  - System load")
    print("  - I/O speed")
    print("\nTo get more accurate estimate, monitor first few batches!")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_merged.json"
    estimate_training_time(config_path)

