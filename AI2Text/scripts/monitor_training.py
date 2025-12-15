#!/usr/bin/env python3
"""
Real-time training progress monitor với giao diện đẹp.
Hiển thị progress bars, metrics, và thống kê training.
"""

import re
import time
import sys
from pathlib import Path
from datetime import datetime
import os

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_training_progress(log_file):
    """Extract training progress from log file.
    
    Returns:
        dict: Progress information
    """
    if not Path(log_file).exists():
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return None
    
    # Default values
    progress = {
        'epoch': 0,
        'total_epochs': 50,
        'batch': 0,
        'total_batches': 0,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'wer': None,
        'cer': None,
        'learning_rate': 0.0,
        'best_val_loss': float('inf'),
        'best_wer': float('inf'),
        'last_update': None,
        'status': 'Unknown'
    }
    
    # Parse log file (read from end)
    for line in reversed(lines[-2000:]):  # Check last 2000 lines
        # Epoch info
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match and progress['epoch'] == 0:
            progress['epoch'] = int(epoch_match.group(1))
            progress['total_epochs'] = int(epoch_match.group(2))
        
        # Batch info
        batch_match = re.search(r'Batch (\d+)/(\d+)', line)
        if batch_match:
            progress['batch'] = int(batch_match.group(1))
            progress['total_batches'] = int(batch_match.group(2))
        
        # Loss values
        train_loss_match = re.search(r'Train Loss:\s+([\d.]+)', line)
        if train_loss_match:
            progress['train_loss'] = float(train_loss_match.group(1))
        
        val_loss_match = re.search(r'Val Loss:\s+([\d.]+)', line)
        if val_loss_match:
            progress['val_loss'] = float(val_loss_match.group(1))
        
        # WER/CER
        wer_match = re.search(r'WER:\s+([\d.]+)', line)
        if wer_match:
            progress['wer'] = float(wer_match.group(1))
        
        cer_match = re.search(r'CER:\s+([\d.]+)', line)
        if cer_match:
            progress['cer'] = float(cer_match.group(1))
        
        # Learning rate - multiple formats
        lr_match = re.search(r'Learning Rate:\s+([\d.e+-]+)', line)
        if not lr_match:
            # Try alternative format: lr: 3.0000e-04
            lr_match = re.search(r"['\"]lr['\"]:\s*['\"]?([\d.e+-]+)", line)
        if not lr_match:
            # Try format: lr=3.0000e-04
            lr_match = re.search(r"lr[=:]\s*([\d.e+-]+)", line)
        if lr_match:
            try:
                progress['learning_rate'] = float(lr_match.group(1))
            except:
                pass
        
        # Best metrics
        best_val_match = re.search(r'Best Val Loss:\s+([\d.]+)', line)
        if best_val_match:
            progress['best_val_loss'] = float(best_val_match.group(1))
        
        best_wer_match = re.search(r'Best WER:\s+([\d.]+)', line)
        if best_wer_match:
            progress['best_wer'] = float(best_wer_match.group(1))
        
        # Timestamp
        time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if time_match and not progress['last_update']:
            progress['last_update'] = time_match.group(1)
        
        # Status
        if 'Training completed' in line:
            progress['status'] = 'Completed'
        elif 'Training interrupted' in line:
            progress['status'] = 'Interrupted'
        elif 'Epoch' in line and 'Summary' in line:
            progress['status'] = 'Training'
    
    # Calculate progress percentages
    if progress['total_batches'] > 0:
        epoch_progress = (progress['batch'] / progress['total_batches']) * 100
    else:
        epoch_progress = 0.0
    
    if progress['total_epochs'] > 0:
        overall_progress = ((progress['epoch'] - 1) / progress['total_epochs']) * 100 + (epoch_progress / progress['total_epochs'])
    else:
        overall_progress = 0.0
    
    progress['epoch_progress'] = epoch_progress
    progress['overall_progress'] = overall_progress
    
    return progress

def create_progress_bar(filled, total, length=50, filled_char='█', empty_char='░'):
    """Create a progress bar string.
    
    Args:
        filled: Filled portion (0-100)
        total: Total (100)
        length: Bar length
        filled_char: Character for filled portion
        empty_char: Character for empty portion
    
    Returns:
        str: Progress bar
    """
    filled_len = int(length * filled / total) if total > 0 else 0
    empty_len = length - filled_len
    return filled_char * filled_len + empty_char * empty_len

def format_time(seconds):
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def print_monitor(progress):
    """Print beautiful training monitor.
    
    Args:
        progress: Progress dictionary
    """
    if not progress:
        print("❌ Log file not found or no progress data")
        print("   Make sure training is running and log file exists.")
        return
    
    # Clear screen
    clear_screen()
    
    # Header
    print("=" * 80)
    print("🚀 ASR TRAINING PROGRESS MONITOR - REAL-TIME")
    print("=" * 80)
    print()
    
    # Status and time
    status_emoji = {
        'Training': '🔄',
        'Completed': '✅',
        'Interrupted': '⚠️',
        'Unknown': '❓'
    }
    emoji = status_emoji.get(progress['status'], '❓')
    print(f"{emoji} Status: {progress['status']}")
    if progress['last_update']:
        print(f"📅 Last Update: {progress['last_update']}")
    print()
    
    # Epoch info
    print("📊 EPOCH PROGRESS")
    print("-" * 80)
    epoch_bar = create_progress_bar(
        progress['epoch_progress'], 100, length=60
    )
    print(f"Epoch: {progress['epoch']}/{progress['total_epochs']}")
    print(f"[{epoch_bar}] {progress['epoch_progress']:.2f}%")
    if progress['total_batches'] > 0:
        print(f"Batch: {progress['batch']}/{progress['total_batches']}")
    print()
    
    # Overall progress
    print("📈 OVERALL PROGRESS")
    print("-" * 80)
    overall_bar = create_progress_bar(
        progress['overall_progress'], 100, length=60
    )
    print(f"[{overall_bar}] {progress['overall_progress']:.2f}%")
    remaining = 100 - progress['overall_progress']
    print(f"⏱️  Remaining: {remaining:.2f}%")
    print()
    
    # Metrics
    print("📉 METRICS")
    print("-" * 80)
    print(f"  🎯 Train Loss:    {progress['train_loss']:.4f}")
    print(f"  ✅ Val Loss:      {progress['val_loss']:.4f}")
    if progress['wer'] is not None:
        print(f"  📝 WER:           {progress['wer']:.4f}")
    if progress['cer'] is not None:
        print(f"  📝 CER:           {progress['cer']:.4f}")
    if progress['learning_rate'] > 0:
        print(f"  📈 Learning Rate:  {progress['learning_rate']:.6e}")
    else:
        print(f"  📈 Learning Rate:  N/A (warmup phase)")
    print()
    
    # Best metrics
    print("🏆 BEST METRICS")
    print("-" * 80)
    if progress['best_val_loss'] < float('inf'):
        print(f"  Best Val Loss:    {progress['best_val_loss']:.6f}")
    else:
        print(f"  Best Val Loss:    N/A (not set yet)")
    if progress['best_wer'] is not None and progress['best_wer'] < float('inf'):
        print(f"  Best WER:         {progress['best_wer']:.6f}")
    else:
        print(f"  Best WER:         N/A (not calculated yet)")
    print()
    
    # Checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print("💾 CHECKPOINTS")
            print("-" * 80)
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for i, ckpt in enumerate(checkpoints[:3], 1):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
                print(f"  {i}. {ckpt.name} ({size_mb:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M')})")
            print()
    
    print("=" * 80)
    print("Press Ctrl+C to exit | Updates every 3 seconds")
    print("=" * 80)

def main():
    """Main monitoring loop."""
    # Get log file path
    script_dir = Path(__file__).parent.parent
    log_file = script_dir / "logs" / "training.log"
    
    print("🔍 Starting training monitor...")
    print(f"📂 Watching: {log_file}")
    print("Press Ctrl+C to exit\n")
    time.sleep(2)
    
    try:
        while True:
            progress = get_training_progress(log_file)
            print_monitor(progress)
            time.sleep(3)  # Update every 3 seconds
    except KeyboardInterrupt:
        clear_screen()
        print("\n\n✅ Monitor stopped.")
        print("Training is still running in the background.")
        print("Run this script again to resume monitoring.\n")

if __name__ == "__main__":
    main()
