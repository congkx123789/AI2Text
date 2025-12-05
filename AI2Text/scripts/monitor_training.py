#!/usr/bin/env python3
"""
Real-time training progress monitor.
Shows progress bars and metrics in terminal.
"""

import re
import time
import sys
from pathlib import Path

def get_progress(log_file):
    """Extract progress from log file."""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_epoch = 1
    current_batch = 0
    total_batches = 1780
    latest_loss = 0.0
    last_time = None
    
    for line in reversed(lines[-1000:]):
        epoch_match = re.search(r'Epoch (\d+)/150', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        batch_match = re.search(r'Batch (\d+)/(\d+)', line)
        if batch_match:
            current_batch = int(batch_match.group(1))
            total_batches = int(batch_match.group(2))
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            if loss_match:
                latest_loss = float(loss_match.group(1))
            time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if time_match:
                last_time = time_match.group(1)
            break
    
    return {
        'epoch': current_epoch,
        'batch': current_batch,
        'total_batches': total_batches,
        'loss': latest_loss,
        'last_time': last_time
    }

def print_progress(progress):
    """Print formatted progress."""
    if not progress:
        print("❌ Log file not found or no progress data")
        return
    
    epoch = progress['epoch']
    batch = progress['batch']
    total = progress['total_batches']
    loss = progress['loss']
    last_time = progress['last_time']
    
    # Calculate percentages
    epoch_progress = (batch / total) * 100 if total > 0 else 0
    overall_progress = ((epoch - 1) / 150) * 100 + (epoch_progress / 150)
    
    # Create progress bars
    bar_length = 50
    epoch_filled = int(bar_length * epoch_progress / 100)
    epoch_bar = '█' * epoch_filled + '░' * (bar_length - epoch_filled)
    
    overall_filled = int(bar_length * overall_progress / 100)
    overall_bar = '█' * overall_filled + '░' * (bar_length - overall_filled)
    
    # Clear screen and print
    print("\033[2J\033[H", end='')  # Clear screen
    print("=" * 70)
    print("🚀 ASR TRAINING PROGRESS - REAL-TIME MONITOR")
    print("=" * 70)
    print(f"📅 Last Update: {last_time or 'N/A'}")
    print(f"📊 Epoch: {epoch}/150")
    print(f"📦 Batch: {batch}/{total}")
    print(f"📉 Loss: {loss:.4f}")
    print()
    print(f"Epoch Progress:  [{epoch_bar}] {epoch_progress:.2f}%")
    print()
    print(f"Overall Progress: [{overall_bar}] {overall_progress:.2f}%")
    print()
    print(f"⏱️  Remaining: {100 - overall_progress:.2f}%")
    print("=" * 70)
    print("\nPress Ctrl+C to exit")

def main():
    log_file = Path(__file__).parent.parent / "logs" / "training.log"
    
    print("Starting training monitor...")
    print("Watching:", log_file)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            progress = get_progress(log_file)
            print_progress(progress)
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")

if __name__ == "__main__":
    main()















