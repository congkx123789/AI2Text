#!/usr/bin/env python3
"""
Script to extract and display training metrics from log file.
Shows loss, learning rate, WER, CER from the latest training progress.
"""

import re
import sys
from pathlib import Path

def extract_metrics_from_log(log_file):
    """Extract metrics from training log file."""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Read last 500 lines to find recent metrics
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        recent_lines = lines[-500:] if len(lines) > 500 else lines
    
    # Patterns to match
    patterns = {
        'epoch_summary': re.compile(r'📊 EPOCH (\d+)/(\d+) SUMMARY'),
        'train_loss': re.compile(r'🎯 Train Loss:\s+([0-9.e-]+)'),
        'val_loss': re.compile(r'✅ Val Loss:\s+([0-9.e-]+)'),
        'learning_rate': re.compile(r'📈 Learning Rate:\s+([0-9.e-]+)'),
        'wer': re.compile(r'📝 WER:\s+([0-9.e-]+)'),
        'cer': re.compile(r'📝 CER:\s+([0-9.e-]+)'),
        'best_val_loss': re.compile(r'🏆 Best Val Loss:\s+([0-9.e-]+)'),
        'best_wer': re.compile(r'🏆 Best WER:\s+([0-9.e-]+)'),
        'progress_bar': re.compile(r'🚀 Epoch (\d+)/(\d+):\s+(\d+)%\|\s+\| (\d+)/(\d+) \[([^\]]+)\]'),
        'progress_metrics': re.compile(r"('loss':\s*'([0-9.e-]+)'|'avg_loss':\s*'([0-9.e-]+)'|'lr':\s*'([0-9.e-]+)'|'best_val':\s*'([0-9.e-]+|N/A)'|'best_wer':\s*'([0-9.e-]+|N/A)')"),
    }
    
    # Try to extract from progress bar postfix (if logged)
    # Look for lines with metrics in postfix format
    metrics = {
        'current_epoch': None,
        'current_batch': None,
        'total_batches': None,
        'progress_pct': None,
        'loss': None,
        'avg_loss': None,
        'lr': None,
        'best_val_loss': None,
        'best_wer': None,
        'wer': None,
        'cer': None,
        'time_elapsed': None,
        'time_remaining': None,
    }
    
    # Extract from epoch summary
    for line in reversed(recent_lines):
        # Epoch summary
        match = patterns['epoch_summary'].search(line)
        if match and not metrics['current_epoch']:
            metrics['current_epoch'] = int(match.group(1))
            metrics['total_epochs'] = int(match.group(2))
        
        # Train/Val Loss
        match = patterns['train_loss'].search(line)
        if match:
            metrics['train_loss'] = float(match.group(1))
        
        match = patterns['val_loss'].search(line)
        if match:
            metrics['val_loss'] = float(match.group(1))
        
        # Learning Rate
        match = patterns['learning_rate'].search(line)
        if match:
            metrics['lr'] = float(match.group(1))
        
        # WER/CER
        match = patterns['wer'].search(line)
        if match:
            metrics['wer'] = float(match.group(1))
        
        match = patterns['cer'].search(line)
        if match:
            metrics['cer'] = float(match.group(1))
        
        # Best metrics
        match = patterns['best_val_loss'].search(line)
        if match:
            metrics['best_val_loss'] = float(match.group(1))
        
        match = patterns['best_wer'].search(line)
        if match:
            metrics['best_wer'] = float(match.group(1))
    
    # Extract from latest progress bar
    for line in reversed(recent_lines):
        # Progress bar with metrics in postfix
        if '🚀 Epoch' in line:
            # Extract epoch and batch info
            match = patterns['progress_bar'].search(line)
            if match:
                metrics['current_epoch'] = int(match.group(1))
                metrics['total_epochs'] = int(match.group(2))
                metrics['progress_pct'] = int(match.group(3))
                metrics['current_batch'] = int(match.group(4))
                metrics['total_batches'] = int(match.group(5))
                time_info = match.group(6)
                # Parse time info: "07:08<30:29, 1.35batch/s"
                time_match = re.search(r'(\d+:\d+)<(\d+:\d+)', time_info)
                if time_match:
                    metrics['time_elapsed'] = time_match.group(1)
                    metrics['time_remaining'] = time_match.group(2)
            
            # Try to extract metrics from postfix (if they appear in log)
            # Look for patterns like 'loss': '4.1234' or loss=4.1234
            loss_match = re.search(r"loss['\"]?\s*[:=]\s*['\"]?([0-9.e-]+)", line, re.IGNORECASE)
            if loss_match and not metrics['loss']:
                metrics['loss'] = float(loss_match.group(1))
            
            avg_loss_match = re.search(r"avg_loss['\"]?\s*[:=]\s*['\"]?([0-9.e-]+)", line, re.IGNORECASE)
            if avg_loss_match and not metrics['avg_loss']:
                metrics['avg_loss'] = float(avg_loss_match.group(1))
            
            lr_match = re.search(r"lr['\"]?\s*[:=]\s*['\"]?([0-9.e-]+)", line, re.IGNORECASE)
            if lr_match and not metrics['lr']:
                metrics['lr'] = float(lr_match.group(1))
            
            break
    
    return metrics

def display_metrics(metrics):
    """Display metrics in a nice format."""
    print("=" * 70)
    print("📊 TRAINING METRICS - REAL-TIME STATUS")
    print("=" * 70)
    
    # Current progress
    if metrics.get('current_epoch') is not None:
        epoch_info = f"Epoch {metrics['current_epoch']}"
        if metrics.get('total_epochs'):
            epoch_info += f"/{metrics['total_epochs']}"
        print(f"🔄 {epoch_info}", end="")
        
        if metrics.get('current_batch') and metrics.get('total_batches'):
            batch_info = f" | Batch {metrics['current_batch']}/{metrics['total_batches']}"
            print(batch_info, end="")
            if metrics.get('progress_pct'):
                print(f" ({metrics['progress_pct']}%)", end="")
        print()
        
        if metrics.get('time_elapsed') and metrics.get('time_remaining'):
            print(f"⏱️  Time: {metrics['time_elapsed']} elapsed | {metrics['time_remaining']} remaining")
    
    print("-" * 70)
    
    # Loss metrics
    print("📉 LOSS METRICS:")
    if metrics.get('loss'):
        print(f"   Current Loss:     {metrics['loss']:.6f}")
    if metrics.get('avg_loss'):
        print(f"   Average Loss:     {metrics['avg_loss']:.6f}")
    if metrics.get('train_loss'):
        print(f"   Train Loss:       {metrics['train_loss']:.6f}")
    if metrics.get('val_loss'):
        print(f"   Val Loss:         {metrics['val_loss']:.6f}")
    if metrics.get('best_val_loss'):
        print(f"   🏆 Best Val Loss:  {metrics['best_val_loss']:.6f}")
    
    print("-" * 70)
    
    # Learning rate
    print("📈 LEARNING RATE:")
    if metrics.get('lr'):
        print(f"   Current LR:       {metrics['lr']:.2e}")
    else:
        print("   Current LR:       N/A (checking log...)")
    
    print("-" * 70)
    
    # WER/CER metrics
    print("📝 ACCURACY METRICS:")
    if metrics.get('wer') is not None:
        print(f"   WER:              {metrics['wer']:.4f}")
    else:
        print("   WER:              N/A (not calculated yet)")
    
    if metrics.get('cer') is not None:
        print(f"   CER:              {metrics['cer']:.4f}")
    else:
        print("   CER:              N/A (not calculated yet)")
    
    if metrics.get('best_wer'):
        print(f"   🏆 Best WER:       {metrics['best_wer']:.4f}")
    
    print("=" * 70)

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/training_restart.log'
    metrics = extract_metrics_from_log(log_file)
    display_metrics(metrics)

if __name__ == '__main__':
    main()

