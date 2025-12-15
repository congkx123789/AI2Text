#!/usr/bin/env python3
"""
Simple script to show current training metrics from log file.
"""

import re
import sys
from pathlib import Path

def show_metrics(log_file):
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Read last 500 lines
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        recent = lines[-500:] if len(lines) > 500 else lines
    
    print("=" * 70)
    print("📊 TRAINING METRICS - CURRENT STATUS")
    print("=" * 70)
    print()
    
    # Find latest progress bar
    latest_progress = None
    for line in reversed(recent):
        if '🚀 Epoch' in line:
            latest_progress = line.strip()
            break
    
    if latest_progress:
        # Parse progress bar
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', latest_progress)
        batch_match = re.search(r'(\d+)/(\d+) \[', latest_progress)
        pct_match = re.search(r'(\d+)%\|', latest_progress)
        time_match = re.search(r'\[(\d+:\d+)<(\d+:\d+), ([\d.]+)batch/s\]', latest_progress)
        
        if epoch_match:
            print(f"🔄 Epoch: {epoch_match.group(1)}/{epoch_match.group(2)}")
        
        if batch_match:
            print(f"📦 Batch: {batch_match.group(1)}/{batch_match.group(2)}", end="")
            if pct_match:
                print(f" ({pct_match.group(1)}%)")
            else:
                print()
        
        if time_match:
            print(f"⏱️  Time: {time_match.group(1)} elapsed | {time_match.group(2)} remaining")
            print(f"⚡ Speed: {time_match.group(3)} batch/s")
    
    print()
    print("-" * 70)
    print("📉 METRICS:")
    print("-" * 70)
    
    # Look for logged metrics
    metrics_found = False
    for line in reversed(recent):
        if 'Batch' in line and ('Loss:' in line or 'LR:' in line):
            # Extract metrics
            batch_match = re.search(r'Batch (\d+)/(\d+)', line)
            loss_match = re.search(r'Loss: ([\d.e-]+)', line)
            avg_loss_match = re.search(r'Avg Loss: ([\d.e-]+)', line)
            lr_match = re.search(r'LR: ([\d.e-]+)', line)
            best_val_match = re.search(r'Best Val Loss: ([\d.e-]+|N/A)', line)
            best_wer_match = re.search(r'Best WER: ([\d.e-]+|N/A)', line)
            
            if batch_match:
                print(f"   📍 Last logged at Batch {batch_match.group(1)}/{batch_match.group(2)}")
                metrics_found = True
            
            if loss_match:
                print(f"   📉 Current Loss:     {loss_match.group(1)}")
            
            if avg_loss_match:
                print(f"   📊 Average Loss:     {avg_loss_match.group(1)}")
            
            if lr_match:
                print(f"   📈 Learning Rate:    {lr_match.group(1)}")
            
            if best_val_match and best_val_match.group(1) != 'N/A':
                print(f"   🏆 Best Val Loss:    {best_val_match.group(1)}")
            
            if best_wer_match and best_wer_match.group(1) != 'N/A':
                print(f"   🏆 Best WER:         {best_wer_match.group(1)}")
            
            break
    
    # Look for epoch summary
    for line in reversed(recent):
        if '📊 EPOCH' in line or 'EPOCH SUMMARY' in line:
            # Find following metrics
            idx = recent.index(line)
            for i in range(idx, min(idx + 20, len(recent))):
                summary_line = recent[i]
                if 'Train Loss:' in summary_line:
                    match = re.search(r'Train Loss:\s+([\d.e-]+)', summary_line)
                    if match:
                        print(f"   🎯 Train Loss:      {match.group(1)}")
                elif 'Val Loss:' in summary_line:
                    match = re.search(r'Val Loss:\s+([\d.e-]+)', summary_line)
                    if match:
                        print(f"   ✅ Val Loss:        {match.group(1)}")
                elif 'Learning Rate:' in summary_line:
                    match = re.search(r'Learning Rate:\s+([\d.e-]+)', summary_line)
                    if match:
                        print(f"   📈 Learning Rate:   {match.group(1)}")
                elif 'WER:' in summary_line:
                    match = re.search(r'WER:\s+([\d.e-]+)', summary_line)
                    if match:
                        print(f"   📝 WER:             {match.group(1)}")
                elif 'CER:' in summary_line:
                    match = re.search(r'CER:\s+([\d.e-]+)', summary_line)
                    if match:
                        print(f"   📝 CER:             {match.group(1)}")
            break
    
    if not metrics_found:
        print("   ⚠️  Metrics will appear after batch 50")
        print("   💡 Progress bar shows: loss, avg_loss, lr, best_val, best_wer")
    
    print()
    print("=" * 70)
    print("💡 To watch real-time: tail -f", log_file, "| grep -E '(Loss|LR|WER|CER)'")
    print("=" * 70)

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/training_restart.log'
    show_metrics(log_file)

