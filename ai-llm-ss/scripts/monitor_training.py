#!/usr/bin/env python3
"""Real-time training monitor."""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

def get_training_process():
    """Get training process info."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'train_ctc'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            if pids:
                # Get main process (first one, usually the parent)
                pid = pids[0]
                try:
                    # Get process elapsed time
                    result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'etime,pid,cmd'],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            return lines[1].split(None, 2)
                except:
                    pass
    except:
        pass
    return None

def get_gpu_info():
    """Get GPU memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            line = result.stdout.strip()
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    used = int(parts[0])
                    total = int(parts[1])
                    util = int(parts[2])
                    return used, total, util
    except:
        pass
    return None, None, None

def get_latest_checkpoint(checkpoint_dir):
    """Get latest checkpoint info."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, None
    
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    if checkpoints:
        latest = checkpoints[0]
        # Extract epoch number
        epoch = int(latest.stem.split('_')[-1])
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        return epoch, mtime
    return None, None

def get_config_info():
    """Get training config."""
    config_path = Path('config/train_merged.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def format_time(seconds):
    """Format seconds to readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"

def parse_etime(etime_str):
    """Parse elapsed time string from ps."""
    # Format: [[DD-]hh:]mm:ss
    parts = etime_str.split(':')
    if len(parts) == 3:  # DD-hh:mm:ss
        days, hours, mins = parts
        days = int(days.split('-')[0])
        hours = int(hours)
        mins = int(mins)
        return days * 86400 + hours * 3600 + mins * 60
    elif len(parts) == 2:  # mm:ss
        mins, secs = map(int, parts)
        return mins * 60 + secs
    return 0

def monitor():
    """Main monitoring loop."""
    checkpoint_dir = Path('data/results/checkpoints')
    config = get_config_info()
    
    print("\033[2J\033[H", end='')  # Clear screen
    print("=" * 80)
    print("TRAINING MONITOR - Real-time Status")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Process info
    proc_info = get_training_process()
    if proc_info:
        etime_str = proc_info[0]
        pid = proc_info[1]
        elapsed_seconds = parse_etime(etime_str)
        
        print(f"✓ Training Process: PID {pid}")
        print(f"  Elapsed time: {format_time(elapsed_seconds)} ({etime_str})")
    else:
        print("✗ Training process not found!")
        return
    
    # GPU info
    gpu_used, gpu_total, gpu_util = get_gpu_info()
    if gpu_used is not None:
        gpu_percent = (gpu_used / gpu_total) * 100
        print(f"\n📊 GPU Status:")
        print(f"  Memory: {gpu_used:,} MB / {gpu_total:,} MB ({gpu_percent:.1f}%)")
        print(f"  Utilization: {gpu_util}%")
    
    # Checkpoint info
    latest_epoch, checkpoint_time = get_latest_checkpoint(checkpoint_dir)
    if latest_epoch and config:
        total_epochs = config.get('epochs', 20)
        progress = (latest_epoch / total_epochs) * 100
        
        print(f"\n📁 Training Progress:")
        print(f"  Current epoch: {latest_epoch}/{total_epochs} ({progress:.1f}%)")
        if checkpoint_time:
            time_since_checkpoint = datetime.now() - checkpoint_time
            print(f"  Last checkpoint: {checkpoint_time.strftime('%H:%M:%S')} ({format_time(time_since_checkpoint.total_seconds())} ago)")
        
        # Estimate remaining time
        if latest_epoch > 0 and elapsed_seconds > 0:
            time_per_epoch = elapsed_seconds / latest_epoch
            remaining_epochs = total_epochs - latest_epoch
            remaining_time = time_per_epoch * remaining_epochs
            total_time = time_per_epoch * total_epochs
            
            print(f"\n⏱️  Time Estimates:")
            print(f"  Time per epoch: ~{format_time(time_per_epoch)}")
            print(f"  Remaining: ~{format_time(remaining_time)}")
            print(f"  Total: ~{format_time(total_time)}")
    
    # Config summary
    if config:
        print(f"\n⚙️  Configuration:")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  Effective batch: {config.get('batch_size', 0) * config.get('gradient_accumulation_steps', 1)}")
        print(f"  Workers: {config.get('num_workers', 0)}")
        print(f"  AMP: {config.get('amp', False)}")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit. Auto-refresh every 5 seconds...")

if __name__ == "__main__":
    try:
        while True:
            monitor()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

