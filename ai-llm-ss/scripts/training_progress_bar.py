#!/usr/bin/env python3
"""
Beautiful real-time training progress monitor with progress bars.
Usage: python scripts/training_progress_bar.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

class Colors:
    """ANSI color codes for terminal."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def progress_bar(current, total, width=50, filled_char='█', empty_char='░', 
                 show_percent=True, color=None):
    """Create beautiful progress bar."""
    if total == 0:
        bar = empty_char * width
        percent = 0
    else:
        filled = int(width * current / total)
        bar = filled_char * filled + empty_char * (width - filled)
        percent = (current / total) * 100
    
    color_code = color if color else Colors.RESET
    if show_percent:
        return f"{color_code}{bar}{Colors.RESET} {percent:5.1f}%"
    return f"{color_code}{bar}{Colors.RESET}"

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
                pid = pids[0]
                try:
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
        return None, None, None
    
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    if checkpoints:
        latest = checkpoints[0]
        epoch = int(latest.stem.split('_')[-1])
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        
        # Get loss from checkpoint
        try:
            import torch
            ckpt = torch.load(latest, map_location='cpu')
            loss = ckpt.get('loss', None)
        except:
            loss = None
        
        return epoch, mtime, loss
    return None, None, None

def get_config_info():
    """Get training config."""
    config_path = Path('config/train_merged.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def parse_etime(etime_str):
    """Parse elapsed time string from ps."""
    parts = etime_str.split(':')
    if len(parts) == 3:
        days, hours, mins = parts
        days = int(days.split('-')[0])
        hours = int(hours)
        mins = int(mins)
        return days * 86400 + hours * 3600 + mins * 60
    elif len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    return 0

def monitor():
    """Main monitoring loop."""
    checkpoint_dir = Path('data/results/checkpoints')
    config = get_config_info()
    
    clear_screen()
    
    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 TRAINING PROGRESS MONITOR{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.GRAY}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print()
    
    # Process info
    proc_info = get_training_process()
    if proc_info:
        etime_str = proc_info[0]
        pid = proc_info[1]
        elapsed_seconds = parse_etime(etime_str)
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Training Process: {Colors.BOLD}PID {pid}{Colors.RESET}")
        print(f"  Elapsed time: {Colors.YELLOW}{format_time(elapsed_seconds)}{Colors.RESET} ({etime_str})")
    else:
        print(f"{Colors.RED}✗ Training process not found!{Colors.RESET}")
        return
    
    # GPU info
    gpu_used, gpu_total, gpu_util = get_gpu_info()
    if gpu_used is not None:
        gpu_percent = (gpu_used / gpu_total) * 100
        
        print()
        print(f"{Colors.BOLD}📊 GPU Status{Colors.RESET}")
        print(f"  Memory: {gpu_used:,} MB / {gpu_total:,} MB ({gpu_percent:.1f}%)")
        gpu_mem_bar = progress_bar(gpu_used, gpu_total, width=40, color=Colors.BLUE)
        print(f"  [{gpu_mem_bar}]")
        
        print(f"  Utilization: {gpu_util}%")
        # Color based on utilization
        if gpu_util >= 80:
            gpu_color = Colors.GREEN
        elif gpu_util >= 50:
            gpu_color = Colors.YELLOW
        else:
            gpu_color = Colors.RED
        gpu_util_bar = progress_bar(gpu_util, 100, width=40, color=gpu_color)
        print(f"  [{gpu_util_bar}]")
    
    # Checkpoint info
    latest_epoch, checkpoint_time, loss = get_latest_checkpoint(checkpoint_dir)
    if latest_epoch and config:
        total_epochs = config.get('epochs', 20)
        progress = (latest_epoch / total_epochs) * 100
        
        print()
        print(f"{Colors.BOLD}📁 Training Progress{Colors.RESET}")
        print(f"  Current epoch: {Colors.BOLD}{Colors.CYAN}{latest_epoch}{Colors.RESET} / {total_epochs}")
        
        # Epoch progress bar
        epoch_color = Colors.GREEN if progress >= 50 else Colors.YELLOW
        epoch_bar = progress_bar(latest_epoch, total_epochs, width=50, color=epoch_color)
        print(f"  [{epoch_bar}]")
        
        if loss is not None:
            print(f"  Loss: {Colors.YELLOW}{loss:.4f}{Colors.RESET}")
        
        if checkpoint_time:
            time_since_checkpoint = datetime.now() - checkpoint_time
            print(f"  Last checkpoint: {checkpoint_time.strftime('%H:%M:%S')} "
                  f"({Colors.GRAY}{format_time(time_since_checkpoint.total_seconds())} ago{Colors.RESET})")
        
        # Estimate remaining time - use checkpoint times for accuracy
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), 
                            key=lambda x: x.stat().st_mtime)
        if len(checkpoints) >= 2:
            # Calculate from actual checkpoint times
            times = []
            for i in range(1, len(checkpoints)):
                prev_time = datetime.fromtimestamp(checkpoints[i-1].stat().st_mtime)
                curr_time = datetime.fromtimestamp(checkpoints[i].stat().st_mtime)
                duration = (curr_time - prev_time).total_seconds()
                times.append(duration)
            
            if times:
                avg_time_per_epoch = sum(times) / len(times)
                remaining_epochs = total_epochs - latest_epoch
                remaining_time = avg_time_per_epoch * remaining_epochs
                
                # Calculate total time from first to last checkpoint + remaining
                if len(checkpoints) > 0:
                    first_time = datetime.fromtimestamp(checkpoints[0].stat().st_mtime)
                    last_time = datetime.fromtimestamp(checkpoints[-1].stat().st_mtime)
                    elapsed_from_checkpoints = (last_time - first_time).total_seconds()
                    total_time = elapsed_from_checkpoints + remaining_time
                else:
                    total_time = avg_time_per_epoch * total_epochs
                
                print()
                print(f"{Colors.BOLD}⏱️  Time Estimates{Colors.RESET}")
                print(f"  Time per epoch: {Colors.CYAN}~{format_time(avg_time_per_epoch)}{Colors.RESET}")
                print(f"  Remaining: {Colors.YELLOW}~{format_time(remaining_time)}{Colors.RESET}")
                print(f"  Total: {Colors.BLUE}~{format_time(total_time)}{Colors.RESET}")
                
                # Overall progress bar
                if len(checkpoints) > 0:
                    first_time = datetime.fromtimestamp(checkpoints[0].stat().st_mtime)
                    elapsed_from_start = (datetime.now() - first_time).total_seconds()
                    overall_progress = (elapsed_from_start / total_time) * 100 if total_time > 0 else 0
                    overall_bar = progress_bar(elapsed_from_start, total_time, width=50, color=Colors.MAGENTA)
                    print(f"  [{overall_bar}]")
        elif latest_epoch > 0 and elapsed_seconds > 0:
            # Fallback: estimate from process time
            time_per_epoch = elapsed_seconds / latest_epoch
            remaining_epochs = total_epochs - latest_epoch
            remaining_time = time_per_epoch * remaining_epochs
            total_time = time_per_epoch * total_epochs
            
            print()
            print(f"{Colors.BOLD}⏱️  Time Estimates{Colors.RESET}")
            print(f"  Time per epoch: {Colors.CYAN}~{format_time(time_per_epoch)}{Colors.RESET}")
            print(f"  Remaining: {Colors.YELLOW}~{format_time(remaining_time)}{Colors.RESET}")
            print(f"  Total: {Colors.BLUE}~{format_time(total_time)}{Colors.RESET}")
            
            # Overall progress bar
            overall_progress = (elapsed_seconds / total_time) * 100 if total_time > 0 else 0
            overall_bar = progress_bar(elapsed_seconds, total_time, width=50, color=Colors.MAGENTA)
            print(f"  [{overall_bar}]")
    
    # Config summary
    if config:
        print()
        print(f"{Colors.BOLD}⚙️  Configuration{Colors.RESET}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  Effective batch: {config.get('batch_size', 0) * config.get('gradient_accumulation_steps', 1)}")
        print(f"  Workers: {config.get('num_workers', 0)}")
        print(f"  AMP: {Colors.GREEN if config.get('amp', False) else Colors.RED}{config.get('amp', False)}{Colors.RESET}")
    
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.GRAY}Press Ctrl+C to exit. Auto-refresh every 2 seconds...{Colors.RESET}")

if __name__ == "__main__":
    try:
        while True:
            monitor()
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Monitoring stopped.{Colors.RESET}")

