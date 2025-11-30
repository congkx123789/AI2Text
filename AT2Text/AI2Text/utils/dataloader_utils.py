"""
Utility functions for optimizing DataLoader settings.
Auto-detects optimal num_workers based on CPU cores.
"""

import os
import multiprocessing
from typing import Optional


def get_optimal_num_workers(
    cpu_cores: Optional[int] = None,
    reserve_cores: int = 2,
    max_workers: int = 16
) -> int:
    """
    Calculate optimal num_workers for DataLoader based on CPU cores.
    
    Args:
        cpu_cores: Number of CPU cores (auto-detected if None)
        reserve_cores: Number of cores to reserve for main process and system
        max_workers: Maximum number of workers (safety limit)
        
    Returns:
        Optimal number of workers for DataLoader
        
    Examples:
        >>> # Auto-detect (recommended)
        >>> num_workers = get_optimal_num_workers()
        >>> 
        >>> # Manual specification
        >>> num_workers = get_optimal_num_workers(cpu_cores=16, reserve_cores=2)
    """
    if cpu_cores is None:
        cpu_cores = multiprocessing.cpu_count()
    
    # Calculate optimal workers: leave some cores for main process and system
    optimal = max(1, cpu_cores - reserve_cores)
    
    # Apply safety limit
    optimal = min(optimal, max_workers)
    
    return optimal


def get_cpu_info() -> dict:
    """
    Get CPU information for optimization recommendations.
    
    Returns:
        Dictionary with CPU information
    """
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = get_optimal_num_workers()
    
    # Try to get CPU model name (Linux)
    cpu_model = "Unknown"
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line.lower():
                        cpu_model = line.split(':')[1].strip()
                        break
    except Exception:
        pass
    
    return {
        'cpu_count': cpu_count,
        'cpu_model': cpu_model,
        'optimal_num_workers': optimal_workers,
        'recommended_num_workers': optimal_workers,
        'max_workers': min(optimal_workers + 2, 16)  # Can go slightly higher
    }


def print_dataloader_recommendations(config_path: Optional[str] = None):
    """
    Print DataLoader optimization recommendations based on system.
    
    Args:
        config_path: Optional path to config file to show current settings
    """
    cpu_info = get_cpu_info()
    
    print("=" * 60)
    print("DATALOADER OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    print(f"CPU Model: {cpu_info['cpu_model']}")
    print(f"CPU Cores: {cpu_info['cpu_count']}")
    print()
    print("RECOMMENDED SETTINGS:")
    print(f"  num_workers: {cpu_info['optimal_num_workers']}")
    print(f"    - Safe starting point")
    print(f"    - Leaves {2} cores for main process and system")
    print()
    print(f"  num_workers: {cpu_info['max_workers']} (if needed)")
    print(f"    - Can try this if {cpu_info['optimal_num_workers']} is not enough")
    print(f"    - Monitor CPU usage - should not be 100%")
    print()
    print("OTHER OPTIMIZATIONS:")
    print("  pin_memory: true")
    print("    - Faster CPU→GPU data transfer")
    print("    - Essential for GPU training")
    print()
    print("  persistent_workers: true")
    print("    - Keep workers alive between epochs")
    print("    - Avoids reinitialization overhead")
    print("    - Only works if num_workers > 0")
    print()
    print("  prefetch_factor: 2-4")
    print("    - Number of batches each worker prefetches")
    print("    - Higher = more memory, but GPU less likely to idle")
    print("    - Recommended: 2-3 for most cases, 4 if GPU is very fast")
    print()
    
    # Show current config if provided
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            current_workers = config.get('num_workers', 'Not set')
            current_pin = config.get('pin_memory', 'Not set')
            current_persistent = config.get('persistent_workers', 'Not set')
            current_prefetch = config.get('prefetch_factor', 'Not set')
            
            print("CURRENT CONFIG:")
            print(f"  num_workers: {current_workers}")
            print(f"  pin_memory: {current_pin}")
            print(f"  persistent_workers: {current_persistent}")
            print(f"  prefetch_factor: {current_prefetch}")
            print()
            
            # Recommendations
            if isinstance(current_workers, int):
                if current_workers < cpu_info['optimal_num_workers']:
                    print(f"⚠️  Consider increasing num_workers to {cpu_info['optimal_num_workers']}")
                elif current_workers > cpu_info['max_workers']:
                    print(f"⚠️  num_workers might be too high, consider reducing to {cpu_info['optimal_num_workers']}")
                else:
                    print("✅ num_workers looks good")
        except Exception as e:
            print(f"⚠️  Could not read config: {e}")
    
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Get DataLoader optimization recommendations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file to check current settings'
    )
    
    args = parser.parse_args()
    print_dataloader_recommendations(args.config)

