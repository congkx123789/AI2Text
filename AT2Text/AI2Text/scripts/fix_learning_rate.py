#!/usr/bin/env python3
"""
Script để điều chỉnh Learning Rate trong config file.

Nếu learning rate quá cao, script này sẽ giảm xuống 1/10 hoặc 1/2.
"""

import yaml
import argparse
import sys
from pathlib import Path


def fix_learning_rate(config_path: str, factor: float = 0.1, backup: bool = True):
    """
    Điều chỉnh learning rate trong config file.
    
    Args:
        config_path: Đường dẫn đến file config
        factor: Hệ số để nhân với learning rate (mặc định 0.1 = giảm 1/10)
        backup: Có tạo backup file không
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ File không tồn tại: {config_path}")
        return False
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    old_lr = config.get('learning_rate', None)
    
    if old_lr is None:
        print("⚠️  Không tìm thấy 'learning_rate' trong config file.")
        print("   Thêm vào config:")
        print("   learning_rate: 0.00001")
        return False
    
    # Calculate new learning rate
    new_lr = old_lr * factor
    
    print("=" * 80)
    print("ĐIỀU CHỈNH LEARNING RATE")
    print("=" * 80)
    print(f"Config file: {config_path}")
    print(f"Old learning rate: {old_lr}")
    print(f"Factor: {factor} ({'giảm 1/10' if factor == 0.1 else f'giảm {1/factor:.1f}x'})")
    print(f"New learning rate: {new_lr}")
    print()
    
    # Create backup
    if backup:
        backup_path = config_file.with_suffix('.yaml.backup')
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Backup created: {backup_path}")
    
    # Update config
    config['learning_rate'] = new_lr
    
    # Save config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config file: {config_path}")
    print()
    print("📝 Next steps:")
    print("   1. Chạy lại training với learning rate mới")
    print("   2. Kiểm tra lại với diagnose_loss_drop.py")
    print()
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fix learning rate in config file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Giảm learning rate xuống 1/10 (khuyến nghị)
  python fix_learning_rate.py --config configs/default.yaml
  
  # Giảm learning rate xuống 1/2
  python fix_learning_rate.py --config configs/default.yaml --factor 0.5
  
  # Không tạo backup
  python fix_learning_rate.py --config configs/default.yaml --no-backup
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--factor', type=float, default=0.1,
                       help='Factor to multiply learning rate (default: 0.1 = reduce by 10x)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup file')
    
    args = parser.parse_args()
    
    success = fix_learning_rate(
        config_path=args.config,
        factor=args.factor,
        backup=not args.no_backup
    )
    
    sys.exit(0 if success else 1)

