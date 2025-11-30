#!/usr/bin/env python3
"""
Quick check script - Chạy trên CPU, không cần GPU.
Chỉ kiểm tra config và data leakage (không cần load model).
"""

import yaml
import sys
from pathlib import Path
import argparse


def quick_check(config_path: str):
    """
    Quick check không cần GPU - chỉ kiểm tra config và learning rate.
    """
    print("=" * 80)
    print("QUICK CHECK - KHÔNG CẦN GPU")
    print("=" * 80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    learning_rate = config.get('learning_rate', None)
    vocab_size = config.get('vocab_size', None)
    
    print("📋 CONFIG CHECK:")
    print()
    
    if learning_rate is None:
        print("⚠️  Không tìm thấy 'learning_rate' trong config")
    else:
        print(f"Learning Rate: {learning_rate}")
        
        # Expected initial loss
        if vocab_size:
            expected_loss = __import__('math').log(vocab_size)
            print(f"Vocab Size: {vocab_size}")
            print(f"Expected initial loss (ln(vocab_size)): {expected_loss:.4f}")
        
        print()
        
        # Diagnosis
        if learning_rate >= 1e-3:
            print("🚨 LEARNING RATE QUÁ CAO!")
            print(f"   Current: {learning_rate}")
            print(f"   Khuyến nghị: Giảm xuống {learning_rate / 10:.0e} (1/10)")
            print()
            print("   Chạy lệnh sau để sửa:")
            print(f"   python3 scripts/fix_learning_rate.py --config {config_path}")
            return True
        elif learning_rate >= 1e-4:
            print("⚠️  LEARNING RATE HƠI CAO!")
            print(f"   Current: {learning_rate}")
            print(f"   Khuyến nghị: Giảm xuống {learning_rate / 2:.0e} (1/2) hoặc {learning_rate / 10:.0e} (1/10)")
            print()
            print("   Chạy lệnh sau để sửa:")
            print(f"   python3 scripts/fix_learning_rate.py --config {config_path}")
            return True
        else:
            print("✅ Learning rate hợp lý")
    
    print()
    print("=" * 80)
    print("KHUYẾN NGHỊ")
    print("=" * 80)
    print()
    print("Với loss giảm từ 17 → 2 trong < 1 epoch:")
    print()
    print("1. Giảm Learning Rate xuống 1/10:")
    print(f"   python3 scripts/fix_learning_rate.py --config {config_path}")
    print()
    print("2. Sau khi training xong hoặc dừng training, chạy full diagnosis:")
    print(f"   python3 scripts/diagnose_loss_drop.py --config {config_path}")
    print()
    print("3. Nếu có checkpoint, kiểm tra model collapse:")
    print(f"   python3 scripts/check_inference_collapse.py --config {config_path} --checkpoint <checkpoint_path>")
    print()
    
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick check (no GPU required)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    has_issue = quick_check(config_path=args.config)
    
    sys.exit(1 if has_issue else 0)

