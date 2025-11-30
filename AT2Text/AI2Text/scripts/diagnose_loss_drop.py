#!/usr/bin/env python3
"""
Script tổng hợp để chẩn đoán vấn đề Loss giảm quá nhanh.

Chạy tất cả các checks:
1. Model Collapse (output blank)
2. Data Leakage
3. Train vs Val Loss gap
4. Loss Function correctness

Sau đó đưa ra khuyến nghị.
"""

import sys
import subprocess
from pathlib import Path
import argparse
import yaml


def run_check(script_name: str, config_path: str, checkpoint_path: str = None, needs_checkpoint: bool = False, **kwargs):
    """Chạy một check script."""
    script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path), '--config', config_path]
    
    if checkpoint_path and needs_checkpoint:
        cmd.extend(['--checkpoint', checkpoint_path])
    
    for key, value in kwargs.items():
        cmd.extend([key, str(value)])
    
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print('='*80)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def diagnose_loss_drop(config_path: str, checkpoint_path: str = None):
    """
    Chạy tất cả các checks để chẩn đoán vấn đề loss giảm quá nhanh.
    
    Args:
        config_path: Đường dẫn đến file config
        checkpoint_path: Đường dẫn đến checkpoint (optional)
    """
    print("=" * 80)
    print("CHẨN ĐOÁN VẤN ĐỀ: LOSS GIẢM QUÁ NHANH")
    print("=" * 80)
    print()
    print("Script này sẽ chạy các kiểm tra sau:")
    print("1. Model Collapse - Model có output blank không?")
    print("2. Data Leakage - Target có lẫn vào input không?")
    print("3. Train vs Val Loss - Gap có quá lớn không?")
    print("4. Loss Function - Công thức tính loss có đúng không?")
    print()
    
    # Load config để lấy learning rate
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    learning_rate = config.get('learning_rate', 1e-4)
    print(f"Current Learning Rate: {learning_rate}")
    print()
    
    # Run checks
    results = {}
    
    # Check 1: Model Collapse
    results['collapse'] = run_check(
        'check_inference_collapse.py',
        config_path,
        checkpoint_path,
        needs_checkpoint=True,
        **{'--num_samples': '20'}
    )
    
    # Check 2: Data Leakage
    results['leakage'] = run_check(
        'check_data_leakage.py',
        config_path,
        checkpoint_path,
        needs_checkpoint=False,
        **{'--num_samples': '50'}
    )
    
    # Check 3: Loss Gap
    results['loss_gap'] = run_check(
        'check_loss_validation.py',
        config_path,
        checkpoint_path,
        needs_checkpoint=True,
        **{'--num_batches': '50'}
    )
    
    # Check 4: Loss Function
    results['loss_function'] = run_check(
        'check_loss_function.py',
        config_path,
        checkpoint_path,
        needs_checkpoint=True,
        **{'--num_batches': '20'}
    )
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("TÓM TẮT VÀ KHUYẾN NGHỊ")
    print("=" * 80)
    print()
    
    issues_found = []
    
    if not results['collapse']:
        issues_found.append("Model Collapse")
    if not results['leakage']:
        issues_found.append("Data Leakage")
    if not results['loss_gap']:
        issues_found.append("Train/Val Loss Gap")
    if not results['loss_function']:
        issues_found.append("Loss Function Issue")
    
    if issues_found:
        print("🚨 CÁC VẤN ĐỀ ĐÃ PHÁT HIỆN:")
        for issue in issues_found:
            print(f"   - {issue}")
        print()
    else:
        print("✅ Không phát hiện vấn đề nghiêm trọng.")
        print()
    
    # Learning rate check
    if learning_rate >= 1e-3:
        print("⚠️  LEARNING RATE QUÁ CAO!")
        print(f"   Current LR: {learning_rate}")
        print("   Khuyến nghị: Giảm xuống 1/10 (ví dụ: 1e-3 → 1e-4)")
        print()
        print("   Cách sửa:")
        print("   1. Mở file config YAML")
        print("   2. Tìm 'learning_rate'")
        print(f"   3. Đổi từ {learning_rate} thành {learning_rate / 10}")
        print()
    elif learning_rate >= 1e-4:
        print("⚠️  LEARNING RATE HƠI CAO!")
        print(f"   Current LR: {learning_rate}")
        print("   Khuyến nghị: Giảm xuống 1/2 (ví dụ: 1e-4 → 5e-5)")
        print()
    
    # General recommendations
    print("KHUYẾN NGHỊ TỔNG QUÁT:")
    print()
    print("1. Nếu Model Collapse:")
    print("   - Giảm learning rate xuống 1/10")
    print("   - Kiểm tra lại loss function (CTC loss có zero_infinity=True)")
    print("   - Kiểm tra output_lengths >= text_lengths")
    print()
    print("2. Nếu Data Leakage:")
    print("   - Kiểm tra lại code trong dataset.py")
    print("   - Đảm bảo input audio và text label hoàn toàn tách biệt")
    print("   - Kiểm tra file paths không chứa transcript")
    print()
    print("3. Nếu Train Loss << Val Loss:")
    print("   - Tăng regularization (dropout, weight decay)")
    print("   - Kiểm tra train/val split")
    print("   - Kiểm tra data leakage")
    print()
    print("4. Nếu Loss Function sai:")
    print("   - Đảm bảo CTC Loss dùng reduction='mean'")
    print("   - Kiểm tra output_lengths >= text_lengths")
    print("   - Kiểm tra log_softmax được tính đúng")
    print()
    print("5. Nếu Loss giảm quá nhanh (17 → 2 trong < 1 epoch):")
    print("   - Giảm learning rate xuống 1/10")
    print("   - Kiểm tra model có bị collapse không")
    print("   - Kiểm tra loss function scale")
    print("   - Với CTC Loss, loss thường giảm từ từ (10 → 8 → 6...)")
    print()
    
    return len(issues_found) > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diagnose rapid loss drop issue',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check with config only (random model)
  python diagnose_loss_drop.py --config configs/default.yaml
  
  # Check with checkpoint
  python diagnose_loss_drop.py --config configs/default.yaml --checkpoint checkpoints/best.pt
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    
    args = parser.parse_args()
    
    has_issues = diagnose_loss_drop(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    sys.exit(1 if has_issues else 0)

