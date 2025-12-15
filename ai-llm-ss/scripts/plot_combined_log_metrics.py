#!/usr/bin/env python3
"""
Vẽ biểu đồ kết hợp Loss, WER, CER trên cùng một biểu đồ với log scale để so sánh xu hướng giảm.
"""
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_checkpoint_info(checkpoint_dir):
    """Load loss và epoch từ tất cả checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    epochs = []
    train_losses = []
    
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    for ckpt_path in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                epoch = checkpoint.get('epoch')
                loss = checkpoint.get('loss')
                if epoch is not None and loss is not None:
                    epochs.append(epoch)
                    train_losses.append(float(loss))
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
            continue
    
    return epochs, train_losses

def load_test_metrics(metrics_path):
    """Load metrics từ file test results."""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    return None

def normalize_for_display(values, target_range=(0.1, 10)):
    """Chuẩn hóa giá trị để hiển thị trên cùng scale log."""
    values = np.array(values)
    min_val, max_val = values.min(), values.max()
    if max_val == min_val:
        return values
    
    # Scale về [0, 1]
    normalized = (values - min_val) / (max_val - min_val)
    # Scale về target_range
    scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]
    return scaled

def create_combined_log_plot(checkpoint_dir, metrics_path, output_path=None):
    """Tạo biểu đồ kết hợp Loss, WER, CER với log scale."""
    
    # Load dữ liệu
    epochs, train_losses = load_checkpoint_info(checkpoint_dir)
    test_metrics = load_test_metrics(metrics_path)
    
    if not epochs:
        print("Error: No checkpoint data found.")
        return None
    
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    
    # Test metrics
    test_wer = test_metrics['metrics']['wer'] * 100 if test_metrics else None
    test_cer = test_metrics['metrics']['cer'] * 100 if test_metrics else None
    test_epoch = test_metrics.get('epoch', epochs[-1]) if test_metrics else epochs[-1]
    
    # Tạo figure với 2 subplots: một cho loss riêng, một kết hợp cả 3
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    plt.suptitle('Biểu Đồ Log Scale: Loss, WER, CER\n(Hiển thị xu hướng giảm dần)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # --- SUBPLOT 1: TRAINING LOSS (LOG SCALE) ---
    if len(epochs) > 1:
        ax1.semilogy(epochs, train_losses, 'b-o', label='Train Loss', linewidth=3, markersize=10)
    else:
        ax1.semilogy(epochs, train_losses, 'b-o', label='Train Loss', linewidth=3, markersize=20)
        ax1.annotate(f'Loss: {train_losses[0]:.4f}\nEpoch: {int(epochs[0])}', 
                    xy=(epochs[0], train_losses[0]), 
                    xytext=(epochs[0] + 1, train_losses[0] * 2),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.set_title('Training Loss\n(Log Scale)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (log scale)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')
    ax1.legend(fontsize=11, loc='upper right')
    
    # Highlight best loss
    if len(train_losses) > 0:
        min_loss_idx = np.argmin(train_losses)
        min_loss_epoch = epochs[min_loss_idx]
        min_loss_val = train_losses[min_loss_idx]
        ax1.axvline(x=min_loss_epoch, color='red', linestyle=':', alpha=0.7, linewidth=2)
        if len(epochs) > 1:
            ax1.plot(min_loss_epoch, min_loss_val, 'ro', markersize=15, 
                    label=f'Best: {min_loss_val:.4f}', zorder=5)
            ax1.legend(fontsize=11, loc='upper right')
    
    # --- SUBPLOT 2: COMBINED PLOT (Loss, WER, CER) ---
    # Vẽ Loss
    if len(epochs) > 1:
        line1 = ax2.semilogy(epochs, train_losses, 'b-o', label='Train Loss', 
                            linewidth=2.5, markersize=8, alpha=0.8)
    else:
        line1 = ax2.semilogy(epochs, train_losses, 'b-o', label='Train Loss', 
                            linewidth=2.5, markersize=15, alpha=0.8)
    
    # Vẽ WER và CER (test values)
    if test_wer is not None:
        # Tạo array epochs cho test (chỉ có 1 điểm)
        test_epochs = np.array([test_epoch])
        test_wers = np.array([test_wer])
        line2 = ax2.semilogy(test_epochs, test_wers, 'purple', marker='s', 
                            markersize=12, label=f'Test WER: {test_wer:.2f}%', 
                            linewidth=3, linestyle='-', alpha=0.9)
        ax2.axvline(x=test_epoch, color='purple', linestyle=':', alpha=0.5, linewidth=1.5)
    
    if test_cer is not None:
        test_epochs = np.array([test_epoch])
        test_cers = np.array([test_cer])
        line3 = ax2.semilogy(test_epochs, test_cers, 'orange', marker='^', 
                            markersize=12, label=f'Test CER: {test_cer:.2f}%', 
                            linewidth=3, linestyle='--', alpha=0.9)
        ax2.axvline(x=test_epoch, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax2.set_title('Kết Hợp: Loss, WER, CER\n(Log Scale - So sánh xu hướng)', 
                 fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Giá trị (log scale)', fontsize=11)
    
    # Set xlim
    max_epoch = max(epochs.max() if epochs.size > 0 else 1, test_epoch)
    ax2.set_xlim([0, max_epoch + 1])
    
    # Set ylim dựa trên tất cả giá trị
    all_values = list(train_losses)
    if test_wer is not None:
        all_values.append(test_wer)
    if test_cer is not None:
        all_values.append(test_cer)
    
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        # Mở rộng range một chút
        ax2.set_ylim([min_val * 0.3, max_val * 3])
    
    ax2.grid(True, linestyle='--', alpha=0.6, which='both')
    ax2.legend(fontsize=10, loc='upper right')
    
    # Thêm text box với thông tin tổng hợp
    if test_metrics:
        info_text = f"""Kết Quả Test:
• WER: {test_wer:.2f}%
• CER: {test_cer:.2f}%
• RTF: {test_metrics['metrics']['rtf']:.4f}
• Samples: {test_metrics['metrics']['total_samples']:,}"""
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Thêm giải thích về log scale
    explanation_text = "Lưu ý: Trục Y dùng log scale\nđể hiển thị rõ xu hướng giảm"
    ax2.text(0.98, 0.02, explanation_text, transform=ax2.transAxes, 
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Lưu biểu đồ
    if output_path is None:
        output_path = Path(__file__).parent.parent / "experiments" / "reports" / "combined_log_metrics.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Để chỗ cho suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Biểu đồ kết hợp log scale đã được lưu tại: {output_path}")
    
    # Hiển thị biểu đồ
    try:
        plt.show()
    except:
        pass
    
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ kết hợp Loss, WER, CER với log scale")
    parser.add_argument("--checkpoint_dir", default="data/results/checkpoints",
                      help="Thư mục chứa checkpoints")
    parser.add_argument("--metrics", default="experiments/reports/metrics.json",
                      help="File metrics từ test")
    parser.add_argument("--output", default="experiments/reports/combined_log_metrics.png",
                      help="Đường dẫn lưu biểu đồ")
    args = parser.parse_args()
    
    create_combined_log_plot(args.checkpoint_dir, args.metrics, args.output)

if __name__ == "__main__":
    main()

