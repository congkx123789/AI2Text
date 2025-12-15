#!/usr/bin/env python3
"""
Vẽ biểu đồ Loss, WER, CER với scale logarit để hiển thị sự giảm dần của các metrics.
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

def create_log_scale_plots(checkpoint_dir, metrics_path, output_path=None):
    """Tạo biểu đồ với scale logarit cho Loss, WER, CER."""
    
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
    
    # Tạo figure với 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle('Training Metrics với Log Scale\n(Hiển thị sự giảm dần của Loss, WER, CER)', 
                 fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: TRAINING LOSS (LOG SCALE) ---
    ax1 = axes[0]
    if len(epochs) > 1:
        ax1.semilogy(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2.5, markersize=8)
    else:
        # Nếu chỉ có 1 điểm, vẽ điểm lớn
        ax1.semilogy(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2.5, markersize=15)
        # Thêm annotation
        ax1.annotate(f'Loss: {train_losses[0]:.4f}\nEpoch: {int(epochs[0])}', 
                    xy=(epochs[0], train_losses[0]), 
                    xytext=(epochs[0] + 0.5, train_losses[0] * 1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_title('Training Loss (Log Scale)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (log scale)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')
    ax1.legend(fontsize=10)
    
    # Highlight best loss
    if len(train_losses) > 0:
        min_loss_idx = np.argmin(train_losses)
        min_loss_epoch = epochs[min_loss_idx]
        min_loss_val = train_losses[min_loss_idx]
        ax1.axvline(x=min_loss_epoch, color='red', linestyle=':', alpha=0.7, linewidth=2)
        if len(epochs) > 1:
            ax1.plot(min_loss_epoch, min_loss_val, 'ro', markersize=12, 
                    label=f'Best: {min_loss_val:.4f}')
            ax1.legend(fontsize=10)
    
    # --- SUBPLOT 2: WER (LOG SCALE) ---
    ax2 = axes[1]
    if test_wer is not None:
        # Vẽ đường ngang cho test WER
        ax2.axhline(y=test_wer, color='purple', linestyle='-', linewidth=3, 
                    label=f'Test WER: {test_wer:.2f}%', alpha=0.8)
        ax2.axvline(x=test_epoch, color='green', linestyle=':', alpha=0.6, linewidth=2)
        
        # Nếu có nhiều điểm validation WER, có thể vẽ ở đây
        # Hiện tại chỉ có test WER
        ax2.text(test_epoch, test_wer * 1.1, f' Test\n Epoch {test_epoch}', 
                color='green', fontsize=10, fontweight='bold', ha='left')
    
    ax2.set_yscale('log')
    ax2.set_title('Word Error Rate - WER (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('WER (%) - log scale', fontsize=11)
    ax2.set_xlim([0, max(epochs.max() if epochs.size > 0 else 1, test_epoch) + 1])
    if test_wer is not None:
        ax2.set_ylim([test_wer * 0.5, test_wer * 2])
    ax2.grid(True, linestyle='--', alpha=0.6, which='both')
    ax2.legend(fontsize=10, loc='upper right')
    
    # Thêm text box với thông tin
    if test_wer is not None:
        info_text = f"""Test Results:
• WER: {test_wer:.2f}%
• Samples: {test_metrics['metrics']['total_samples']:,}
• Exact Match: {test_metrics['metrics']['exact_accuracy']:.2f}%"""
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # --- SUBPLOT 3: CER (LOG SCALE) ---
    ax3 = axes[2]
    if test_cer is not None:
        # Vẽ đường ngang cho test CER
        ax3.axhline(y=test_cer, color='orange', linestyle='--', linewidth=3, 
                   label=f'Test CER: {test_cer:.2f}%', alpha=0.8)
        ax3.axvline(x=test_epoch, color='green', linestyle=':', alpha=0.6, linewidth=2)
        
        ax3.text(test_epoch, test_cer * 1.1, f' Test\n Epoch {test_epoch}', 
                color='green', fontsize=10, fontweight='bold', ha='left')
    
    ax3.set_yscale('log')
    ax3.set_title('Character Error Rate - CER (Log Scale)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('CER (%) - log scale', fontsize=11)
    ax3.set_xlim([0, max(epochs.max() if epochs.size > 0 else 1, test_epoch) + 1])
    if test_cer is not None:
        ax3.set_ylim([test_cer * 0.5, test_cer * 2])
    ax3.grid(True, linestyle='--', alpha=0.6, which='both')
    ax3.legend(fontsize=10, loc='upper right')
    
    # Thêm text box với thông tin
    if test_cer is not None:
        info_text = f"""Test Results:
• CER: {test_cer:.2f}%
• RTF: {test_metrics['metrics']['rtf']:.4f}
• Speed: {test_metrics['metrics']['total_audio_seconds']/test_metrics['metrics']['total_inference_seconds']:.1f}x"""
        ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    if output_path is None:
        output_path = Path(__file__).parent.parent / "experiments" / "reports" / "log_scale_metrics.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Biểu đồ log scale đã được lưu tại: {output_path}")
    
    # Hiển thị biểu đồ
    try:
        plt.show()
    except:
        pass  # Ignore if non-interactive
    
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ Loss, WER, CER với log scale")
    parser.add_argument("--checkpoint_dir", default="data/results/checkpoints",
                      help="Thư mục chứa checkpoints")
    parser.add_argument("--metrics", default="experiments/reports/metrics.json",
                      help="File metrics từ test")
    parser.add_argument("--output", default="experiments/reports/log_scale_metrics.png",
                      help="Đường dẫn lưu biểu đồ")
    args = parser.parse_args()
    
    create_log_scale_plots(args.checkpoint_dir, args.metrics, args.output)

if __name__ == "__main__":
    main()

