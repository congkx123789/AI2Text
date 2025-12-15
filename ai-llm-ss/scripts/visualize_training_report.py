#!/usr/bin/env python3
"""
Tạo biểu đồ báo cáo chất lượng model từ training logs và test results.
"""
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_checkpoint_info(checkpoint_dir):
    """Load loss và epoch từ tất cả checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    epochs = []
    train_losses = []
    
    # Tìm tất cả checkpoint files
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
        data = json.load(f)
    return data

def create_training_report(checkpoint_dir, metrics_path, output_path=None):
    """Tạo biểu đồ báo cáo training và test results."""
    
    # Load dữ liệu
    epochs, train_losses = load_checkpoint_info(checkpoint_dir)
    test_metrics = load_test_metrics(metrics_path)
    
    if not epochs:
        print("Error: No checkpoint data found. Please train the model first.")
        return
    
    # Chuẩn bị dữ liệu
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    
    # Test metrics
    test_wer = test_metrics['metrics']['wer'] * 100  # Convert to percentage
    test_cer = test_metrics['metrics']['cer'] * 100
    test_epoch = test_metrics.get('epoch', epochs[-1] if epochs.size > 0 else 1)
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(f'Model Training & Evaluation Report\nFinal Test: WER={test_wer:.2f}%, CER={test_cer:.2f}%', 
                 fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: TRAINING LOSS CURVE ---
    if len(epochs) > 1:
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    else:
        # Nếu chỉ có 1 điểm, vẽ bar chart hoặc điểm lớn
        ax1.bar(epochs, train_losses, width=0.5, color='steelblue', alpha=0.7, label='Train Loss')
        ax1.scatter(epochs, train_losses, s=200, color='red', zorder=5, label='Current Loss')
    
    ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Highlight best loss
    if len(train_losses) > 0:
        min_loss_idx = np.argmin(train_losses)
        min_loss_epoch = epochs[min_loss_idx]
        min_loss_val = train_losses[min_loss_idx]
        if len(epochs) > 1:
            ax1.axvline(x=min_loss_epoch, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)
            ax1.plot(min_loss_epoch, min_loss_val, 'ro', markersize=10, label=f'Best: {min_loss_val:.4f}')
            ax1.text(min_loss_epoch, max(train_losses)*0.95, f' Best\n Epoch {int(min_loss_epoch)}', 
                    color='red', fontsize=10, fontweight='bold', ha='left')
        else:
            # Hiển thị thông tin loss cho 1 điểm
            ax1.text(min_loss_epoch, min_loss_val * 1.1, f'Loss: {min_loss_val:.4f}\nEpoch: {int(min_loss_epoch)}', 
                    color='red', fontsize=11, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax1.legend(fontsize=11)
    
    # --- SUBPLOT 2: PERFORMANCE METRICS ---
    # Vẽ WER và CER trên cùng biểu đồ (nếu có validation data)
    # Hiện tại chỉ có test data, nên vẽ đường ngang
    ax2.axhline(y=test_wer, color='purple', linestyle='-', linewidth=3, 
                label=f'Test WER: {test_wer:.2f}%', alpha=0.8)
    ax2.axhline(y=test_cer, color='orange', linestyle='--', linewidth=3, 
                label=f'Test CER: {test_cer:.2f}%', alpha=0.8)
    
    # Thêm annotation cho test epoch
    ax2.axvline(x=test_epoch, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    ax2.text(test_epoch, max(test_wer, test_cer) * 0.9, f' Test\n Epoch {test_epoch}', 
            color='green', fontsize=10, fontweight='bold', ha='left')
    
    ax2.set_title('Word & Character Error Rates (Test Set)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_xlim([0, max(epochs.max() if epochs.size > 0 else 1, test_epoch) + 1])
    ax2.set_ylim([0, max(test_wer, test_cer) * 1.2])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11, loc='upper right')
    
    # Thêm text box với thông tin chi tiết
    info_text = f"""Test Results:
• Total Samples: {test_metrics['metrics']['total_samples']:,}
• Exact Match: {test_metrics['metrics']['exact_accuracy']:.2f}%
• Word Accuracy: {test_metrics['metrics']['word_accuracy']:.2f}%
• Sentence Error Rate: {test_metrics['metrics']['sentence_error_rate']:.2f}%
• RTF: {test_metrics['metrics']['rtf']:.4f}"""
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    if output_path is None:
        output_path = Path(__file__).parent.parent / "experiments" / "reports" / "training_report.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Biểu đồ đã được lưu tại: {output_path}")
    
    # Hiển thị biểu đồ
    plt.show()
    
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tạo biểu đồ báo cáo training và test")
    parser.add_argument("--checkpoint_dir", default="data/results/checkpoints",
                      help="Thư mục chứa checkpoints")
    parser.add_argument("--metrics", default="experiments/reports/metrics.json",
                      help="File metrics từ test")
    parser.add_argument("--output", default="experiments/reports/training_report.png",
                      help="Đường dẫn lưu biểu đồ")
    args = parser.parse_args()
    
    create_training_report(args.checkpoint_dir, args.metrics, args.output)

if __name__ == "__main__":
    main()

