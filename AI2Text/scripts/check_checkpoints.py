#!/usr/bin/env python3
"""
Script to check checkpoint status and verify checkpoint saving.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def check_checkpoints():
    """Check checkpoint status."""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("❌ Checkpoint directory does not exist!")
        return
    
    # List all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    best_model = checkpoint_dir / "best_model.pt"
    
    print("=" * 80)
    print("📊 CHECKPOINT STATUS")
    print("=" * 80)
    
    if best_model.exists():
        try:
            ckpt = torch.load(best_model, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', 'N/A')
            val_loss = ckpt.get('best_val_loss', 'N/A')
            size_mb = best_model.stat().st_size / (1024**2)
            val_loss_str = f"{val_loss:.6f}" if isinstance(val_loss, float) else str(val_loss)
            print(f"✅ Best Model: epoch={epoch}, val_loss={val_loss_str}, size={size_mb:.1f}MB")
        except Exception as e:
            print(f"❌ Error loading best_model.pt: {e}")
    else:
        print("❌ best_model.pt not found!")
    
    print(f"\n📁 Periodic Checkpoints: {len(checkpoints)} files")
    
    if checkpoints:
        # Get epoch numbers
        epochs = []
        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', -1)
                epochs.append(epoch)
            except Exception as e:
                print(f"⚠️  Error loading {ckpt_path.name}: {e}")
        
        if epochs:
            epochs = sorted(epochs)
            print(f"   Latest: epoch {epochs[-1]}")
            print(f"   Range: epoch {epochs[0]} to {epochs[-1]}")
            
            # Check for missing epochs
            expected_epochs = set(range(epochs[0], epochs[-1] + 1))
            actual_epochs = set(epochs)
            missing = expected_epochs - actual_epochs
            
            if missing:
                print(f"\n⚠️  MISSING CHECKPOINTS: {sorted(missing)}")
            else:
                print(f"\n✅ All checkpoints present from epoch {epochs[0]} to {epochs[-1]}")
    
    print("=" * 80)

if __name__ == "__main__":
    check_checkpoints()

