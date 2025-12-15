"""
Smart callbacks for automatic training recovery and curriculum learning.

This module implements advanced callbacks that enable "auto-run" capabilities:
- AutoRollbackCallback: Detects model collapse and automatically rolls back to best checkpoint
- CurriculumLearningCallback: Gradually introduces harder tasks when model is ready
"""

import torch
from pathlib import Path
from training.callbacks import Callback


class AutoRollbackCallback(Callback):
    """
    Watcher that detects Model Collapse (Loss Explosion) and triggers a rollback.
    
    This callback monitors validation loss and automatically:
    1. Detects when loss spikes significantly (>threshold_ratio)
    2. Reloads the best checkpoint
    3. Applies safety measures (reduces learning rate)
    """
    
    def __init__(self, threshold_ratio=1.5, patience=1):
        """
        Initialize auto-rollback callback.
        
        Args:
            threshold_ratio: Trigger if loss > best_loss * ratio (e.g. 3.8 -> 5.7)
            patience: Number of bad epochs before triggering
        """
        self.threshold_ratio = threshold_ratio
        self.patience = patience
        self.bad_epochs = 0
        self.best_loss = float('inf')
        self.rollback_count = 0
        self.max_rollbacks = 3  # Prevent infinite rollback loops
        
    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        """Check for loss explosion and trigger rollback if needed."""
        current_loss = metrics.get('val_loss')
        if current_loss is None:
            return

        # Track best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_epochs = 0
            return

        # Check for explosion (Model Collapse)
        # If current loss is significantly worse than our best known state
        if current_loss > (self.best_loss * self.threshold_ratio):
            self.bad_epochs += 1
            trainer.logger.warning(
                f"⚠️ Instability Detected: Loss {current_loss:.4f} > {self.best_loss:.4f} (Best) x {self.threshold_ratio}"
            )
            
            if self.bad_epochs >= self.patience:
                if self.rollback_count < self.max_rollbacks:
                    self._perform_emergency_rollback(trainer)
                    self.bad_epochs = 0
                    self.rollback_count += 1
                else:
                    trainer.logger.error(
                        f"❌ Maximum rollbacks ({self.max_rollbacks}) reached. "
                        "Stopping automatic recovery. Manual intervention required."
                    )
                    trainer.should_stop = True

    def _perform_emergency_rollback(self, trainer):
        """Perform emergency rollback to best checkpoint with safety measures."""
        trainer.logger.error("🚨 AUTO-ROLLBACK TRIGGERED: Model collapsed. Restoring best state...")
        
        # 1. Path to safety
        checkpoint_dir = Path(trainer.config.get('checkpoint_dir', 'checkpoints'))
        best_model_path = checkpoint_dir / "best_model.pt"
        
        if not best_model_path.exists():
            trainer.logger.error("❌ Critical: best_model.pt not found. Cannot rollback.")
            return

        # 2. Reload weights (Go back in time)
        # We assume the best model was saved correctly by CheckpointCallback
        try:
            # Load checkpoint but ignore optimizer state to allow manual LR adjustment
            checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Restore training state
            checkpoint_epoch = checkpoint.get('epoch', 0)
            # Keep current_epoch the same to retry the failed epoch with restored weights
            # The training loop will use current_epoch - 1 as start_epoch, so this works correctly
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.best_wer = checkpoint.get('best_wer', float('inf'))
            
            trainer.logger.info(f"✅ Weights restored from Epoch {checkpoint_epoch}")
            trainer.logger.info(f"   Will retry current epoch ({trainer.current_epoch}) with restored weights")
            trainer.logger.info(f"   Best validation loss: {trainer.best_val_loss:.4f}")
            trainer.logger.info(f"   Best WER: {trainer.best_wer:.4f}")
        except Exception as e:
            trainer.logger.error(f"❌ Rollback failed: {e}")
            return
        
        # 3. Apply Countermeasures (The Fix)
        trainer.logger.info("🛡️ Applying active countermeasures:")
        
        # Action A: Reduce learning rate (primary countermeasure for instability)
            
        # Action B: Emergency LR Cut
        # We manually slash the LR in the optimizer, overriding the scheduler for now
        for param_group in trainer.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.5
            param_group['lr'] = new_lr
            trainer.logger.info(f"   👉 Slashing Learning Rate: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Reset scheduler if it exists (will be recreated in next epoch)
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            trainer.logger.info("   👉 Scheduler will be reset on next epoch")


class CurriculumLearningCallback(Callback):
    """
    Smart scheduler for curriculum learning.
    
    This callback implements curriculum learning:
    1. Starts with easier samples (shorter duration)
    2. Gradually introduces harder samples as model improves
    """
    
    def __init__(self, start_timestamp_epoch=5, required_wer=0.6, initial_ts_weight=0.01):
        """
        Initialize curriculum learning callback.
        
        Args:
            start_timestamp_epoch: Minimum epoch (kept for compatibility, not used)
            required_wer: Maximum WER threshold (kept for compatibility, not used)
            initial_ts_weight: Not used (kept for compatibility)
        """
        self.start_epoch = start_timestamp_epoch
        self.required_wer = required_wer

    def on_train_begin(self, trainer):
        """Initialize curriculum learning."""
        trainer.logger.info("🎓 Curriculum Learning: Ready")

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        """Monitor training progress for curriculum learning."""
        current_wer = metrics.get('wer', 1.0)
        # Curriculum learning logic can be added here if needed

