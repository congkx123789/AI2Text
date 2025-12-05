"""
Smart callbacks for automatic training recovery and curriculum learning.

This module implements advanced callbacks that enable "auto-run" capabilities:
- AutoRollbackCallback: Detects model collapse and automatically rolls back to best checkpoint
- CurriculumLearningCallback: Gradually introduces harder tasks (timestamps) when model is ready
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
    3. Applies safety measures (disables timestamps, reduces learning rate)
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
        
        # Action A: Kill the Timestamp Head (Primary suspect for instability)
        if trainer.use_timestamps:
            trainer.logger.info("   👉 Disabling Timestamp Training (Weight 0.0)")
            trainer.use_timestamps = False  # This flag controls the model forward pass
            trainer.timestamp_loss_weight = 0.0  # This ensures 0 gradient from TS head
            
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
    Smart scheduler that only enables complex tasks (Timestamps) when the model is ready.
    
    This callback implements curriculum learning:
    1. Starts with timestamps disabled (pure ASR training)
    2. Enables timestamps only when WER is below threshold and minimum epochs passed
    3. Gradually increases timestamp loss weight
    """
    
    def __init__(self, start_timestamp_epoch=5, required_wer=0.6, initial_ts_weight=0.01):
        """
        Initialize curriculum learning callback.
        
        Args:
            start_timestamp_epoch: Minimum epoch before enabling timestamps
            required_wer: Maximum WER threshold to enable timestamps
            initial_ts_weight: Starting weight for timestamp loss when enabled
        """
        self.start_epoch = start_timestamp_epoch
        self.required_wer = required_wer
        self.target_ts_weight = initial_ts_weight
        self.timestamps_enabled = False
        self.original_ts_weight = None  # Store original weight from config

    def on_train_begin(self, trainer):
        """Force disable timestamps at start for stability."""
        # Store original weight from config
        self.original_ts_weight = trainer.config.get('timestamp_loss_weight', 0.1)
        
        # Always start with timestamps OFF to ensure ASR convergence first
        if trainer.use_timestamps:
            trainer.logger.info("🎓 Curriculum: Starting with Timestamps DISABLED (Stability Mode)")
            trainer.use_timestamps = False
            self.timestamps_enabled = False
            # Store the intended weight to restore later
            self.target_ts_weight = self.original_ts_weight
            trainer.timestamp_loss_weight = 0.0
        else:
            # If timestamps were already disabled, respect that
            trainer.logger.info("🎓 Curriculum: Timestamps already disabled, will enable when ready")
            self.target_ts_weight = self.original_ts_weight

    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        """Check if ready to enable timestamps and gradually increase weight."""
        current_wer = metrics.get('wer', 1.0)
        
        # Phase 1: Check if ready to enable
        if not self.timestamps_enabled:
            # Criteria: Minimum epochs passed AND WER is decent (< required_wer)
            if epoch >= self.start_epoch and current_wer < self.required_wer:
                trainer.logger.info(
                    f"🎓 Curriculum: Level Up! Enabling Timestamps "
                    f"(Epoch {epoch}, WER {current_wer:.2f} < {self.required_wer})"
                )
                trainer.use_timestamps = True
                trainer.timestamp_loss_weight = 0.01  # Start gentle
                self.timestamps_enabled = True
                
        # Phase 2: Gradually ramp up difficulty
        elif self.timestamps_enabled:
            # Slowly increase TS weight back to target
            if trainer.timestamp_loss_weight < self.target_ts_weight:
                new_weight = min(self.target_ts_weight, trainer.timestamp_loss_weight * 1.5)
                if new_weight > trainer.timestamp_loss_weight:
                    trainer.logger.info(
                        f"🎓 Curriculum: Increasing Timestamp Weight "
                        f"{trainer.timestamp_loss_weight:.4f} -> {new_weight:.4f}"
                    )
                    trainer.timestamp_loss_weight = new_weight

