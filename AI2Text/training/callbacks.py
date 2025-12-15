"""
Training callbacks for ASR model.

This module implements callback functions for training lifecycle events:
- Checkpoint saving
- Early stopping
- Learning rate scheduling
- Logging
- Metrics tracking
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging
from utils.logger import setup_logger


class Callback(ABC):
    """Base callback class for training events."""
    
    def on_train_begin(self, trainer):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Called at the end of each batch."""
        pass


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints.
    
    Saves checkpoints at specified intervals and maintains the best model
    based on validation loss.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 save_best: bool = True,
                 save_every_n_epochs: int = 5,
                 monitor_metric: str = "val_loss",
                 mode: str = "min"):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            save_best (bool): Save best model based on monitor_metric
            save_every_n_epochs (int): Save checkpoint every N epochs
            monitor_metric (str): Metric to monitor for best model
            mode (str): "min" or "max" - whether lower or higher is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint at end of epoch."""
        try:
            trainer.logger.debug(f"CheckpointCallback.on_epoch_end called for epoch {epoch}")
            current_value = metrics.get(self.monitor_metric, None)
            
            # Check if this is the best model (only if monitor_metric exists)
            if current_value is not None:
                is_best = False
                if self.mode == "min":
                    is_best = current_value < self.best_value
                else:
                    is_best = current_value > self.best_value
                
                if is_best:
                    self.best_value = current_value
                    self.best_epoch = epoch
                    
                    if self.save_best:
                        trainer.logger.info(f"🏆 New best {self.monitor_metric}: {current_value:.6f} (epoch {epoch})")
                        self._save_checkpoint(trainer, epoch, metrics, "best_model.pt")
            else:
                # Warn if monitor_metric is missing (but still save periodic checkpoints)
                trainer.logger.warning(
                    f"⚠️  {self.monitor_metric} not found in metrics. "
                    f"Best model tracking disabled, but periodic checkpoints will still be saved."
                )
            
            # Save checkpoint every epoch (if save_every_n_epochs = 1) or periodically
            # Always save periodic checkpoints even if monitor_metric is missing
            if self.save_every_n_epochs > 0:
                if epoch % self.save_every_n_epochs == 0:
                    trainer.logger.info(f"💾 Saving periodic checkpoint for epoch {epoch} (save_every={self.save_every_n_epochs})...")
                    self._save_checkpoint(
                        trainer, epoch, metrics, 
                        f"checkpoint_epoch_{epoch}.pt"
                    )
                else:
                    trainer.logger.debug(f"Skipping checkpoint for epoch {epoch} (not a multiple of {self.save_every_n_epochs})")
            else:
                trainer.logger.warning(f"⚠️  Checkpoint saving disabled (save_every_n_epochs={self.save_every_n_epochs})")
        except Exception as e:
            trainer.logger.error(f"❌ Error in CheckpointCallback.on_epoch_end for epoch {epoch}: {e}", exc_info=True)
            raise
    
    def _save_checkpoint(self, trainer, epoch: int, metrics: Dict[str, float], filename: str):
        """Save checkpoint to disk."""
        try:
            # Ensure directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_loss': metrics.get('val_loss', float('inf')),
                'best_wer': metrics.get('wer', float('inf')),
                'best_cer': metrics.get('cer', float('inf')) if 'cer' in metrics else float('inf'),
                'config': trainer.config,
                'metrics': metrics,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                'global_step': getattr(trainer, 'global_step', epoch * len(trainer.train_loader))
            }
            
            # Save scheduler state if exists
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Save checkpoint with error handling
            torch.save(checkpoint, checkpoint_path)
            
            # Verify file was actually saved
            if checkpoint_path.exists():
                file_size = checkpoint_path.stat().st_size / (1024**2)  # MB
                trainer.logger.info(f"✅ Saved checkpoint: {checkpoint_path} ({file_size:.1f} MB)")
            else:
                trainer.logger.error(f"❌ Failed to save checkpoint: {checkpoint_path} (file not found after save)")
                
        except Exception as e:
            trainer.logger.error(f"❌ Error saving checkpoint {filename}: {e}", exc_info=True)
            raise


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metrics.
    
    Stops training if the monitored metric doesn't improve for a specified
    number of epochs (patience).
    """
    
    def __init__(self,
                 monitor_metric: str = "val_loss",
                 patience: int = 10,
                 mode: str = "min",
                 min_delta: float = 0.0):
        """
        Initialize early stopping callback.
        
        Args:
            monitor_metric (str): Metric to monitor
            patience (int): Number of epochs to wait before stopping
            mode (str): "min" or "max" - whether lower or higher is better
            min_delta (float): Minimum change to qualify as improvement
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.wait_count = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor_metric, None)
        
        if current_value is None:
            return
        
        # Check for improvement
        improved = False
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Stop if patience exceeded
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            trainer.logger.info(
                f"Early stopping triggered after epoch {epoch}. "
                f"Best {self.monitor_metric}: {self.best_value:.4f}"
            )


class LoggingCallback(Callback):
    """
    Callback for logging training progress.
    
    Logs metrics to console and file at specified intervals.
    """
    
    def __init__(self, log_every_n_batches: int = 50):
        """
        Initialize logging callback.
        
        Args:
            log_every_n_batches (int): Log every N batches
        """
        self.log_every_n_batches = log_every_n_batches
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Log epoch start."""
        trainer.logger.info(f"\n{'='*60}")
        trainer.logger.info(f"Epoch {epoch}/{trainer.num_epochs}")
        trainer.logger.info(f"{'='*60}")
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Log batch progress to file only (not console to avoid tqdm conflict)."""
        if batch_idx % self.log_every_n_batches == 0:
            # Log only to file handlers to avoid interfering with tqdm progress bar
            # Create a record and emit only to file handlers
            record = logging.LogRecord(
                name=trainer.logger.name,
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=f"Batch {batch_idx}/{trainer.current_epoch_batches} - Loss: {loss:.4f}",
                args=(),
                exc_info=None
            )
            # Only emit to file handlers
            for handler in trainer.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(record)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log epoch results."""
        trainer.logger.info(f"\nEpoch {epoch} Results:")
        for metric_name, value in metrics.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    trainer.logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    trainer.logger.info(f"  {metric_name}: {value}")
            else:
                trainer.logger.info(f"  {metric_name}: N/A")


class MetricsCallback(Callback):
    """
    Callback for tracking and logging metrics.
    
    Calculates and logs WER, CER, and other metrics during training.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize metrics callback.
        
        Args:
            log_every_n_epochs (int): Log metrics every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.metrics_history = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Track and log metrics."""
        # Store metrics history (file-based logging, no database)
        metrics_with_epoch = {**metrics, 'epoch': epoch}
        self.metrics_history.append(metrics_with_epoch)
    
    def get_metrics_history(self):
        """Get all tracked metrics."""
        return self.metrics_history


class CallbackManager:
    """
    Manages multiple callbacks and executes them at appropriate times.
    
    Coordinates all callbacks during training lifecycle.
    """
    
    def __init__(self, callbacks: list = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks (list): List of Callback instances
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer):
        """Execute all callbacks on training begin."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        """Execute all callbacks on training end."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Execute all callbacks on epoch begin."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Execute all callbacks on epoch end."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer, batch_idx: int):
        """Execute all callbacks on batch begin."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Execute all callbacks on batch end."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)

