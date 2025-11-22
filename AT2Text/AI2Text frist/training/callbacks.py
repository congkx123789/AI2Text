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
        current_value = metrics.get(self.monitor_metric, None)
        
        if current_value is None:
            return
        
        # Check if this is the best model
        is_best = False
        if self.mode == "min":
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
        
        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch
            
            if self.save_best:
                self._save_checkpoint(trainer, epoch, metrics, "best_model.pt")
        
        # Save periodic checkpoint
        if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
            self._save_checkpoint(
                trainer, epoch, metrics, 
                f"checkpoint_epoch_{epoch}.pt"
            )
    
    def _save_checkpoint(self, trainer, epoch: int, metrics: Dict[str, float], filename: str):
        """Save checkpoint to disk."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_loss': metrics.get('val_loss', float('inf')),
            'best_wer': metrics.get('wer', float('inf')),
            'config': trainer.config,
            'metrics': metrics
        }
        
        # Save scheduler state if exists
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        trainer.logger.info(f"Saved checkpoint: {checkpoint_path}")


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
    
    def __init__(self, log_every_n_batches: int = 10):
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
        """Log batch progress."""
        if batch_idx % self.log_every_n_batches == 0:
            trainer.logger.info(
                f"Batch {batch_idx}/{trainer.current_epoch_batches} - "
                f"Loss: {loss:.4f}"
            )
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Log epoch results."""
        trainer.logger.info(f"\nEpoch {epoch} Results:")
        for metric_name, value in metrics.items():
            trainer.logger.info(f"  {metric_name}: {value:.4f}")


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
        # Store metrics history
        metrics_with_epoch = {**metrics, 'epoch': epoch}
        self.metrics_history.append(metrics_with_epoch)
        
        # Log to database if available
        if hasattr(trainer, 'db') and hasattr(trainer, 'training_run_id'):
            if trainer.training_run_id:
                trainer.db.add_epoch_metrics(
                    training_run_id=trainer.training_run_id,
                    epoch=epoch,
                    train_loss=metrics.get('train_loss', 0.0),
                    val_loss=metrics.get('val_loss', 0.0),
                    learning_rate=metrics.get('learning_rate', 0.0),
                    wer=metrics.get('wer', None),
                    cer=metrics.get('cer', None),
                    epoch_time=metrics.get('epoch_time', None)
                )
    
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

