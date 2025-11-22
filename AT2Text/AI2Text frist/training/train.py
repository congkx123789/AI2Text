"""
Main training script for ASR model.
Optimized for resource-constrained environments.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import time
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from training.callbacks import (
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricsCallback
)
from utils.metrics import calculate_wer, calculate_cer
from utils.logger import setup_logger


class ASRTrainer:
    """Trainer class for ASR model with optimizations for weak hardware."""
    
    def __init__(self, config: dict, db: ASRDatabase):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
            db: Database instance
        """
        self.config = config
        self.db = db
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('ASRTrainer', config.get('log_file', 'training.log'))
        
        # Setup components
        self._setup_preprocessing()
        self._setup_model()
        self._setup_optimization()
        
        # Training state
        self.current_epoch = 0
        self.current_epoch_batches = 0
        self.num_epochs = 0
        self.best_val_loss = float('inf')
        self.best_wer = float('inf')
        self.training_run_id = None
        self.should_stop = False
        
        # Setup callbacks (following Training Layer architecture)
        self._setup_callbacks()
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_trainable_params():,}")
    
    def _setup_callbacks(self):
        """Setup training callbacks following the Training Layer architecture."""
        self.callback_manager = CallbackManager()
        
        # Checkpoint callback - saves model checkpoints
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            save_best=True,
            save_every_n_epochs=self.config.get('save_every', 5),
            monitor_metric='val_loss',
            mode='min'
        )
        
        # Early stopping callback - stops if no improvement
        if self.config.get('early_stopping', {}).get('enabled', False):
            early_stop_callback = EarlyStoppingCallback(
                monitor_metric='val_loss',
                patience=self.config.get('early_stopping', {}).get('patience', 10),
                mode='min',
                min_delta=self.config.get('early_stopping', {}).get('min_delta', 0.0)
            )
            self.callback_manager.add_callback(early_stop_callback)
        
        # Logging callback - logs training progress
        logging_callback = LoggingCallback(
            log_every_n_batches=self.config.get('log_every_n_batches', 10)
        )
        
        # Metrics callback - tracks and logs metrics
        metrics_callback = MetricsCallback(
            log_every_n_epochs=1
        )
        
        # Add all callbacks
        self.callback_manager.add_callback(checkpoint_callback)
        self.callback_manager.add_callback(logging_callback)
        self.callback_manager.add_callback(metrics_callback)
    
    def _setup_preprocessing(self):
        """Setup preprocessing components."""
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mels=self.config.get('n_mels', 80)
        )
        
        self.augmenter = AudioAugmenter(
            sample_rate=self.config.get('sample_rate', 16000)
        )
        
        self.tokenizer = Tokenizer()
        self.normalizer = VietnameseTextNormalizer()
    
    def _setup_model(self):
        """Setup model and move to device."""
        self.model = ASRModel(
            input_dim=self.config.get('n_mels', 80),
            vocab_size=len(self.tokenizer),
            d_model=self.config.get('d_model', 256),
            num_encoder_layers=self.config.get('num_encoder_layers', 6),
            num_heads=self.config.get('num_heads', 4),
            d_ff=self.config.get('d_ff', 1024),
            dropout=self.config.get('dropout', 0.1)
        )
        
        self.model.to(self.device)
        
        # Use mixed precision training for efficiency
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _setup_optimization(self):
        """Setup optimizer and loss function."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # CTC Loss
        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.blank_token_id,
            zero_infinity=True
        )
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('learning_rate', 1e-4),
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            # Move to device
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, output_lengths = self.model(audio_features, audio_lengths)
                    
                    # CTC loss expects (T, N, C) format
                    logits = logits.transpose(0, 1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.get('grad_clip', 1.0))
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, output_lengths = self.model(audio_features, audio_lengths)
                
                # CTC loss
                logits = logits.transpose(0, 1)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.get('grad_clip', 1.0))
                self.optimizer.step()
            
            # Update scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Training Layer: Callback on_batch_end (Logging)
            self.callback_manager.on_batch_end(self, num_batches - 1, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader) -> tuple:
        """Validate the model.
        
        Returns:
            avg_loss: Average validation loss
            wer: Word error rate
            cer: Character error rate
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(val_loader, desc='Validation'):
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            
            # Forward pass
            logits, output_lengths = self.model(audio_features, audio_lengths)
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Decode predictions for WER/CER calculation
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                # Decode using CTC collapse (remove blanks and duplicates)
                pred_text = self._ctc_decode(pred_tokens)
                ref_text = self.tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
        
        avg_loss = total_loss / num_batches
        wer = calculate_wer(all_references, all_predictions)
        cer = calculate_cer(all_references, all_predictions)
        
        return avg_loss, wer, cer
    
    def _ctc_decode(self, tokens: list) -> str:
        """Simple CTC greedy decoding.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            decoded_text: Decoded text
        """
        # Remove consecutive duplicates
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
                prev = token
        
        # Remove blank tokens
        filtered = [t for t in collapsed if t != self.tokenizer.blank_token_id]
        
        # Decode to text
        return self.tokenizer.decode(filtered)
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """
        Main training loop following the Training Layer architecture.
        
        This method orchestrates the complete training process using the layered
        architecture: Data → Preprocessing → Model → Training → Evaluation.
        
        Args:
            train_loader: Training data loader (from Preprocessing Layer)
            val_loader: Validation data loader (from Preprocessing Layer)
            num_epochs: Number of epochs to train
        """
        self.num_epochs = num_epochs
        self.current_epoch_batches = len(train_loader)
        
        # Setup scheduler (part of Optimizer in Training Layer)
        total_steps = len(train_loader) * num_epochs
        self._setup_scheduler(total_steps)
        
        # Create training run in database (Data Layer)
        model_id = self.db.add_model(
            model_name=self.config.get('model_name', 'ASR_Base'),
            model_type='transformer_ctc',
            architecture='encoder_ctc',
            version='1.0',
            config=self.config,
            total_parameters=self.model.get_num_trainable_params()
        )
        
        self.training_run_id = self.db.create_training_run(
            model_id=model_id,
            run_name=self.config.get('run_name', f'run_{int(time.time())}'),
            config=self.config,
            batch_size=self.config.get('batch_size', 16),
            learning_rate=self.config.get('learning_rate', 1e-4),
            num_epochs=num_epochs,
            optimizer='AdamW',
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        )
        
        start_time = time.time()
        
        # Training Layer: Callback on_train_begin
        self.callback_manager.on_train_begin(self)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Training Layer: Callback on_epoch_begin
            self.callback_manager.on_epoch_begin(self, self.current_epoch)
            
            # Training Layer: Train epoch (Trainer)
            train_loss = self.train_epoch(train_loader)
            
            # Evaluation Layer: Validate (Metrics calculation)
            val_loss, wer, cer = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update best metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_wer = wer
            
            # Prepare metrics dictionary for callbacks
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'wer': wer,
                'cer': cer,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            # Training Layer: Callback on_epoch_end (Checkpoints, Logging, Metrics)
            self.callback_manager.on_epoch_end(self, self.current_epoch, metrics)
            
            # Check early stopping
            if self.should_stop:
                self.logger.info(f'Early stopping triggered at epoch {self.current_epoch}')
                break
        
        total_time = time.time() - start_time
        
        # Complete training run in database (Data Layer)
        self.db.complete_training_run(
            run_id=self.training_run_id,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            best_val_loss=self.best_val_loss,
            best_epoch=self.current_epoch,
            wer=self.best_wer,
            cer=cer,
            total_time=total_time
        )
        
        # Training Layer: Callback on_train_end
        self.callback_manager.on_train_end(self)
        
        self.logger.info(f'Training completed in {total_time:.2f}s')
        self.logger.info(f'Best validation loss: {self.best_val_loss:.4f}')
        self.logger.info(f'Best WER: {self.best_wer:.4f}')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']


def main():
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load data
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    val_df = db.get_split_data('val', config.get('split_version', 'v1'))
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create data loaders
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()
    tokenizer = Tokenizer()
    
    train_loader, val_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter
    )
    
    # Initialize trainer
    trainer = ASRTrainer(config, db)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    trainer.train(train_loader, val_loader, config.get('num_epochs', 50))


if __name__ == '__main__':
    main()

