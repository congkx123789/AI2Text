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
import multiprocessing

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_with_timestamps import ASRModelWithTimestamps, create_timestamp_targets
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
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
    """Trainer class for ASR model with timestamp support and bilingual (English + Vietnamese) training."""
    
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
        
        # Timestamp training config
        self.use_timestamps = self.config.get('use_timestamps', True)
        self.timestamp_loss_weight = self.config.get('timestamp_loss_weight', 0.1)
        self.subsampling_factor = 4  # Model subsamples by 4x
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.hop_length = 160  # Default hop length for mel spectrogram
        
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
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
            self.logger.info(f"Effective batch size: {self.config.get('batch_size', 16) * self.gradient_accumulation_steps}")
        
        # Setup callbacks (following Training Layer architecture)
        self._setup_callbacks()
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_trainable_params():,}")
        self.logger.info(f"Timestamp training: {self.use_timestamps}")
        self.logger.info(f"Bilingual support: English + Vietnamese")
        if self.use_timestamps:
            self.logger.info(f"Timestamp loss weight: {self.timestamp_loss_weight}")
    
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
        
        # Use aggressive augmentation for harder training (prevents overfitting)
        aggressive_aug = self.config.get('aggressive_augmentation', True)
        self.augmenter = AudioAugmenter(
            sample_rate=self.config.get('sample_rate', 16000),
            aggressive=aggressive_aug
        )

        # Select tokenizer type (character-level or BPE)
        tokenizer_type = self.config.get('tokenizer_type', 'char')
        if tokenizer_type == 'bpe':
            from preprocessing.bpe_tokenizer import BPETokenizer
            bpe_path = self.config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json')
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(bpe_path)
            self.logger.info(f"✅ Using BPE tokenizer: {bpe_path} ({len(self.tokenizer)} tokens)")
        else:
            self.tokenizer = Tokenizer()
            self.logger.info(f"✅ Using character-level tokenizer ({len(self.tokenizer)} tokens)")

        # Use bilingual normalizer to support Vietnamese + English
        self.normalizer = BilingualTextNormalizer()
    
    def _setup_model(self):
        """Setup model with timestamp support and move to device."""
        self.model = ASRModelWithTimestamps(
            input_dim=self.config.get('n_mels', 80),
            vocab_size=len(self.tokenizer),
            d_model=self.config.get('d_model', 1024),
            num_encoder_layers=self.config.get('num_encoder_layers', 24),
            num_heads=self.config.get('num_heads', 16),
            d_ff=self.config.get('d_ff', 4096),
            dropout=self.config.get('dropout', 0.1),
            predict_timestamps=self.use_timestamps
        )
        
        self.model.to(self.device)
        
        # PyTorch 2.0+ torch.compile() for faster training
        self.use_compile = self.config.get('use_compile', False) and hasattr(torch, 'compile')
        if self.use_compile:
            compile_mode = self.config.get('compile_mode', 'reduce-overhead')
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                self.logger.info(f"torch.compile() enabled with mode: {compile_mode}")
            except Exception as e:
                self.logger.warning(f"torch.compile() failed: {e}")
                self.use_compile = False
        
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
        
        # CTC Loss for transcription
        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.blank_token_id,
            zero_infinity=True,
            reduction='mean'
        )
        
        # MSE Loss for timestamps
        self.timestamp_criterion = nn.MSELoss(reduction='mean')
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        max_lr = self.config.get('learning_rate', 1e-4)
        warmup_pct = self.config.get('warmup_pct', 0.2)
        
        div_factor = 100.0
        final_div_factor = 10000.0
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        
        initial_lr = max_lr / div_factor
        self.logger.info(f"Learning rate scheduler: Max={max_lr:.2e}, Initial={initial_lr:.2e}, Warmup={warmup_pct*100:.0f}%")
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch with timestamp support.
        
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0
        total_ctc_loss = 0
        total_timestamp_loss = 0
        num_batches = 0
        
        # Get batch size info for progress bar
        config_batch_size = self.config.get('batch_size', 16)
        effective_batch_size = config_batch_size * self.gradient_accumulation_steps
        
        pbar = tqdm(train_loader, 
                   desc=f'Epoch {self.current_epoch} (batch={config_batch_size}, effective={effective_batch_size})', 
                   file=sys.stderr, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio_features = batch['audio_features'].to(self.device, non_blocking=True)
            audio_lengths = batch['audio_lengths'].to(self.device, non_blocking=True)
            text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
            text_lengths = batch['text_lengths'].to(self.device, non_blocking=True)
            
            # Get word timestamps if available
            word_timestamps_list = batch.get('word_timestamps', [None] * len(audio_features))
            
            # Zero gradients only at the start of accumulation cycle
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, output_lengths, timestamps = self.model(
                        audio_features, audio_lengths, return_timestamps=self.use_timestamps
                    )
                    
                    # CTC loss
                    logits_t = logits.transpose(0, 1)
                    log_probs = torch.log_softmax(logits_t, dim=-1)
                    ctc_loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                    
                    # Timestamp loss
                    timestamp_loss = torch.tensor(0.0, device=self.device)
                    if self.use_timestamps and timestamps is not None:
                        timestamp_targets = []
                        valid_samples = []
                        
                        for i in range(len(audio_features)):
                            if word_timestamps_list[i] is not None:
                                target = create_timestamp_targets(
                                    word_timestamps_list[i],
                                    output_lengths[i].item(),
                                    self.subsampling_factor,
                                    self.sample_rate,
                                    self.hop_length
                                ).to(self.device)
                                
                                actual_len = output_lengths[i].item()
                                if target.shape[0] >= actual_len:
                                    target = target[:actual_len]
                                else:
                                    pad = torch.zeros(actual_len - target.shape[0], 2, device=self.device)
                                    target = torch.cat([target, pad], dim=0)
                                
                                timestamp_targets.append(target)
                                valid_samples.append(i)
                        
                        if len(timestamp_targets) > 0:
                            max_len = max(t.shape[0] for t in timestamp_targets)
                            padded_targets = []
                            for i, target in enumerate(timestamp_targets):
                                if target.shape[0] < max_len:
                                    pad = torch.zeros(max_len - target.shape[0], 2, device=self.device)
                                    target = torch.cat([target, pad], dim=0)
                                padded_targets.append(target)
                            
                            targets_tensor = torch.stack(padded_targets)
                            pred_timestamps = timestamps[valid_samples, :max_len, :]
                            timestamp_loss = self.timestamp_criterion(pred_timestamps, targets_tensor)
                    
                    # Combined loss
                    loss = ctc_loss + self.timestamp_loss_weight * timestamp_loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Batch {batch_idx}: Loss is NaN/Inf, skipping")
                        continue
                    
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config.get('grad_clip', 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Forward pass without AMP
                logits, output_lengths, timestamps = self.model(
                    audio_features, audio_lengths, return_timestamps=self.use_timestamps
                )
                
                logits_t = logits.transpose(0, 1)
                log_probs = torch.log_softmax(logits_t, dim=-1)
                ctc_loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                
                timestamp_loss = torch.tensor(0.0, device=self.device)
                if self.use_timestamps and timestamps is not None:
                    timestamp_targets = []
                    valid_samples = []
                    
                    for i in range(len(audio_features)):
                        if word_timestamps_list[i] is not None:
                            target = create_timestamp_targets(
                                word_timestamps_list[i],
                                output_lengths[i].item(),
                                self.subsampling_factor,
                                self.sample_rate,
                                self.hop_length
                            ).to(self.device)
                            
                            actual_len = output_lengths[i].item()
                            if target.shape[0] >= actual_len:
                                target = target[:actual_len]
                            else:
                                pad = torch.zeros(actual_len - target.shape[0], 2, device=self.device)
                                target = torch.cat([target, pad], dim=0)
                            
                            timestamp_targets.append(target)
                            valid_samples.append(i)
                    
                    if len(timestamp_targets) > 0:
                        max_len = max(t.shape[0] for t in timestamp_targets)
                        padded_targets = []
                        for i, target in enumerate(timestamp_targets):
                            if target.shape[0] < max_len:
                                pad = torch.zeros(max_len - target.shape[0], 2, device=self.device)
                                target = torch.cat([target, pad], dim=0)
                            padded_targets.append(target)
                        
                        targets_tensor = torch.stack(padded_targets)
                        pred_timestamps = timestamps[valid_samples, :max_len, :]
                        timestamp_loss = self.timestamp_criterion(pred_timestamps, targets_tensor)
                
                loss = ctc_loss + self.timestamp_loss_weight * timestamp_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"Batch {batch_idx}: Loss is NaN/Inf, skipping")
                    continue
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config.get('grad_clip', 1.0))
                    self.optimizer.step()
            
            # Update scheduler (every step, not every accumulation step)
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            true_loss = loss.item() * self.gradient_accumulation_steps
            true_ctc = ctc_loss.item()
            true_ts = timestamp_loss.item() if isinstance(timestamp_loss, torch.Tensor) else 0.0
            
            total_loss += true_loss
            total_ctc_loss += true_ctc
            total_timestamp_loss += true_ts
            num_batches += 1
            
            self.callback_manager.on_batch_end(self, num_batches - 1, loss.item())
            pbar.set_postfix({'loss': true_loss, 'ctc': true_ctc, 'ts': true_ts})
        
        avg_loss = total_loss / num_batches
        avg_ctc = total_ctc_loss / num_batches
        avg_ts = total_timestamp_loss / num_batches
        
        self.logger.info(f"Epoch {self.current_epoch} - Loss: {avg_loss:.4f} (CTC: {avg_ctc:.4f}, TS: {avg_ts:.4f})")
        
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
            
            # Forward pass with timestamps
            logits, output_lengths, timestamps = self.model(
                audio_features, audio_lengths, return_timestamps=self.use_timestamps
                )
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"Val batch {num_batches}: Loss is NaN/Inf, skipping")
                continue
            
            total_loss += loss.item()
            
            # Decode predictions for WER/CER calculation
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                pred_text = self._ctc_decode(pred_tokens)
                ref_text = self.tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
            
            # Increment AFTER processing the batch
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        wer = calculate_wer(all_references, all_predictions)
        cer = calculate_cer(all_references, all_predictions)
        
        # Log summary
        empty_preds = sum(1 for p in all_predictions if len(p.strip()) == 0)
        if empty_preds > 0:
            self.logger.warning(f"Validation: {empty_preds}/{len(all_predictions)} empty predictions")
        
        if wer >= 0.95:
            self.logger.error(f"WER >= 0.95 (Model not learning!) - Check predictions")
        
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
        # CRITICAL FIX: Calculate remaining steps if resuming from checkpoint
        remaining_epochs = num_epochs - self.current_epoch + 1  # +1 because current_epoch is next to train
        if remaining_epochs <= 0:
            remaining_epochs = num_epochs  # Safety: if somehow negative, use full epochs
        total_steps = len(train_loader) * remaining_epochs
        self._setup_scheduler(total_steps)
        
        # If resuming, fast-forward scheduler to correct step
        if self.current_epoch > 1:
            # Calculate steps already completed
            steps_completed = len(train_loader) * (self.current_epoch - 1)
            # Fast-forward scheduler to current position
            for _ in range(min(steps_completed, total_steps)):
                try:
                    self.scheduler.step()
                except:
                    break
            self.logger.info(f"Fast-forwarded scheduler to step {steps_completed} (epoch {self.current_epoch})")
        
        # Create training run in database (Data Layer)
        model_id = self.db.add_model(
            model_name=self.config.get('model_name', 'ASR_Base'),
            model_type='transformer_ctc',
            architecture='encoder_ctc',
            version='1.0',
            config=self.config,
            total_parameters=self.model.get_num_trainable_params()
        )
        
        base_run_name = self.config.get('run_name', f'run_{int(time.time())}')
        run_name = f"{base_run_name}_{int(time.time())}"
        self.training_run_id = self.db.create_training_run(
            model_id=model_id,
            run_name=run_name,
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
        
        # CRITICAL FIX: Start from current_epoch if resuming, otherwise start from 0
        start_epoch = self.current_epoch - 1 if self.current_epoch > 0 else 0
        for epoch in range(start_epoch, num_epochs):
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
        """Load model checkpoint.
        
        Supports seamless continuation across different datasets (47k -> 257k).
        When resuming for fine-tuning, the scheduler will be recreated with new
        total_steps based on the new dataset size. This is intentional for
        curriculum learning scenarios.
        """
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state (handles missing keys gracefully for architecture changes)
        model_state = checkpoint['model_state_dict']
        current_model_state = self.model.state_dict()
        
        # Filter out keys that don't exist in current model (e.g., if timestamp head was added/removed)
        filtered_state = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in model_state.items():
            if key in current_model_state:
                if current_model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    unexpected_keys.append(f"{key} (shape mismatch: {current_model_state[key].shape} vs {value.shape})")
            else:
                missing_keys.append(key)
        
        # Load the filtered state
        self.model.load_state_dict(filtered_state, strict=False)
        
        if missing_keys:
            self.logger.info(f"⚠️  Missing keys (will use random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    self.logger.info(f"   - {key}")
        
        if unexpected_keys:
            self.logger.info(f"⚠️  Shape mismatches (will use random init): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    self.logger.info(f"   - {key}")
        
        # Load optimizer state (may have different shapes if dataset size changed)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load optimizer state (will use fresh optimizer): {e}")
            self.logger.info("   This is normal when switching datasets - optimizer will reinitialize")
        
        # CRITICAL FIX: Resume from the next epoch, not epoch 0
        checkpoint_epoch = checkpoint.get('epoch', 0)
        self.current_epoch = checkpoint_epoch + 1  # Continue from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_wer = checkpoint.get('best_wer', float('inf'))
        
        self.logger.info(f"✅ Loaded checkpoint from epoch {checkpoint_epoch}")
        self.logger.info(f"   Resuming training from epoch {self.current_epoch}")
        self.logger.info(f"   Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"   Best WER: {self.best_wer:.4f}")
        self.logger.info(f"   ✅ Can continue training with new dataset size (47k -> 257k)")
        
        # Note: Scheduler will be recreated in train() method with correct total_steps
        # This ensures proper LR scheduling for the remaining epochs


def main():
    # Set multiprocessing start method for better CPU core utilization
    # 'spawn' is more compatible but 'fork' is faster on Linux
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            # Already set, continue
            pass
    
    parser = argparse.ArgumentParser(description='Train ASR model with timestamp support and bilingual (English + Vietnamese) training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--language', type=str, default=None,
                       help='Filter training data by language (e.g., "en" or "vi"). '
                            'Useful for sequential training: train English first, then Vietnamese. '
                            'If not specified, trains on both English and Vietnamese.')
    parser.add_argument('--use_timestamps', action='store_true', default=None,
                       help='Enable timestamp training (overrides config)')
    parser.add_argument('--no_timestamps', action='store_true',
                       help='Disable timestamp training (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override timestamp config from command line
    if args.use_timestamps:
        config['use_timestamps'] = True
    elif args.no_timestamps:
        config['use_timestamps'] = False
    # If not specified, use config default (defaults to True)
    if 'use_timestamps' not in config:
        config['use_timestamps'] = True
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load data (with optional language filtering)
    language = args.language or config.get('language_filter', None)
    train_df = db.get_split_data('train', config.get('split_version', 'v1'), language=language)
    val_df = db.get_split_data('val', config.get('split_version', 'v1'), language=language)
    
    if language:
        print(f"Training samples ({language}): {len(train_df)}")
        print(f"Validation samples ({language}): {len(val_df)}")
    else:
        print(f"Training samples (all languages - English + Vietnamese): {len(train_df)}")
        print(f"Validation samples (all languages - English + Vietnamese): {len(val_df)}")
    
    # Log timestamp configuration
    use_timestamps = config.get('use_timestamps', True)
    print(f"\n{'='*80}")
    print(f"📊 TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Timestamp training: {'✅ Enabled' if use_timestamps else '❌ Disabled'}")
    if use_timestamps:
        print(f"Timestamp loss weight: {config.get('timestamp_loss_weight', 0.1)}")
    print(f"Dataset: {len(train_df)} train, {len(val_df)} val samples")
    print(f"Batch size: {config.get('batch_size', 16)}")
    print(f"Epochs: {config.get('num_epochs', 50)}")
    print(f"{'='*80}\n")
    
    # Create data loaders
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()

    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
        print(f"✅ Using BPE tokenizer: {bpe_path} ({len(tokenizer)} tokens)")
    else:
        tokenizer = Tokenizer()
        print(f"✅ Using character-level tokenizer ({len(tokenizer)} tokens)")
    
    train_loader, val_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter,
        persistent_workers=config.get('persistent_workers', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        sort_by_length=config.get('sort_by_length', True),  # Enable length sorting
        use_bucketing=config.get('use_bucketing', False),  # Optional: use bucketing sampler
        num_buckets=config.get('num_buckets', 10),
        cache_in_ram=config.get('cache_in_ram', False)  # Cache data in RAM (reduces CPU load)
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

