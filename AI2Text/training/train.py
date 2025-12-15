"""
Training script for ASR model.

OPTIMIZED FOR:
- RTX 5060TI 16GB VRAM: Gradient checkpointing, mixed precision, efficient batching
- Ryzen 9 9990X: Multi-core data loading, optimized CPU operations
- 64GB RAM: Large batch sizes, better caching
- SSD 3000MB/s: Fast I/O throughput

Features:
- Seq2Seq Transformer architecture with encoder-decoder
- Bilingual support (Vietnamese + English)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling with warmup
- Automatic checkpointing
- Auto-rollback on model collapse
- Curriculum learning
- WER/CER metrics calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import os
import time
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
from training.dataset import create_data_loaders
from training.callbacks import (
    CallbackManager, CheckpointCallback, EarlyStoppingCallback,
    LoggingCallback, MetricsCallback
)
from training.smart_callbacks import AutoRollbackCallback, CurriculumLearningCallback
from utils.logger import setup_logger
from utils.manifest_loader import load_merged_dataset
from utils.metrics import calculate_wer, calculate_cer
from utils.ctc_loss import CTCLoss
from training.scheduled_sampling import ScheduledSampling


class ASRTrainer:
    """Trainer for ASR model with full training loop."""
    
    def __init__(self, config: Dict):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logger
        log_file = config.get('log_file', 'logs/training.log')
        self.logger = setup_logger('ASRTrainer', log_file)
        
        # Training state
        self.current_epoch = 0
        self.current_epoch_batches = 0
        self.global_step = 0  # Track total training steps for scheduler logic
        self.should_stop = False
        self.best_val_loss = float('inf')
        self.best_wer = float('inf')
        self.best_cer = float('inf')
        self.num_epochs = config.get('num_epochs', 50)  # For callbacks
        
        # Setup components
        self._setup_components()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        # Use bfloat16 for better numerical stability (RTX 5060TI supports it)
        self.amp_dtype = torch.bfloat16 if config.get('use_bf16', True) and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler() if self.use_amp else None
        
        self.logger.info("=" * 60)
        self.logger.info("ASR Trainer Initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_amp}")
        if self.use_amp:
            dtype_str = "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16"
            self.logger.info(f"AMP Dtype: {dtype_str}")
        self.logger.info("=" * 60)
    
    def _setup_components(self):
        """Setup model, tokenizer, audio processor, and data loaders."""
        # Load tokenizer
        tokenizer_type = self.config.get('tokenizer_type', 'sentencepiece')
        if tokenizer_type == 'sentencepiece':
            tokenizer_path = self.config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
            self.tokenizer = SentencePieceTokenizer(tokenizer_path)
            vocab_size = self.config.get('vocab_size', 3500)
        else:
            # Fallback to BPE tokenizer
            from preprocessing.bpe_tokenizer import BPETokenizer
            tokenizer_path = self.config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
            self.tokenizer = BPETokenizer(tokenizer_path)
            vocab_size = len(self.tokenizer)
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mels=self.config.get('n_mels', 80),
            n_fft=self.config.get('n_fft', 400),
            hop_length=self.config.get('hop_length', 160),
            win_length=self.config.get('win_length', 400)
        )
        
        # Audio augmenter (for training)
        self.augmenter = AudioAugmenter(
            sample_rate=self.config.get('sample_rate', 16000),
            aggressive=True  # Use aggressive augmentation for harder training
        )
        
        # Load data
        dataset_root = self.config.get('dataset_root', 'data/processed/full_merged_dataset')
        language_filter = self.config.get('language_filter', None)
        
        train_df = load_merged_dataset('train', dataset_root, language=language_filter)
        val_df = load_merged_dataset('val', dataset_root, language=language_filter)
        
        self.logger.info(f"Training samples: {len(train_df):,}")
        self.logger.info(f"Validation samples: {len(val_df):,}")
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            train_df=train_df,
            val_df=val_df,
            audio_processor=self.audio_processor,
            tokenizer=self.tokenizer,
            batch_size=self.config.get('batch_size', 64),
            val_batch_size=self.config.get('val_batch_size', 128),
            num_workers=self.config.get('num_workers', 12),
            augmenter=self.augmenter,
            persistent_workers=self.config.get('persistent_workers', True),
            prefetch_factor=self.config.get('prefetch_factor', 4),
            sort_by_length=self.config.get('sort_by_length', True),
            use_bucketing=self.config.get('use_bucketing', False),
            cache_in_ram=self.config.get('cache_in_ram', False)
        )
        
        # Create model
        # Tạm thời tắt gradient checkpointing vì có conflict với CTC output
        # Có thể bật lại sau khi fix checkpointing
        use_checkpointing = self.config.get('use_gradient_checkpointing', False)
        self.model = ASRModel(
            input_dim=self.config.get('n_mels', 80),
            vocab_size=vocab_size,
            d_model=self.config.get('d_model', 256),
            num_encoder_layers=self.config.get('num_encoder_layers', 14),
            num_decoder_layers=self.config.get('num_decoder_layers', 6),
            num_heads=self.config.get('num_heads', 8),
            d_ff=self.config.get('d_ff', 2048),
            dropout=self.config.get('dropout', 0.2),
            num_languages=2,  # Vietnamese + English
            use_gradient_checkpointing=use_checkpointing  # Tắt tạm thời để tránh lỗi với CTC
        )
        
        self.model.to(self.device)
        
        # Log model info
        num_params = self.model.get_num_params()
        num_trainable = self.model.get_num_trainable_params()
        self.logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
        
        # Optimizer
        learning_rate = self.config.get('learning_rate', 0.0003)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        num_epochs = self.config.get('num_epochs', 50)
        warmup_pct = self.config.get('warmup_pct', 0.03)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        
        # Account for gradient accumulation: global_step increments once per gradient_accumulation_steps batches
        batches_per_step = gradient_accumulation_steps
        total_batches = len(self.train_loader) * num_epochs
        total_steps = total_batches // batches_per_step  # Actual optimizer steps
        
        self.warmup_steps = int(total_steps * warmup_pct)
        
        # Use faster cosine annealing - reduce T_max to make LR decrease quicker
        # Instead of full remaining steps, use half to make decay 2x faster
        cosine_period = self.config.get('lr_decay_period', None)
        if cosine_period is None:
            # Default: use 50% of remaining steps for faster decay
            cosine_period = int((total_steps - self.warmup_steps) * 0.5)
        
        # Cosine annealing with warmup - faster decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_period,
            eta_min=learning_rate * 0.01
        )
        
        # Warmup scheduler
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        
        self.logger.info(f"Learning rate: {learning_rate:.2e}")
        self.logger.info(f"Total optimizer steps: {total_steps:,} (batches: {total_batches:,}, accumulation: {batches_per_step})")
        self.logger.info(f"Warmup steps: {self.warmup_steps:,} ({warmup_pct*100:.1f}% of {total_steps:,} steps)")
        self.logger.info(f"Cosine annealing period: {cosine_period:,} steps (after warmup)")
        
        # Special token IDs (set early for CTC loss)
        self.sos_token_id = getattr(self.tokenizer, 'sos_token_id', 2)
        self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', 3)
        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        # Loss function (CrossEntropyLoss with label smoothing)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,  # Ignore padding tokens
            reduction='mean'
        )
        
        # CTC Loss for hybrid CTC/Attention training
        # Helps encoder learn better alignment
        use_ctc = self.config.get('use_ctc_loss', True)
        ctc_weight = self.config.get('ctc_weight', 0.3)  # Weight for CTC loss
        if use_ctc:
            self.ctc_criterion = CTCLoss(blank_id=self.pad_token_id)
            self.ctc_weight = ctc_weight
            self.logger.info(f"✅ CTC Loss enabled (weight: {ctc_weight})")
        else:
            self.ctc_criterion = None
            self.ctc_weight = 0.0
        
        # Scheduled Sampling to reduce teacher forcing
        # Gradually forces model to use encoder outputs instead of just language patterns
        use_scheduled_sampling = self.config.get('use_scheduled_sampling', True)
        if use_scheduled_sampling:
            initial_prob = self.config.get('teacher_forcing_initial', 1.0)
            final_prob = self.config.get('teacher_forcing_final', 0.5)
            self.scheduled_sampling = ScheduledSampling(
                initial_prob=initial_prob,
                final_prob=final_prob,
                decay_type='linear'
            )
            self.logger.info(f"✅ Scheduled Sampling enabled ({initial_prob:.2f} -> {final_prob:.2f})")
        else:
            self.scheduled_sampling = None
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callback_manager = CallbackManager()
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
            save_best=True,
            save_every_n_epochs=self.config.get('save_every', 5),
            monitor_metric='val_loss',
            mode='min'
        )
        self.callback_manager.add_callback(checkpoint_callback)
        
        # Logging callback
        logging_callback = LoggingCallback(log_every_n_batches=50)
        self.callback_manager.add_callback(logging_callback)
        
        # Metrics callback
        metrics_callback = MetricsCallback()
        self.callback_manager.add_callback(metrics_callback)
        
        # Auto-rollback callback
        auto_rollback_config = self.config.get('auto_rollback', {})
        if auto_rollback_config.get('enabled', True):
            auto_rollback = AutoRollbackCallback(
                threshold_ratio=auto_rollback_config.get('threshold_ratio', 1.3),
                patience=auto_rollback_config.get('patience', 1)
            )
            self.callback_manager.add_callback(auto_rollback)
        
        # Curriculum learning callback
        curriculum_config = self.config.get('curriculum_learning', {})
        if curriculum_config.get('enabled', True):
            curriculum = CurriculumLearningCallback(
                start_timestamp_epoch=curriculum_config.get('short_sentence_epochs', 3),
                required_wer=curriculum_config.get('required_wer', 0.70),
                initial_ts_weight=curriculum_config.get('initial_ts_weight', 0.01)
            )
            self.callback_manager.add_callback(curriculum)
    
    def _prepare_target_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Prepare target tokens for teacher forcing (shift right, add SOS).
        
        Args:
            text_tokens: Target token IDs (batch, seq_len)
            
        Returns:
            shifted_tokens: Tokens shifted right with SOS prepended (batch, seq_len+1)
        """
        batch_size = text_tokens.size(0)
        device = text_tokens.device
        
        # Create SOS tokens for each sample
        sos_tokens = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        
        # Concatenate SOS + target tokens
        # Target: [SOS, token1, token2, ..., tokenN]
        shifted_tokens = torch.cat([sos_tokens, text_tokens], dim=1)
        
        return shifted_tokens
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     target_lengths: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            target_lengths: Actual target lengths (batch,)
            
        Returns:
            loss: Scalar loss value
        """
        # Flatten for loss calculation
        logits_flat = logits.reshape(-1, logits.size(-1))  # (batch*seq_len, vocab_size)
        targets_flat = targets.reshape(-1)  # (batch*seq_len,)
        
        # Compute loss
        loss = self.criterion(logits_flat, targets_flat)
        
        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        grad_clip = self.config.get('grad_clip', 0.5)
        
        self.current_epoch_batches = len(self.train_loader)
        
        # Timing for throughput calculation
        epoch_start_time = time.time()
        batch_times = []
        
        # Print epoch header
        print("\n" + "="*100)
        print(f"🚀 TRAINING EPOCH {epoch+1}/{self.config.get('num_epochs', 50)}")
        print("="*100)
        print(f"📊 Total batches: {len(self.train_loader):,} | Effective batch size: {self.config.get('batch_size', 64) * gradient_accumulation_steps}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*100)
        
        # Enhanced progress bar with detailed info
        progress_bar = tqdm(
            self.train_loader,
            desc=f"🚀 Epoch {epoch+1}/{self.config.get('num_epochs', 50)}",
            unit="batch",
            ncols=140,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            miniters=1,
            maxinterval=1.0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            # Move to device
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            language_ids = batch['language_ids'].to(self.device)
            
            # Prepare target tokens (shift right, add SOS)
            target_tokens = self._prepare_target_tokens(text_tokens)
            
            # Apply scheduled sampling if enabled
            decoder_input = target_tokens[:, :-1]  # Remove last token (EOS) for input
            if self.scheduled_sampling is not None and self.model.training:
                # Get current teacher forcing probability
                tf_prob = self.scheduled_sampling.get_probability(epoch, self.num_epochs)
                
                # For scheduled sampling, we need to generate predictions step-by-step
                # For now, use simpler approach: randomly replace some tokens with predictions
                # This is a simplified version - full implementation would require autoregressive generation
                # For efficiency, we'll use a simpler approach: just reduce teacher forcing gradually
                # by using a mask (this is an approximation)
                if torch.rand(1).item() > tf_prob:
                    # Use previous predictions (simplified - in practice would need autoregressive)
                    # For now, we'll still use teacher forcing but log the probability
                    pass
            
            # Forward pass with mixed precision (using bf16 for better stability)
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Encoder-decoder forward with optional CTC output
                return_ctc = self.ctc_criterion is not None
                model_output = self.model(
                    x=audio_features,
                    tgt_tokens=decoder_input,
                    lengths=audio_lengths,
                    language_ids=language_ids,
                    return_ctc=return_ctc
                )
                
                if return_ctc:
                    logits, _, ctc_logits = model_output
                else:
                    logits, _ = model_output
                    ctc_logits = None
                
                # Compute attention loss (target is shifted tokens without SOS)
                attn_loss = self._compute_loss(
                    logits,
                    target_tokens[:, 1:],  # Remove SOS, keep rest
                    text_lengths
                )
                
                # Compute CTC loss if enabled
                batch_loss = attn_loss  # Use batch_loss instead of total_loss to avoid overwriting accumulator
                if ctc_logits is not None and self.ctc_criterion is not None:
                    # CTC needs encoder output lengths (after subsampling)
                    encoder_lengths = (audio_lengths / 2).long()  # Encoder subsamples by 2x
                    ctc_loss = self.ctc_criterion(
                        ctc_logits,
                        text_tokens,  # Target tokens (without SOS/EOS shift)
                        encoder_lengths,
                        text_lengths
                    )
                    # Combine losses
                    batch_loss = (1 - self.ctc_weight) * attn_loss + self.ctc_weight * ctc_loss
                
                # Scale loss for gradient accumulation
                loss = batch_loss / gradient_accumulation_steps
            
            # Extract batch loss value for accumulation (after autocast block)
            batch_loss_value = batch_loss.item()
            
            # Calculate WER/CER every 100 batches for monitoring
            if batch_idx % 100 == 0 and batch_idx > 0:
                try:
                    with torch.no_grad():
                        # Generate predictions for a small sample
                        sample_size = min(8, audio_features.size(0))
                        sample_audio = audio_features[:sample_size]
                        sample_lengths = audio_lengths[:sample_size]
                        sample_lang_ids = language_ids[:sample_size] if language_ids is not None else None
                        
                        # Generate predictions
                        generated_tokens = self.model.generate(
                            sample_audio,
                            lengths=sample_lengths,
                            language_ids=sample_lang_ids,
                            max_len=self.config.get('val_max_len', 128),
                            sos_token_id=self.sos_token_id,
                            eos_token_id=self.eos_token_id,
                            pad_token_id=self.pad_token_id,
                            temperature=1.0
                        )
                        
                        # Decode predictions and references
                        sample_refs = []
                        sample_preds = []
                        for i in range(sample_size):
                            # Decode prediction
                            gen_seq = generated_tokens[i].cpu().tolist()
                            decoded_tokens = []
                            for token in gen_seq:
                                if token == self.eos_token_id:
                                    break
                                if token != self.sos_token_id and token != self.pad_token_id:
                                    decoded_tokens.append(token)
                            pred_text = self.tokenizer.decode(decoded_tokens)
                            sample_preds.append(pred_text)
                            
                            # Get reference
                            ref_tokens = text_tokens[i].cpu().tolist()
                            ref_decoded = []
                            for token in ref_tokens:
                                if token == self.eos_token_id or token == self.pad_token_id:
                                    break
                                if token != self.sos_token_id:
                                    ref_decoded.append(token)
                            ref_text = self.tokenizer.decode(ref_decoded)
                            sample_refs.append(ref_text)
                        
                        # Calculate WER/CER
                        batch_wer = calculate_wer(sample_refs, sample_preds)
                        batch_cer = calculate_cer(sample_refs, sample_preds)
                        
                        # Log to progress bar
                        progress_bar.set_postfix({
                            'loss': f'{loss.item()*gradient_accumulation_steps:.3f}',
                            'WER': f'{batch_wer*100:.1f}%',
                            'CER': f'{batch_cer*100:.1f}%'
                        })
                        
                        # Log to file
                        self.logger.info(
                            f"Batch {batch_idx}/{len(self.train_loader)} | "
                            f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | "
                            f"WER: {batch_wer:.4f} ({batch_wer*100:.2f}%) | "
                            f"CER: {batch_cer:.4f} ({batch_cer*100:.2f}%)"
                        )
                except Exception as e:
                    # Don't crash training if WER/CER calculation fails
                    self.logger.warning(f"Failed to calculate WER/CER at batch {batch_idx}: {e}")
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                
                # Update learning rate (warmup + cosine annealing)
                self.global_step += 1
                if self.global_step <= self.warmup_steps:
                    # During warmup: use warmup scheduler
                    self.warmup_scheduler.step()
                else:
                    # After warmup: use cosine annealing scheduler
                    # Ensure LR is at base_lr when transitioning from warmup to cosine
                    if self.global_step == self.warmup_steps + 1:
                        # First step after warmup: reset optimizer LR to base_lr
                        base_lr = self.config.get('learning_rate', 0.0003)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = base_lr
                        # Reset cosine scheduler to start from beginning
                        self.scheduler.last_epoch = -1
                        self.logger.info(f"Warmup complete at step {self.warmup_steps}, starting cosine annealing from LR={base_lr:.2e}")
                    
                    # Step cosine annealing scheduler
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate loss (use the actual batch loss value, not the scaled loss)
            total_loss += batch_loss_value
            num_batches += 1
            
            # Calculate batch time for throughput
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            if len(batch_times) > 100:
                batch_times.pop(0)
            avg_batch_time = sum(batch_times) / len(batch_times)
            throughput = self.config.get('batch_size', 64) / avg_batch_time if avg_batch_time > 0 else 0
            
            # Update progress bar with detailed metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss_so_far = total_loss / num_batches if num_batches > 0 else 0.0
            current_loss = loss.item() * gradient_accumulation_steps
            
            # Get GPU memory if available
            gpu_mem_used = 0
            gpu_mem_total = 0
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Enhanced progress bar display - chỉ loss và LR (tất cả trên một dòng)
            progress_bar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'Avg': f"{avg_loss_so_far:.4f}",
                'LR': f"{current_lr:.2e}",
                'BestVal': f"{self.best_val_loss:.4f}" if self.best_val_loss < float('inf') else "N/A"
            }, refresh=True)
            
            # Log metrics every 50 batches for monitoring - chỉ loss và LR
            if batch_idx % 50 == 0:
                best_val_str = f"{self.best_val_loss:.6f}" if self.best_val_loss < float('inf') else "N/A"
                self.logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item() * gradient_accumulation_steps:.6f} | "
                    f"Avg Loss: {avg_loss_so_far:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Best Val Loss: {best_val_str}"
                )
                # Force flush to ensure log is written immediately
                for handler in self.logger.handlers:
                    handler.flush()
            
            # Callback: batch end
            self.callback_manager.on_batch_end(self, batch_idx, loss.item() * gradient_accumulation_steps)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(timedelta(seconds=int(epoch_time)))
        
        # Print epoch completion summary
        print("\n" + "="*100)
        print(f"✅ TRAINING EPOCH {epoch+1} COMPLETE")
        print("="*100)
        print(f"📊 Train Loss: {avg_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"⏱️  Epoch Time: {epoch_time_str}")
        print("="*100)
        
        metrics = {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'best_val_loss': self.best_val_loss,
            'best_wer': self.best_wer if self.best_wer < float('inf') else None,
            'best_cer': self.best_cer if self.best_cer < float('inf') else None
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        val_max_batches = self.config.get('val_max_batches', None)
        val_subset_pct = self.config.get('val_subset_pct', None)
        use_autoregressive_validation = self.config.get('use_autoregressive_validation', False)
        calculate_val_wer = self.config.get('calculate_val_wer', False)
        val_max_len = self.config.get('val_max_len', 128)
        
        # Determine number of batches to validate
        num_val_batches = len(self.val_loader)
        if val_max_batches:
            num_val_batches = min(num_val_batches, val_max_batches)
        elif val_subset_pct:
            num_val_batches = int(num_val_batches * val_subset_pct)
        
        # Print validation header
        print("\n" + "="*100)
        print(f"✅ VALIDATION EPOCH {epoch+1}")
        print("="*100)
        print(f"📊 Validation batches: {num_val_batches:,}")
        print("-"*100)
        
        progress_bar = tqdm(
            list(self.val_loader)[:num_val_batches],
            desc=f"✅ Validation {epoch+1}",
            unit="batch",
            ncols=140,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            miniters=1,
            maxinterval=1.0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            language_ids = batch['language_ids'].to(self.device)
            transcripts = batch['transcripts']
            
            # Prepare target tokens
            target_tokens = self._prepare_target_tokens(text_tokens)
            
            # Forward pass (using bf16 for better stability)
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                logits, _ = self.model(
                    x=audio_features,
                    tgt_tokens=target_tokens[:, :-1],
                    lengths=audio_lengths,
                    language_ids=language_ids
                )
                
                loss = self._compute_loss(
                    logits,
                    target_tokens[:, 1:],
                    text_lengths
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Generate predictions for WER/CER calculation
            if calculate_val_wer:
                if use_autoregressive_validation:
                    # Use autoregressive generation (slower but more accurate)
                    generated_tokens = self.model.generate(
                        audio_features,
                        lengths=audio_lengths,
                        language_ids=language_ids,
                        max_len=val_max_len,
                        sos_token_id=self.sos_token_id,
                        eos_token_id=self.eos_token_id,
                        pad_token_id=self.pad_token_id,
                        temperature=1.0
                    )
                    
                    # Decode generated tokens
                    for i in range(generated_tokens.size(0)):
                        gen_seq = generated_tokens[i].cpu().tolist()
                        decoded_tokens = []
                        for token in gen_seq:
                            if token == self.eos_token_id:
                                break
                            if token != self.sos_token_id and token != self.pad_token_id:
                                decoded_tokens.append(token)
                        
                        pred_text = self.tokenizer.decode(decoded_tokens)
                        all_predictions.append(pred_text)
                        all_references.append(transcripts[i])
                else:
                    # Use greedy decoding from logits (faster)
                    pred_tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                    
                    # Decode predictions
                    for i in range(pred_tokens.size(0)):
                        pred_seq = pred_tokens[i].cpu().tolist()
                        decoded_tokens = []
                        for token in pred_seq:
                            if token == self.eos_token_id:
                                break
                            if token != self.pad_token_id:
                                decoded_tokens.append(token)
                        
                        pred_text = self.tokenizer.decode(decoded_tokens)
                        all_predictions.append(pred_text)
                        all_references.append(transcripts[i])
            
            # Calculate running average loss
            running_avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Prepare metrics for progress bar - chỉ loss và LR
            val_metrics_dict = {
                'Loss': f"{loss.item():.4f}",
                'Avg': f"{running_avg_loss:.4f}",
                'LR': f"{current_lr:.2e}",
                'BestVal': f"{self.best_val_loss:.4f}" if self.best_val_loss < float('inf') else "N/A"
            }
            
            progress_bar.set_postfix(val_metrics_dict, refresh=True)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'best_val_loss': self.best_val_loss,
            'best_wer': self.best_wer if self.best_wer < float('inf') else None,
            'best_cer': self.best_cer if self.best_cer < float('inf') else None
        }
        
        # Calculate WER/CER if requested
        if calculate_val_wer and len(all_predictions) > 0:
            wer = calculate_wer(all_references, all_predictions)
            cer = calculate_cer(all_references, all_predictions)
            metrics['wer'] = wer
            metrics['cer'] = cer
        
        # Print final validation summary - chỉ loss và LR
        print("\n" + "="*100)
        print(f"✅ VALIDATION COMPLETE - EPOCH {epoch+1}")
        print("="*100)
        print(f"📊 Val Loss: {avg_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"🏆 Best Val Loss: {self.best_val_loss:.6f}")
        print("="*100 + "\n")
        
        return metrics
    
    def train(self, resume_from: Optional[str] = None, start_epoch: int = 0):
        """Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
            start_epoch: Starting epoch (for resume)
        """
        num_epochs = self.config.get('num_epochs', 50)
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            # Continue from next epoch after checkpoint
            start_epoch = self.current_epoch + 1
            self.logger.info(f"Resumed training from epoch {self.current_epoch}, continuing from epoch {start_epoch}")
        
        # Callback: training begin
        self.callback_manager.on_train_begin(self)
        
        try:
            for epoch in range(start_epoch, num_epochs):
                if self.should_stop:
                    self.logger.info("Training stopped early")
                    break
                
                self.current_epoch = epoch
                
                # Callback: epoch begin
                self.callback_manager.on_epoch_begin(self, epoch)
                
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Update best metrics
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                
                if 'wer' in val_metrics and val_metrics['wer'] is not None and val_metrics['wer'] < self.best_wer:
                    self.best_wer = val_metrics['wer']
                
                if 'cer' in val_metrics and val_metrics['cer'] is not None and val_metrics['cer'] < self.best_cer:
                    self.best_cer = val_metrics['cer']
                
                # Callback: epoch end
                self.callback_manager.on_epoch_end(self, epoch, metrics)
                
                # Log epoch summary - chỉ loss và LR
                print("\n" + "="*100)
                print(f"📊 EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
                print("="*100)
                print(f"  🎯 Train Loss:    {metrics.get('train_loss', 0):.6f}")
                print(f"  ✅ Val Loss:      {metrics.get('val_loss', 0):.6f}")
                print(f"  📈 Learning Rate: {metrics.get('learning_rate', 0):.6e}")
                print("-" * 100)
                print(f"  🏆 Best Val Loss:  {self.best_val_loss:.6f}")
                print("="*100 + "\n")
                
                # Also log to file
                self.logger.info("\n" + "="*80)
                self.logger.info(f"📊 EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
                self.logger.info("="*80)
                self.logger.info(f"  🎯 Train Loss:    {metrics.get('train_loss', 0):.6f}")
                self.logger.info(f"  ✅ Val Loss:      {metrics.get('val_loss', 0):.6f}")
                self.logger.info(f"  📈 Learning Rate: {metrics.get('learning_rate', 0):.6e}")
                self.logger.info("-" * 80)
                self.logger.info(f"  🏆 Best Val Loss:  {self.best_val_loss:.6f}")
                self.logger.info("="*80 + "\n")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            raise
        finally:
            # Callback: training end
            self.callback_manager.on_train_end(self)
            self.logger.info("Training completed")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state FIRST (needed for scheduler recalculation)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', self.current_epoch * len(self.train_loader))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_wer = checkpoint.get('best_wer', float('inf'))
        self.best_cer = checkpoint.get('best_cer', float('inf'))
        
        # Calculate target LR BEFORE loading optimizer state
        target_lr = None
        if hasattr(self, 'scheduler'):
            # Calculate how many steps into cosine annealing we are
            steps_into_cosine = max(0, self.global_step - self.warmup_steps)
            cosine_period = self.scheduler.T_max
            
            # If we're past the cosine period, keep LR at minimum
            if steps_into_cosine >= cosine_period:
                target_lr = self.scheduler.eta_min
                self.logger.info(f"Resuming: past cosine period, LR set to minimum: {self.scheduler.eta_min:.2e}")
            elif steps_into_cosine > 0:
                # Calculate current LR based on cosine schedule
                import math
                base_lr = self.config.get('learning_rate', 0.0003)
                current_lr_ratio = 0.5 * (1 + math.cos(math.pi * steps_into_cosine / cosine_period))
                target_lr = self.scheduler.eta_min + (base_lr - self.scheduler.eta_min) * current_lr_ratio
                self.logger.info(f"Resuming: target LR calculated: {target_lr:.6e} (step {steps_into_cosine}/{cosine_period} in cosine schedule)")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # NOW set the correct LR AFTER loading optimizer (to override old LR from checkpoint)
        if target_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            self.logger.info(f"Resuming: LR set to {target_lr:.6e} (overriding checkpoint LR)")
            
            # Reset scheduler to continue from here
            if hasattr(self, 'scheduler'):
                steps_into_cosine = max(0, self.global_step - self.warmup_steps)
                self.scheduler.last_epoch = steps_into_cosine - 1
        
        # Handle learning rate on resume
        use_constant_lr = self.config.get('use_constant_lr_on_resume', False)
        if not use_constant_lr and 'learning_rate' in checkpoint:
            # Restore learning rate from checkpoint
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = checkpoint['learning_rate']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
        self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best WER: {self.best_wer:.4f}")
        self.logger.info(f"Best CER: {self.best_cer:.4f}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch (for resume)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = ASRTrainer(config)
    
    # Train
    trainer.train(resume_from=args.resume, start_epoch=args.start_epoch)


if __name__ == '__main__':
    main()

