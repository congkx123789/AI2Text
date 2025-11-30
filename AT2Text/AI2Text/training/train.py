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
import logging
import os
import multiprocessing

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer, BilingualTextNormalizer
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
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
            self.logger.info(f"Effective batch size: {self.config.get('batch_size', 16) * self.gradient_accumulation_steps}")
        
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
            bpe_path = self.config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(bpe_path)
        else:
            self.tokenizer = Tokenizer()

        # Use bilingual normalizer to support Vietnamese + English
        self.normalizer = BilingualTextNormalizer()
    
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
        
        # PyTorch 2.0+ torch.compile() for faster training
        # Compiles model graph for optimal GPU performance
        self.use_compile = self.config.get('use_compile', False) and hasattr(torch, 'compile')
        if self.use_compile:
            compile_mode = self.config.get('compile_mode', 'reduce-overhead')
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                self.logger.info(f"🚀 torch.compile() enabled with mode: {compile_mode}")
                self.logger.info("   Note: First epoch will be slower due to compilation, subsequent epochs will be faster")
            except Exception as e:
                self.logger.warning(f"torch.compile() failed: {e}, continuing without compilation")
                self.use_compile = False
        elif self.config.get('use_compile', False):
            self.logger.warning("torch.compile() requested but not available (PyTorch < 2.0.0 or no CUDA)")
        
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
        # Explicitly set reduction='mean' to ensure consistent loss calculation
        # zero_infinity=True replaces inf loss with 0 when input_length < target_length
        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.blank_token_id,
            zero_infinity=True,
            reduction='mean'  # Explicit: mean over batch (default, but make it clear)
        )
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler with proper warmup to prevent gradient explosion."""
        max_lr = self.config.get('learning_rate', 1e-4)
        warmup_pct = self.config.get('warmup_pct', 0.2)  # 20% warmup by default
        
        # CRITICAL FIX: Prevent blank token collapse
        # Model đã bị collapse vào blank token trap (output toàn blank)
        # Nguyên nhân: Initial LR quá cao → model nhảy vọt vào local minima
        # Giải pháp: Start với LR rất nhỏ, warmup từ từ
        div_factor = 100.0  # Start at max_lr/100 (rất nhỏ để tránh blank trap)
        final_div_factor = 10000.0  # End at max_lr/10000 (rất nhỏ ở cuối)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_pct,  # 20% warmup to prevent gradient explosion
            anneal_strategy='cos',
            div_factor=div_factor,  # Start at max_lr/10 (higher initial LR)
            final_div_factor=final_div_factor  # End at max_lr/1000
        )
        
        initial_lr = max_lr / div_factor
        self.logger.info(f"📈 Learning rate scheduler (FIXED for blank collapse):")
        self.logger.info(f"   Max LR: {max_lr}")
        self.logger.info(f"   Initial LR: {initial_lr:.2e} (warmup {warmup_pct*100:.0f}%)")
        self.logger.info(f"   ⚠️  Initial LR rất nhỏ để tránh blank token collapse")
        self.logger.info(f"   ⚠️  Model sẽ học từ từ, không nhảy vọt vào blank trap")
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Get batch size info for progress bar
        config_batch_size = self.config.get('batch_size', 16)
        effective_batch_size = config_batch_size * self.gradient_accumulation_steps
        
        # Configure tqdm to write to stderr to avoid conflict with logger
        # Show batch size info in progress bar to clarify confusion
        # Pipelining: CPU processes data ahead while GPU trains (simultaneous)
        pbar = tqdm(train_loader, 
                   desc=f'Epoch {self.current_epoch} (batch={config_batch_size}, effective={effective_batch_size})', 
                   file=sys.stderr, dynamic_ncols=True)
        
        # Pipeline optimization: Use non_blocking transfer to overlap CPU/GPU work
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio_features = batch['audio_features'].to(self.device, non_blocking=True)
            audio_lengths = batch['audio_lengths'].to(self.device, non_blocking=True)
            text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
            text_lengths = batch['text_lengths'].to(self.device, non_blocking=True)
            
            # Zero gradients only at the start of accumulation cycle
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, output_lengths = self.model(audio_features, audio_lengths)
                    
                    # Validate output_lengths >= text_lengths (required for CTC)
                    # This check helps catch potential issues early
                    invalid_mask = output_lengths < text_lengths
                    if invalid_mask.any():
                        invalid_count = invalid_mask.sum().item()
                        self.logger.warning(
                            f"⚠️  Batch {batch_idx}: {invalid_count} samples with "
                            f"output_length < text_length. CTC loss may be 0 (zero_infinity=True)."
                        )
                        # Log details for first invalid sample
                        if invalid_count > 0:
                            first_invalid = invalid_mask.nonzero()[0].item()
                            self.logger.warning(
                                f"   Sample {first_invalid}: output_len={output_lengths[first_invalid].item()}, "
                                f"text_len={text_lengths[first_invalid].item()}"
                            )
                    
                    # CTC loss expects (T, N, C) format
                    logits = logits.transpose(0, 1)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                    
                    # Check for NaN/Inf loss (shouldn't happen with zero_infinity=True)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(
                            f"❌ Batch {batch_idx}: Loss is NaN/Inf! "
                            f"This should not happen with zero_infinity=True."
                        )
                        # Skip this batch
                        continue
                    
                    # Scale loss by accumulation steps for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Update weights only after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config.get('grad_clip', 1.0))
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                logits, output_lengths = self.model(audio_features, audio_lengths)
                
                # Validate output_lengths >= text_lengths (required for CTC)
                invalid_mask = output_lengths < text_lengths
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum().item()
                    self.logger.warning(
                        f"⚠️  Batch {batch_idx}: {invalid_count} samples with "
                        f"output_length < text_length. CTC loss may be 0 (zero_infinity=True)."
                    )
                
                # CTC loss
                logits = logits.transpose(0, 1)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(
                        f"❌ Batch {batch_idx}: Loss is NaN/Inf! "
                        f"This should not happen with zero_infinity=True."
                    )
                    continue
                
                # Scale loss by accumulation steps for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                # Update weights only after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.config.get('grad_clip', 1.0))
                    self.optimizer.step()
            
            # Update scheduler (every step, not every accumulation step)
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Accumulate loss (multiply back to get true loss)
            true_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += true_loss
            num_batches += 1
            
            # Training Layer: Callback on_batch_end (Logging)
            self.callback_manager.on_batch_end(self, num_batches - 1, loss.item())
            
            # Update progress bar with TRUE loss (not scaled loss)
            # This matches the final average loss calculation
            pbar.set_postfix({'loss': true_loss})
        
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
            
            # Validate output_lengths >= text_lengths (required for CTC)
            invalid_mask = output_lengths < text_lengths
            if invalid_mask.any():
                invalid_count = invalid_mask.sum().item()
                self.logger.warning(
                    f"⚠️  Val batch {num_batches}: {invalid_count} samples with "
                    f"output_length < text_length. CTC loss may be 0 (zero_infinity=True)."
                )
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"❌ Val batch {num_batches}: Loss is NaN/Inf! "
                    f"This should not happen with zero_infinity=True."
                )
                continue
            
            total_loss += loss.item()
            
            # Decode predictions for WER/CER calculation
            predictions = torch.argmax(logits, dim=-1)
            
            # ============================================================
            # DEBUG CHECKLIST #1: In thử Output ngay lập tức
            # ============================================================
            is_first_batch = (num_batches == 0)
            if is_first_batch:
                self.logger.info("=" * 80)
                self.logger.info("🔍 DEBUG CHECKLIST #1: In thử Output ngay lập tức")
                self.logger.info("=" * 80)
                
                # Decode using tokenizer.batch_decode (as requested in checklist)
                # Get first sample for immediate inspection
                first_pred_tokens = predictions[0, :output_lengths[0]].cpu().unsqueeze(0)
                first_label_tokens = text_tokens[0, :text_lengths[0]].cpu().unsqueeze(0)
                
                # Use tokenizer decode (will handle CTC collapse internally)
                try:
                    decoded_preds = [self._ctc_decode(predictions[0, :output_lengths[0]].cpu().tolist())]
                    decoded_labels = [self.tokenizer.decode(text_tokens[0, :text_lengths[0]].cpu().tolist())]
                    
                    self.logger.info("")
                    self.logger.info("🚨 IMMEDIATE OUTPUT CHECK (First Sample):")
                    self.logger.info(f"  Pred: '{decoded_preds[0]}'")
                    self.logger.info(f"  True: '{decoded_labels[0]}'")
                    self.logger.info("")
                    
                    # Check for empty prediction (Trường hợp 1)
                    if len(decoded_preds[0].strip()) == 0:
                        self.logger.error("  ❌ TRƯỜNG HỢP 1: Pred rỗng tuếch!")
                        self.logger.error("     → Lỗi do CTC Loss (input quá ngắn) hoặc Learning Rate sai")
                        self.logger.error("     → Hoặc Tokenizer bị sai index")
                    
                    # Check for <unk> tokens (Trường hợp 2)
                    pred_tokens_list = predictions[0, :output_lengths[0]].cpu().tolist()
                    if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                        unk_count = sum(1 for t in pred_tokens_list if t == self.tokenizer.unk_token_id)
                        if unk_count > len(pred_tokens_list) * 0.5:  # >50% are <unk>
                            self.logger.error("  ❌ TRƯỜNG HỢP 2: Pred chứa nhiều <unk> tokens!")
                            self.logger.error("     → Tokenizer chưa cover được bộ từ vựng (Vocab)")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Error decoding: {e}")
            
            # ============================================================
            # DEBUG CHECKLIST #2: Kiểm tra "Độ dài Audio sau khi nén"
            # ============================================================
            if is_first_batch:
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info("🔍 DEBUG CHECKLIST #2: Kiểm tra Độ dài Audio sau khi nén")
                self.logger.info("=" * 80)
                self.logger.info("Model ASR dùng CNN subsampling 4x (2 conv stride 2)")
                self.logger.info("")
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                # Decode using CTC collapse (remove blanks and duplicates)
                pred_text = self._ctc_decode(pred_tokens)
                ref_text = self.tokenizer.decode(ref_tokens)
                
                # CRITICAL: Check subsampling ratio
                audio_len_original = audio_lengths[i].item()  # Original audio length (spectrogram frames)
                audio_len_after_subsample = output_lengths[i].item()  # After 4x subsampling
                text_len = text_lengths[i].item()
                
                # Calculate expected subsampling
                expected_subsample = audio_len_original // 4  # Model subsamples by 4x
                actual_subsample = audio_len_after_subsample
                
                if is_first_batch:
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"Sample {i+1}/{predictions.size(0)}:")
                    self.logger.info(f"{'='*80}")
                    self.logger.info(f"  📝 Reference:")
                    self.logger.info(f"     Text: '{ref_text}'")
                    self.logger.info(f"     Tokens: {ref_tokens[:30]}..." if len(ref_tokens) > 30 else f"     Tokens: {ref_tokens}")
                    self.logger.info(f"     Length: {len(ref_tokens)} tokens")
                    self.logger.info(f"  🤖 Prediction:")
                    self.logger.info(f"     Text: '{pred_text}'")
                    self.logger.info(f"     Tokens: {pred_tokens[:30]}..." if len(pred_tokens) > 30 else f"     Tokens: {pred_tokens}")
                    self.logger.info(f"     Length: {len(pred_tokens)} tokens")
                    self.logger.info(f"  📊 Audio Length Analysis (CRITICAL):")
                    self.logger.info(f"     Original audio length (spectrogram): {audio_len_original} frames")
                    self.logger.info(f"     Expected after 4x subsampling: ~{expected_subsample} frames")
                    self.logger.info(f"     Actual output length: {actual_subsample} frames")
                    self.logger.info(f"     Text length: {text_len} tokens")
                    self.logger.info(f"     Output/Text ratio: {actual_subsample/text_len:.2f}x" if text_len > 0 else "     Output/Text ratio: N/A")
                    
                    # CRITICAL CHECK: Input (output_length) > Output (text_length)
                    if actual_subsample < text_len:
                        self.logger.error(f"     🚨 LỖI CTC: Input ({actual_subsample}) < Output ({text_len})!")
                        self.logger.error(f"        → CTC Loss sẽ ra vô cực (Infinity) hoặc model output rỗng")
                        self.logger.error(f"        → Giải pháp: Lọc bỏ audio quá ngắn (< 1s) hoặc text quá dài")
                    elif actual_subsample < text_len * 1.2:  # Warning if ratio is too close
                        self.logger.warning(f"     ⚠️  CẢNH BÁO: Ratio quá gần ({actual_subsample/text_len:.2f}x)")
                        self.logger.warning(f"        → Nên có ít nhất 1.5x để CTC hoạt động tốt")
                    else:
                        self.logger.info(f"     ✅ OK: Input ({actual_subsample}) > Output ({text_len})")
                    
                    self.logger.info(f"     Blank token ID: {self.tokenizer.blank_token_id}")
                    
                    # Check if all predictions are blank
                    unique_preds = set(pred_tokens)
                    self.logger.info(f"     Unique pred tokens: {unique_preds}")
                    self.logger.info(f"     Number of unique tokens: {len(unique_preds)}")
                    
                    # Critical checks
                    if len(pred_text.strip()) == 0:
                        self.logger.error(f"     🚨 PREDICTION IS EMPTY STRING!")
                    elif len(unique_preds) == 1 and list(unique_preds)[0] == self.tokenizer.blank_token_id:
                        self.logger.error(f"     🚨 ALL PREDICTIONS ARE BLANK TOKEN!")
                    elif len(unique_preds) <= 3:
                        self.logger.warning(f"     ⚠️  VERY FEW UNIQUE TOKENS ({len(unique_preds)}) - Model may be collapsing")
                    else:
                        self.logger.info(f"     ✅ Prediction has {len(unique_preds)} unique tokens")
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
            
            # Increment AFTER processing the batch
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        wer = calculate_wer(all_references, all_predictions)
        cer = calculate_cer(all_references, all_predictions)
        
        # CRITICAL: Log summary of predictions for debugging WER=1.0
        self.logger.info("=" * 80)
        self.logger.info("📊 VALIDATION SUMMARY (Debug WER/CER)")
        self.logger.info("=" * 80)
        
        # Count empty predictions
        empty_preds = sum(1 for p in all_predictions if len(p.strip()) == 0)
        blank_only = sum(1 for p in all_predictions if len(set(self.tokenizer.encode(p))) <= 1)
        
        self.logger.info(f"Total samples: {len(all_predictions)}")
        self.logger.info(f"Empty predictions: {empty_preds} ({empty_preds/len(all_predictions)*100:.1f}%)")
        self.logger.info(f"Blank-only predictions: {blank_only} ({blank_only/len(all_predictions)*100:.1f}%)")
        
        # Show first few predictions vs references
        self.logger.info(f"\nFirst 5 predictions vs references:")
        for i in range(min(5, len(all_predictions))):
            self.logger.info(f"  [{i+1}] Ref: '{all_references[i][:50]}...' | Pred: '{all_predictions[i][:50]}...'")
        
        if wer >= 0.95:
            self.logger.error("=" * 80)
            self.logger.error("🚨 CRITICAL: WER >= 0.95 (Model not learning!)")
            self.logger.error("=" * 80)
            self.logger.error("Possible causes:")
            self.logger.error("  1. Model output is all blank/empty")
            self.logger.error("  2. Learning rate too low (stuck in local minima)")
            self.logger.error("  3. output_lengths < text_lengths (CTC alignment issue)")
            self.logger.error("  4. Tokenizer mismatch or encoding issue")
            self.logger.error("")
            self.logger.error("Action required:")
            self.logger.error("  - Check prediction samples above")
            self.logger.error("  - Verify output_lengths >= text_lengths")
            self.logger.error("  - Consider increasing learning rate")
            self.logger.error("  - Check tokenizer encoding/decoding")
        
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
        
        When resuming for fine-tuning, the scheduler will be recreated with new
        total_steps based on the new dataset size. This is intentional for
        curriculum learning scenarios.
        """
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # CRITICAL FIX: Resume from the next epoch, not epoch 0
        checkpoint_epoch = checkpoint.get('epoch', 0)
        self.current_epoch = checkpoint_epoch + 1  # Continue from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_wer = checkpoint.get('best_wer', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint_epoch}")
        self.logger.info(f"Resuming training from epoch {self.current_epoch}")
        self.logger.info(f"Best validation loss from checkpoint: {self.best_val_loss:.4f}")
        
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
    
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--language', type=str, default=None,
                       help='Filter training data by language (e.g., "en" or "vi"). '
                            'Useful for sequential training: train English first, then Vietnamese.')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
        print(f"Training samples (all languages): {len(train_df)}")
        print(f"Validation samples (all languages): {len(val_df)}")
    
    # Create data loaders
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()

    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
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

