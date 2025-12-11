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
import json
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_with_timestamps import ASRModelWithTimestamps, create_timestamp_targets
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
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
from utils.manifest_loader import load_merged_dataset
# Beam search removed - using greedy decode only


class ASRTrainer:
    """Trainer class for ASR model with timestamp support and bilingual (English + Vietnamese) training.
    
    SIMPLIFIED: No database dependency - uses file-based logging instead.
    """
    
    def __init__(self, config: dict):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('ASRTrainer', config.get('log_file', 'training.log'))
        
        # Timestamp training config
        self.use_timestamps = self.config.get('use_timestamps', True)
        self.timestamp_loss_weight = self.config.get('timestamp_loss_weight', 0.1)
        self.subsampling_factor = 2  # FIX: Changed from 4 to 2 (reduced stride)
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
        
        # File-based logging (replaces database)
        self.run_name = config.get('run_name', f'run_{int(time.time())}')
        self.metrics_file = Path(config.get('checkpoint_dir', 'checkpoints')) / f'{self.run_name}_metrics.json'
        self.metrics_history = []
        
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
        """Thiết lập callbacks với cấu hình từ file yaml."""
        from training.callbacks import (
            CallbackManager, CheckpointCallback, EarlyStoppingCallback,
            LoggingCallback, MetricsCallback
        )
        from training.smart_callbacks import AutoRollbackCallback, CurriculumLearningCallback
        
        self.callback_manager = CallbackManager()
        
        # 1. Auto-Rollback (Đọc từ config)
        rollback_conf = self.config.get('auto_rollback', {})
        if rollback_conf.get('enabled', True):
            self.callback_manager.add_callback(
                AutoRollbackCallback(
                    threshold_ratio=rollback_conf.get('threshold_ratio', 1.3),
                    patience=rollback_conf.get('patience', 1)
                )
            )
            self.logger.info(
                f"✅ Auto-Rollback enabled: threshold={rollback_conf.get('threshold_ratio', 1.3)}, "
                f"patience={rollback_conf.get('patience', 1)}"
            )
        
        # 2. Curriculum Learning (Đọc từ config)
        curr_conf = self.config.get('curriculum_learning', {})
        if curr_conf.get('enabled', True):
            self.callback_manager.add_callback(
                CurriculumLearningCallback(
                    start_timestamp_epoch=curr_conf.get('start_timestamp_epoch', 3),
                    required_wer=curr_conf.get('required_wer', 0.70),
                    initial_ts_weight=curr_conf.get('initial_ts_weight', 0.01)
                )
            )
            self.logger.info(
                f"✅ Curriculum Learning enabled: start_epoch={curr_conf.get('start_timestamp_epoch', 3)}, "
                f"required_wer={curr_conf.get('required_wer', 0.70)}"
            )
        
        # --------------------------------
        
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

        # Select tokenizer type (character-level, BPE, or SentencePiece)
        tokenizer_type = self.config.get('tokenizer_type', 'char')
        if tokenizer_type == 'sentencepiece':
            from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
            model_path = self.config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
            self.tokenizer = SentencePieceTokenizer(model_path)
            self.logger.info(f"✅ Using SentencePiece BPE tokenizer: {model_path} ({len(self.tokenizer)} tokens)")
        elif tokenizer_type == 'bpe':
            from preprocessing.bpe_tokenizer import BPETokenizer
            bpe_path = self.config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json')
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(bpe_path)
            self.logger.info(f"✅ Using BPE tokenizer: {bpe_path} ({len(self.tokenizer)} tokens)")
        else:
            # Optional: load character vocab from JSON built by build_vocab.py
            char_vocab_path = self.config.get('char_vocab_path')
            if char_vocab_path:
                vocab_file = Path(char_vocab_path)
                if not vocab_file.exists():
                    self.logger.warning(
                        f"Char vocab file not found: {vocab_file}, falling back to default tokenizer vocab"
                    )
                    self.tokenizer = Tokenizer()
                else:
                    with vocab_file.open("r", encoding="utf-8") as f:
                        vocab_dict = json.load(f)
                    # Convert {char: id} → ordered vocab list by id
                    max_id = max(vocab_dict.values())
                    vocab_list = [None] * (max_id + 1)
                    for ch, idx in vocab_dict.items():
                        if 0 <= idx <= max_id:
                            vocab_list[idx] = ch
                    # Fill any holes with <unk>
                    for i in range(len(vocab_list)):
                        if vocab_list[i] is None:
                            vocab_list[i] = "<unk>"
                    self.tokenizer = Tokenizer(vocab=vocab_list)
                    self.logger.info(
                        f"✅ Using character-level tokenizer from {vocab_file} ({len(self.tokenizer)} tokens)"
                    )
            else:
                self.tokenizer = Tokenizer()
                self.logger.info(f"✅ Using default character-level tokenizer ({len(self.tokenizer)} tokens)")

        # Use bilingual normalizer to support Vietnamese + English
        self.normalizer = BilingualTextNormalizer()
        
        # Using greedy decode only (beam search removed for speed)
        self.logger.info(f"✅ Using Greedy Decoding (fast and efficient)")
    
    def _greedy_ctc_decode(self, logits: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        """
        Fast greedy CTC decoding for training loop.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            lengths: Sequence lengths (batch,)
            
        Returns:
            List of decoded token sequences (one per batch item)
        """
        batch_size = logits.size(0)
        blank_id = self.tokenizer.blank_token_id
        
        # Get predictions (argmax)
        predictions = torch.argmax(logits, dim=-1)  # (batch, time)
        
        results = []
        for b in range(batch_size):
            seq_len = int(lengths[b].item())
            pred_seq = predictions[b, :seq_len].cpu().tolist()
            
            # CTC collapse: remove consecutive duplicates and blanks
            collapsed = []
            prev = None
            for token in pred_seq:
                if token != prev and token != blank_id:
                    collapsed.append(token)
                prev = token
            
            results.append(collapsed)
        
        return results
    
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
        
        # BF16 mixed precision training (không dùng FP16)
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.logger.info("✅ BF16 mixed precision enabled")
    
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
        use_constant_lr = self.config.get('use_constant_lr_on_resume', False) and self.current_epoch > 1
        
        if use_constant_lr:
            # When resuming, use constant LR to avoid LR dropping to 0
            from torch.optim.lr_scheduler import LambdaLR
            # Constant LR after warmup
            warmup_steps = int(total_steps * warmup_pct)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return 1.0  # Constant LR after warmup

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.logger.info(
                f"Learning rate scheduler: Constant LR={max_lr:.2e} (resume mode), Warmup={warmup_pct*100:.0f}%"
            )
        else:
            # Normal OneCycleLR for fresh training
            div_factor = 100.0
            final_div_factor = 10000.0

            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=warmup_pct,
                anneal_strategy='cos',
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )

            initial_lr = max_lr / div_factor
            self.logger.info(
                f"Learning rate scheduler: Max={max_lr:.2e}, Initial={initial_lr:.2e}, Warmup={warmup_pct*100:.0f}%"
            )
    
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
        
        # Time tracking
        epoch_start_time = time.time()
        batch_times = []
        
        # For WER/CER calculation during training
        log_wer_cer_every = self.config.get('log_wer_cer_every', 10)  # Calculate WER/CER every N batches (reduced for more frequent updates)
        all_predictions = []
        all_references = []
        running_wer = None
        running_cer = None
        
        # Get batch size info for progress bar
        config_batch_size = self.config.get('batch_size', 16)
        effective_batch_size = config_batch_size * self.gradient_accumulation_steps
        
        pbar = tqdm(train_loader, 
                   desc=f'Epoch {self.current_epoch} (batch={config_batch_size}, effective={effective_batch_size})', 
                   file=sys.stderr, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
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
            
            # Get language IDs from batch
            language_ids = batch.get('language_ids', None)
            if language_ids is not None:
                language_ids = language_ids.to(self.device)
            
            # Forward pass với BF16 mixed precision
            if self.use_amp:
                # BF16 mixed precision (không dùng FP16, không cần GradScaler)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, output_lengths, timestamps = self.model(
                        audio_features, audio_lengths, return_timestamps=self.use_timestamps,
                        language_ids=language_ids
                    )
                    
                    # --- DEBUG LOGITS (mỗi 100 batch) ---
                    if batch_idx % 100 == 0:
                        with torch.no_grad():
                            # Lấy logits của mẫu đầu tiên trong batch
                            # Shape: [Time_Steps, Vocab_Size]
                            # Convert về float32 nếu đang ở bfloat16
                            sample_logits = logits[0].detach().cpu().float()
                            
                            # Chuyển sang xác suất (0.0 -> 1.0)
                            probs = torch.softmax(sample_logits, dim=-1)
                            
                            # Lấy ra vài khung hình (Frames) ở giữa để xem
                            num_frames = probs.shape[0]
                            start_frame = num_frames // 2  # Xem đoạn giữa audio
                            
                            self.logger.info(f"\n🔍 --- DEBUG LOGITS (Batch {batch_idx}) ---")
                            for t in range(start_frame, min(start_frame + 5, num_frames)):
                                # Lấy Top 3 token có xác suất cao nhất tại thời điểm t
                                top_probs, top_ids = torch.topk(probs[t], k=3)
                                
                                frame_info = []
                                for p, i in zip(top_probs, top_ids):
                                    # Token ID 0 thường là <pad>/<blank>
                                    blank_id = self.tokenizer.blank_token_id
                                    if i.item() == blank_id:
                                        token_str = "BLANK"
                                    else:
                                        # Thử decode token để xem nó là gì
                                        try:
                                            token_text = self.tokenizer.decode([i.item()])
                                            token_str = f"ID_{i.item()}({token_text[:10]})"
                                        except:
                                            token_str = f"ID_{i.item()}"
                                    frame_info.append(f"{token_str}: {p.item():.4f}")
                                
                                self.logger.info(f"Frame {t}: {' | '.join(frame_info)}")
                            self.logger.info("------------------------------------------\n")
                    # --- KẾT THÚC DEBUG ---
                    
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
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass với BF16 (không cần GradScaler)
                loss.backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip', 1.0)
                    )
                    self.optimizer.step()
            else:
                # Forward pass không dùng AMP
                logits, output_lengths, timestamps = self.model(
                    audio_features, audio_lengths, return_timestamps=self.use_timestamps,
                    language_ids=language_ids
                )
                
                # --- DEBUG LOGITS (mỗi 100 batch) ---
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        # Lấy logits của mẫu đầu tiên trong batch
                        # Shape: [Time_Steps, Vocab_Size]
                        # Convert về float32 nếu đang ở bfloat16
                        sample_logits = logits[0].detach().cpu().float()
                        
                        # Chuyển sang xác suất (0.0 -> 1.0)
                        probs = torch.softmax(sample_logits, dim=-1)
                        
                        # Lấy ra vài khung hình (Frames) ở giữa để xem
                        num_frames = probs.shape[0]
                        start_frame = num_frames // 2  # Xem đoạn giữa audio
                        
                        self.logger.info(f"\n🔍 --- DEBUG LOGITS (Batch {batch_idx}) ---")
                        for t in range(start_frame, min(start_frame + 5, num_frames)):
                            # Lấy Top 3 token có xác suất cao nhất tại thời điểm t
                            top_probs, top_ids = torch.topk(probs[t], k=3)
                            
                            frame_info = []
                            for p, i in zip(top_probs, top_ids):
                                # Token ID 0 thường là <pad>/<blank>
                                blank_id = self.tokenizer.blank_token_id
                                if i.item() == blank_id:
                                    token_str = "BLANK"
                                else:
                                    # Thử decode token để xem nó là gì
                                    try:
                                        token_text = self.tokenizer.decode([i.item()])
                                        token_str = f"ID_{i.item()}({token_text[:10]})"
                                    except:
                                        token_str = f"ID_{i.item()}"
                                frame_info.append(f"{token_str}: {p.item():.4f}")
                            
                            self.logger.info(f"Frame {t}: {' | '.join(frame_info)}")
                        self.logger.info("------------------------------------------\n")
                # --- KẾT THÚC DEBUG ---
                
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
            
            # Collect predictions and references for WER/CER calculation (using Greedy Decode for speed)
            if (batch_idx + 1) % log_wer_cer_every == 0 or batch_idx == len(train_loader) - 1:
                with torch.no_grad():
                    # Use fast greedy decode for training loop (beam search is too slow)
                    pred_tokens_list = self._greedy_ctc_decode(logits, output_lengths)
                    
                    for i, pred_tokens in enumerate(pred_tokens_list):
                        ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                        
                        # Decode
                        if len(pred_tokens) > 0:
                            pred_text = self.tokenizer.decode(pred_tokens)
                        else:
                            pred_text = ""
                        ref_text = self.tokenizer.decode(ref_tokens)
                        
                        all_predictions.append(pred_text)
                        all_references.append(ref_text)
            
            self.callback_manager.on_batch_end(self, num_batches - 1, loss.item())
            
            # Time tracking
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Calculate and log WER/CER periodically
            if (batch_idx + 1) % log_wer_cer_every == 0 and len(all_predictions) > 0:
                running_wer = calculate_wer(all_references, all_predictions)
                running_cer = calculate_cer(all_references, all_predictions)
                # Reset for next period
                all_predictions = []
                all_references = []
            
            # Calculate ETA for epoch
            if len(batch_times) > 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(train_loader) - (batch_idx + 1)
                eta_seconds = avg_batch_time * remaining_batches
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
            else:
                eta_str = "N/A"
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar with all metrics including time and LR
            postfix_dict = {
                'loss': f'{true_loss:.3f}',
                'ctc': f'{true_ctc:.3f}',
                'ts': f'{true_ts:.3f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{batch_time:.2f}s',
                'eta': eta_str
            }
            if running_wer is not None:
                postfix_dict['wer'] = f'{running_wer:.3f}'
            if running_cer is not None:
                postfix_dict['cer'] = f'{running_cer:.3f}'
            pbar.set_postfix(postfix_dict)
            
            # Log với đầy đủ metrics mỗi log_wer_cer_every batches
            # Sử dụng tqdm.write() để tránh làm hỏng progress bar
            if (batch_idx + 1) % log_wer_cer_every == 0:
                elapsed_time = time.time() - epoch_start_time
                elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s" if elapsed_time > 60 else f"{int(elapsed_time)}s"
                if running_wer is not None and running_cer is not None:
                    log_msg = (
                        f"Batch {batch_idx + 1}/{len(train_loader)} [{elapsed_str} | ETA: {eta_str}] - "
                        f"Loss: {true_loss:.4f} (CTC: {true_ctc:.4f}, TS: {true_ts:.4f}) | "
                        f"WER: {running_wer:.4f} | CER: {running_cer:.4f} | "
                        f"LR: {current_lr:.2e} | Batch time: {batch_time:.3f}s"
                    )
                else:
                    log_msg = (
                        f"Batch {batch_idx + 1}/{len(train_loader)} [{elapsed_str} | ETA: {eta_str}] - "
                        f"Loss: {true_loss:.4f} (CTC: {true_ctc:.4f}, TS: {true_ts:.4f}) | "
                        f"LR: {current_lr:.2e} | Batch time: {batch_time:.3f}s"
                    )
                # Sử dụng tqdm.write() để in ra mà không làm hỏng progress bar
                pbar.write(log_msg)
                # Vẫn log vào file
                self.logger.info(log_msg)
            # Log mỗi 10 batches với đầy đủ metrics (chỉ khi không phải log_wer_cer_every)
            elif (batch_idx + 1) % 10 == 0:
                elapsed_time = time.time() - epoch_start_time
                elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s" if elapsed_time > 60 else f"{int(elapsed_time)}s"
                # Tính WER/CER nếu chưa có (từ batch trước) - dùng greedy decode cho nhanh
                if running_wer is None or running_cer is None:
                    # Tính nhanh từ batch hiện tại với greedy decode
                    with torch.no_grad():
                        pred_tokens_quick = self._greedy_ctc_decode(logits, output_lengths)
                        quick_predictions = []
                        quick_references = []
                        for i, pred_tokens in enumerate(pred_tokens_quick):
                            ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                            
                            if len(pred_tokens) > 0:
                                pred_text = self.tokenizer.decode(pred_tokens)
                            else:
                                pred_text = ""
                            ref_text = self.tokenizer.decode(ref_tokens)
                            
                            quick_predictions.append(pred_text)
                            quick_references.append(ref_text)
                        
                        if len(quick_predictions) > 0:
                            running_wer = calculate_wer(quick_references, quick_predictions)
                            running_cer = calculate_cer(quick_references, quick_predictions)
                
                if running_wer is not None and running_cer is not None:
                    log_msg = (
                        f"Batch {batch_idx + 1}/{len(train_loader)} [{elapsed_str} | ETA: {eta_str}] - "
                        f"Loss: {true_loss:.4f} (CTC: {true_ctc:.4f}, TS: {true_ts:.4f}) | "
                        f"WER: {running_wer:.4f} | CER: {running_cer:.4f} | "
                        f"LR: {current_lr:.2e} | Batch time: {batch_time:.3f}s"
                    )
                else:
                    log_msg = (
                        f"Batch {batch_idx + 1}/{len(train_loader)} [{elapsed_str} | ETA: {eta_str}] - "
                        f"Loss: {true_loss:.4f} (CTC: {true_ctc:.4f}, TS: {true_ts:.4f}) | "
                        f"LR: {current_lr:.2e} | Batch time: {batch_time:.3f}s"
                    )
                # Sử dụng tqdm.write() để in ra mà không làm hỏng progress bar
                pbar.write(log_msg)
                # Vẫn log vào file
                self.logger.info(log_msg)
        
        avg_loss = total_loss / num_batches
        avg_ctc = total_ctc_loss / num_batches
        avg_ts = total_timestamp_loss / num_batches
        
        # Calculate final WER/CER for the epoch if we have collected samples
        epoch_wer = None
        epoch_cer = None
        if len(all_predictions) > 0:
            epoch_wer = calculate_wer(all_references, all_predictions)
            epoch_cer = calculate_cer(all_references, all_predictions)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s" if epoch_time > 60 else f"{epoch_time:.1f}s"
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Log epoch summary với đầy đủ metrics và thời gian
        self.logger.info("=" * 80)
        if epoch_wer is not None and epoch_cer is not None:
            self.logger.info(
                f"📊 Epoch {self.current_epoch} Training Summary [{epoch_time_str}]"
            )
            self.logger.info(
                f"   Loss: {avg_loss:.4f} | CTC: {avg_ctc:.4f} | TS: {avg_ts:.4f}"
            )
            self.logger.info(
                f"   WER: {epoch_wer:.4f} | CER: {epoch_cer:.4f}"
            )
            self.logger.info(
                f"   LR: {current_lr:.2e} | Avg batch time: {avg_batch_time:.3f}s | Batches: {num_batches}"
            )
        else:
            self.logger.info(
                f"📊 Epoch {self.current_epoch} Training Summary [{epoch_time_str}]"
            )
            self.logger.info(
                f"   Loss: {avg_loss:.4f} | CTC: {avg_ctc:.4f} | TS: {avg_ts:.4f}"
            )
            self.logger.info(
                f"   LR: {current_lr:.2e} | Avg batch time: {avg_batch_time:.3f}s | Batches: {num_batches}"
            )
        self.logger.info("=" * 80)
        
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
            
            # Get language IDs from batch
            language_ids = batch.get('language_ids', None)
            if language_ids is not None:
                language_ids = language_ids.to(self.device)
            
            # Forward pass with timestamps
            logits, output_lengths, timestamps = self.model(
                audio_features, audio_lengths, return_timestamps=self.use_timestamps,
                language_ids=language_ids
                )
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = self.criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"Val batch {num_batches}: Loss is NaN/Inf, skipping")
                continue
            
            total_loss += loss.item()
            
            # Decode predictions for WER/CER calculation using Greedy Decode
            pred_tokens_list = self._greedy_ctc_decode(logits, output_lengths)
            
            # Decode token IDs to text
            predictions_text = []
            for pred_tokens in pred_tokens_list:
                if len(pred_tokens) > 0:
                    pred_text = self.tokenizer.decode(pred_tokens)
                    predictions_text.append(pred_text)
                else:
                    predictions_text.append("")
            
            # DEBUG: Check first sample in first batch
            if num_batches == 0:
                sample_idx = 0
                ref_tokens_debug = text_tokens[sample_idx, :text_lengths[sample_idx]].cpu().tolist()
                
                # Check logits distribution
                sample_logits = logits[sample_idx, :output_lengths[sample_idx], :].cpu()
                probs = torch.softmax(sample_logits, dim=-1)
                max_probs, max_indices = torch.max(probs, dim=-1)
                
                blank_id = self.tokenizer.blank_token_id
                
                self.logger.info(f"🔍 DEBUG Validation Sample 0:")
                self.logger.info(f"   Decoding method: Greedy Decode")
                self.logger.info(f"   Output length: {output_lengths[sample_idx].item()}")
                
                # Beam search results
                pred_tokens_debug = predictions_text[sample_idx] if sample_idx < len(predictions_text) else []
                self.logger.info(f"   Beam search pred tokens (first 30): {pred_tokens_debug[:30]}")
                self.logger.info(f"   Pred tokens length: {len(pred_tokens_debug)}")
                
                self.logger.info(f"   Max prob values (first 10): {[f'{p:.3f}' for p in max_probs[:10].tolist()]}")
                self.logger.info(f"   Max indices (first 30): {max_indices[:30].tolist()}")
                self.logger.info(f"   Ref tokens (first 20): {ref_tokens_debug[:20]}")
                
                # Decode
                pred_text_debug = self.tokenizer.decode(pred_tokens_debug) if len(pred_tokens_debug) > 0 else ""
                ref_text_debug = self.tokenizer.decode(ref_tokens_debug)
                self.logger.info(f"   Decoded pred: '{pred_text_debug}' (len={len(pred_text_debug)})")
                self.logger.info(f"   Decoded ref: '{ref_text_debug}'")
            
            # Process predictions (Greedy Decode)
            for i in range(len(predictions_text)):
                # Beam search results
                pred_tokens = predictions_text[i] if i < len(predictions_text) else []
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                # Decode greedy tokens
                if len(pred_tokens) > 0:
                    pred_text = self.tokenizer.decode(pred_tokens)
                else:
                    pred_text = ""
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
        
        # Log validation summary với đầy đủ metrics
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Validation (Epoch {self.current_epoch})")
        self.logger.info(f"   Loss: {avg_loss:.4f}")
        self.logger.info(f"   WER: {wer:.4f} | CER: {cer:.4f}")
        self.logger.info(f"   Samples: {len(all_predictions)}")
        if empty_preds > 0:
            self.logger.info(f"   ⚠️  Empty predictions: {empty_preds}/{len(all_predictions)}")
        self.logger.info("=" * 80)
        
        # Print first 10 validation outputs
        print("\n" + "=" * 80)
        print(f"📊 FIRST 10 VALIDATION OUTPUTS (Epoch {self.current_epoch})")
        print("=" * 80)
        num_to_show = min(10, len(all_predictions))
        for i in range(num_to_show):
            ref = all_references[i]
            pred = all_predictions[i]
            # Calculate individual WER/CER for this sample
            sample_wer = calculate_wer([ref], [pred])
            sample_cer = calculate_cer([ref], [pred])
            
            match_indicator = "✅" if ref.strip().lower() == pred.strip().lower() else "❌"
            print(f"\n[{i+1}/{num_to_show}] {match_indicator}")
            print(f"  Ground Truth: {ref}")
            print(f"  Prediction:   {pred}")
            print(f"  WER: {sample_wer:.4f} | CER: {sample_cer:.4f}")
        print("=" * 80 + "\n")
        
        return avg_loss, wer, cer
    
    
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
        
        # Initialize training run metadata (file-based, no database)
        self.training_run_id = f"{self.run_name}_{int(time.time())}"
        
        # Save training metadata to JSON file
        training_metadata = {
            'run_id': self.training_run_id,
            'run_name': self.run_name,
            'model_name': self.config.get('model_name', 'ASR_Base'),
            'model_type': 'transformer_ctc',
            'architecture': 'encoder_ctc',
            'version': '1.0',
            'config': self.config,
            'total_parameters': self.model.get_num_trainable_params(),
            'batch_size': self.config.get('batch_size', 16),
            'learning_rate': self.config.get('learning_rate', 1e-4),
            'num_epochs': num_epochs,
            'optimizer': 'AdamW',
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'start_time': time.time()
        }
        
        metadata_file = Path(self.config.get('checkpoint_dir', 'checkpoints')) / f'{self.training_run_id}_metadata.json'
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Training run initialized: {self.training_run_id}")
        
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
            
            # Calculate total training time and ETA
            total_elapsed = time.time() - start_time
            avg_epoch_time = total_elapsed / self.current_epoch if self.current_epoch > 0 else epoch_time
            remaining_epochs = num_epochs - self.current_epoch
            eta_total_seconds = avg_epoch_time * remaining_epochs
            eta_total_str = f"{int(eta_total_seconds // 3600)}h {int((eta_total_seconds % 3600) // 60)}m" if eta_total_seconds > 3600 else f"{int(eta_total_seconds // 60)}m {int(eta_total_seconds % 60)}s"
            
            epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s" if epoch_time > 60 else f"{epoch_time:.1f}s"
            total_time_str = f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s" if total_elapsed > 3600 else f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"
            
            # Update best metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_wer = wer
            
            # Prepare metrics dictionary for callbacks
            metrics = {
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'wer': wer,
                'cer': cer,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'total_time': total_elapsed,
                'eta_total': eta_total_seconds,
                'timestamp': time.time()
            }
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch completion với đầy đủ thời gian và LR
            self.logger.info("=" * 80)
            self.logger.info(f"✅ Epoch {self.current_epoch}/{num_epochs} Completed")
            self.logger.info(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.logger.info(f"   WER: {wer:.4f} | CER: {cer:.4f}")
            self.logger.info(f"   LR: {current_lr:.2e}")
            self.logger.info(f"   Epoch time: {epoch_time_str} | Total: {total_time_str} | ETA: {eta_total_str}")
            self.logger.info("=" * 80)
            
            # Save metrics to history (file-based logging)
            self.metrics_history.append(metrics)
            
            # Training Layer: Callback on_epoch_end (Checkpoints, Logging, Metrics)
            self.callback_manager.on_epoch_end(self, self.current_epoch, metrics)
            
            # Check early stopping
            if self.should_stop:
                self.logger.info(f'Early stopping triggered at epoch {self.current_epoch}')
                break
        
        total_time = time.time() - start_time
        total_time_str = f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s" if total_time > 3600 else f"{int(total_time // 60)}m {int(total_time % 60)}s"
        
        # Calculate average epoch time
        avg_epoch_time = total_time / self.current_epoch if self.current_epoch > 0 else total_time
        avg_epoch_time_str = f"{int(avg_epoch_time // 60)}m {int(avg_epoch_time % 60)}s" if avg_epoch_time > 60 else f"{avg_epoch_time:.1f}s"
        
        # Save final training results to file (replaces database)
        final_results = {
            'run_id': self.training_run_id,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'best_wer': self.best_wer,
            'best_epoch': self.current_epoch,
            'cer': cer,
            'total_time': total_time,
            'total_time_str': total_time_str,
            'avg_epoch_time': avg_epoch_time,
            'avg_epoch_time_str': avg_epoch_time_str,
            'completed_at': time.time()
        }
        
        results_file = Path(self.config.get('checkpoint_dir', 'checkpoints')) / f'{self.training_run_id}_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Save metrics history
        if self.metrics_history:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to: {results_file}")
        
        # Training Layer: Callback on_train_end
        self.callback_manager.on_train_end(self)
        
        # Log final summary với đầy đủ thời gian
        self.logger.info("=" * 80)
        self.logger.info(f"🎉 TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"Total training time: {total_time_str}")
        self.logger.info(f"Average epoch time: {avg_epoch_time_str}")
        self.logger.info(f"Total epochs: {self.current_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} (Epoch {self.current_epoch})")
        self.logger.info(f"Best WER: {self.best_wer:.4f}")
        self.logger.info(f"Final CER: {cer:.4f}")
        self.logger.info("=" * 80)
    
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
    
    # Load data from merged_dataset manifest files (no database needed)
    dataset_root = config.get('dataset_root', 'data/processed/merged_dataset')
    language = args.language or config.get('language_filter', None)
    
    print(f"📂 Loading data from: {dataset_root}")
    train_df = load_merged_dataset('train', dataset_root, language=language)
    val_df = load_merged_dataset('val', dataset_root, language=language)
    
    # CURRICULUM LEARNING: Filter short sentences for first 5 epochs
    curriculum_enabled = config.get('curriculum_learning', {}).get('enabled', False)
    curriculum_epochs = config.get('curriculum_learning', {}).get('short_sentence_epochs', 5)
    curriculum_max_duration = config.get('curriculum_learning', {}).get('max_duration_seconds', 4.0)
    
    if curriculum_enabled and not args.resume:  # Only apply curriculum when starting fresh
        if 'duration_seconds' in train_df.columns:
            original_len = len(train_df)
            filtered_df = train_df[train_df['duration_seconds'] <= curriculum_max_duration].copy()
            if len(filtered_df) > 0:
                train_df = filtered_df
                print(f"🔥 Curriculum Learning Mode: Filtered to {len(train_df)} short samples (<{curriculum_max_duration}s) from {original_len} total")
                print(f"   Will use full dataset after epoch {curriculum_epochs}")
            else:
                # No short samples found, try with longer duration or disable curriculum
                curriculum_max_duration = 10.0  # Try 10 seconds
                filtered_df = train_df[train_df['duration_seconds'] <= curriculum_max_duration].copy()
                if len(filtered_df) > 0:
                    train_df = filtered_df
                    print(f"🔥 Curriculum Learning Mode: No samples <4s found. Using <{curriculum_max_duration}s instead: {len(train_df)} samples")
                    print(f"   Will use full dataset after epoch {curriculum_epochs}")
                else:
                    print("⚠️  Curriculum learning: No short samples found even with 10s threshold. Disabling curriculum, using full dataset.")
                    curriculum_enabled = False
        else:
            print("⚠️  Curriculum learning requested but 'duration_seconds' column not found. Using full dataset.")
            curriculum_enabled = False
    
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
    if tokenizer_type == 'sentencepiece':
        from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
        model_path = config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
        tokenizer = SentencePieceTokenizer(model_path)
        print(f"✅ Using SentencePiece BPE tokenizer: {model_path} ({len(tokenizer)} tokens)")
    elif tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
        print(f"✅ Using BPE tokenizer: {bpe_path} ({len(tokenizer)} tokens)")
    else:
        # Character-level tokenizer; load vocab from JSON if provided
        char_vocab_path = config.get('char_vocab_path')
        if char_vocab_path:
            vocab_file = Path(char_vocab_path)
            if not vocab_file.exists():
                print(f"⚠️ Char vocab file not found: {vocab_file}, falling back to default tokenizer vocab")
                from preprocessing.text_cleaning import Tokenizer
                tokenizer = Tokenizer()
                print(f"✅ Using default character-level tokenizer ({len(tokenizer)} tokens)")
            else:
                with vocab_file.open("r", encoding="utf-8") as f:
                    vocab_dict = json.load(f)
                max_id = max(vocab_dict.values())
                vocab_list = [None] * (max_id + 1)
                for ch, idx in vocab_dict.items():
                    if 0 <= idx <= max_id:
                        vocab_list[idx] = ch
                for i in range(len(vocab_list)):
                    if vocab_list[i] is None:
                        vocab_list[i] = "<unk>"
                tokenizer = Tokenizer(vocab=vocab_list)
                print(f"✅ Using character-level tokenizer from {vocab_file} ({len(tokenizer)} tokens)")
        else:
            tokenizer = Tokenizer()
            print(f"✅ Using character-level tokenizer ({len(tokenizer)} tokens)")
    
    train_loader, val_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        val_batch_size=config.get('val_batch_size', None),
        num_workers=config.get('num_workers', 4),
        augmenter=augmenter,
        persistent_workers=config.get('persistent_workers', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        sort_by_length=config.get('sort_by_length', True),  # Enable length sorting
        use_bucketing=config.get('use_bucketing', False),  # Optional: use bucketing sampler
        num_buckets=config.get('num_buckets', 10),
        cache_in_ram=config.get('cache_in_ram', False)  # Cache data in RAM (reduces CPU load)
    )
    
    # Initialize trainer (no database needed)
    trainer = ASRTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    else:
        # WEIGHT RESET: Reset decoder projection layer to break "tan tan" bias
        import torch.nn.init as init
        if hasattr(trainer.model, 'decoder') and hasattr(trainer.model.decoder, 'linear'):
            projection_layer = trainer.model.decoder.linear
            init.xavier_uniform_(projection_layer.weight)
            if projection_layer.bias is not None:
                init.constant_(projection_layer.bias, 0.0)
            trainer.logger.info("🧹 Reset decoder projection weights to break 'tan tan' bias")
        elif hasattr(trainer.model, 'decoder') and hasattr(trainer.model.decoder, 'projection'):
            projection_layer = trainer.model.decoder.projection
            init.xavier_uniform_(projection_layer.weight)
            if projection_layer.bias is not None:
                init.constant_(projection_layer.bias, 0.0)
            trainer.logger.info("🧹 Reset decoder projection weights to break 'tan tan' bias")
    
    # Train
    trainer.train(train_loader, val_loader, config.get('num_epochs', 50))


if __name__ == '__main__':
    main()

