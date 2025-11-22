"""
Enhanced training with multi-task learning (CTC + Word2Vec).

Extends ASRTrainer to support:
- Multi-task learning with Word2Vec auxiliary task
- Contextual embeddings training
- Cross-modal attention training
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import time
from pathlib import Path
import yaml
from typing import Optional, Dict, List
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.enhanced_asr import EnhancedASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from training.callbacks import CallbackManager
from utils.metrics import calculate_wer, calculate_cer
from utils.logger import setup_logger


class EnhancedASRTrainer:
    """
    Enhanced trainer with multi-task learning.
    
    Supports:
    - CTC loss (main task)
    - Word2Vec auxiliary loss (optional)
    - Contextual embeddings
    - Cross-modal attention
    """
    
    def __init__(self, config: dict, db: ASRDatabase):
        """
        Initialize enhanced trainer.
        
        Args:
            config: Configuration dictionary
            db: Database instance
        """
        self.config = config
        self.db = db
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('EnhancedASRTrainer', config.get('log_file', 'training.log'))
        
        # Setup components
        self._setup_preprocessing()
        self._setup_model()
        self._setup_optimization()
        self._setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_wer = float('inf')
        self.training_run_id = None
        self.should_stop = False
    
    def _setup_preprocessing(self):
        """Setup preprocessing components."""
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mels=self.config.get('n_mels', 80)
        )
        self.augmenter = AudioAugmenter()
        self.tokenizer = Tokenizer()
        self.normalizer = VietnameseTextNormalizer()
    
    def _setup_model(self):
        """Setup enhanced model."""
        self.model = EnhancedASRModel(
            input_dim=self.config.get('n_mels', 80),
            vocab_size=len(self.tokenizer),
            d_model=self.config.get('d_model', 256),
            num_encoder_layers=self.config.get('num_encoder_layers', 6),
            num_heads=self.config.get('num_heads', 4),
            d_ff=self.config.get('d_ff', 1024),
            dropout=self.config.get('dropout', 0.1),
            use_contextual_embeddings=self.config.get('use_contextual_embeddings', True),
            use_cross_modal_attention=self.config.get('use_cross_modal_attention', True),
            use_word2vec_auxiliary=self.config.get('use_word2vec_auxiliary', False),
            word2vec_dim=self.config.get('word2vec_dim', 256)
        )
        
        self.model.to(self.device)
        
        # Mixed precision
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _setup_optimization(self):
        """Setup optimizer and loss functions."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # CTC Loss (main task)
        self.ctc_criterion = nn.CTCLoss(
            blank=self.tokenizer.blank_token_id,
            zero_infinity=True
        )
        
        # Word2Vec auxiliary loss (optional)
        self.use_word2vec_aux = self.config.get('use_word2vec_auxiliary', False)
        if self.use_word2vec_aux:
            self.word2vec_criterion = nn.MSELoss()
            self.word2vec_weight = self.config.get('word2vec_weight', 0.1)
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        from training.callbacks import (
            CallbackManager,
            CheckpointCallback,
            EarlyStoppingCallback,
            LoggingCallback,
            MetricsCallback
        )
        
        self.callback_manager = CallbackManager()
        
        # Add callbacks
        self.callback_manager.add_callback(
            CheckpointCallback(
                checkpoint_dir=self.config.get('checkpoint_dir', 'checkpoints'),
                save_best=True,
                save_every_n_epochs=self.config.get('save_every', 5)
            )
        )
        self.callback_manager.add_callback(LoggingCallback())
        self.callback_manager.add_callback(MetricsCallback())
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch with multi-task learning."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(audio_features, audio_lengths, 
                                      text_context=text_tokens,  # Contextual biasing
                                      text_lengths=text_lengths)
                    
                    logits = output['logits']
                    logits_t = logits.transpose(0, 1)
                    log_probs = torch.log_softmax(logits_t, dim=-1)
                    
                    # CTC loss (main task)
                    ctc_loss = self.ctc_criterion(
                        log_probs, text_tokens, 
                        output['output_lengths'], text_lengths
                    )
                    
                    # Word2Vec auxiliary loss (optional)
                    total_loss_value = ctc_loss
                    if self.use_word2vec_aux and 'word2vec_embeddings' in output:
                        # Auxiliary task: predict word embeddings
                        # This is a simplified version - in practice, you'd compare
                        # with target word embeddings
                        word2vec_loss = self.word2vec_weight * output['word2vec_embeddings'].mean()
                        total_loss_value = ctc_loss + word2vec_loss
                    
                    loss = total_loss_value
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.get('grad_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(audio_features, audio_lengths,
                                  text_context=text_tokens,
                                  text_lengths=text_lengths)
                
                logits = output['logits']
                logits_t = logits.transpose(0, 1)
                log_probs = torch.log_softmax(logits_t, dim=-1)
                
                ctc_loss = self.ctc_criterion(
                    log_probs, text_tokens,
                    output['output_lengths'], text_lengths
                )
                
                total_loss_value = ctc_loss
                if self.use_word2vec_aux and 'word2vec_embeddings' in output:
                    word2vec_loss = self.word2vec_weight * output['word2vec_embeddings'].mean()
                    total_loss_value = ctc_loss + word2vec_loss
                
                loss = total_loss_value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.config.get('grad_clip', 1.0))
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            self.callback_manager.on_batch_end(self, num_batches - 1, loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader) -> tuple:
        """Validate the model."""
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
            
            output = self.model(audio_features, audio_lengths)
            logits = output['logits']
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = self.ctc_criterion(log_probs, text_tokens,
                                     output['output_lengths'], text_lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Decode
            predictions = torch.argmax(logits, dim=-1)
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output['output_lengths'][i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                pred_text = self._ctc_decode(pred_tokens)
                ref_text = self.tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
        
        avg_loss = total_loss / num_batches
        from utils.metrics import calculate_wer, calculate_cer
        wer = calculate_wer(all_references, all_predictions)
        cer = calculate_cer(all_references, all_predictions)
        
        return avg_loss, wer, cer
    
    def _ctc_decode(self, tokens: list) -> str:
        """CTC greedy decoding."""
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
                prev = token
        
        filtered = [t for t in collapsed if t != self.tokenizer.blank_token_id]
        return self.tokenizer.decode(filtered)
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Main training loop."""
        # Similar to ASRTrainer.train but with enhanced model
        from training.train import ASRTrainer
        
        # Use base trainer logic but with enhanced model
        # For simplicity, use ASRTrainer structure with this model
        
        self.logger.info("Starting enhanced training with multi-task learning...")
        
        # Training loop (simplified - similar structure to ASRTrainer)
        total_steps = len(train_loader) * num_epochs
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('learning_rate', 1e-4),
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Create training run
        model_id = self.db.add_model(
            model_name=self.config.get('model_name', 'EnhancedASR'),
            model_type='enhanced_transformer_ctc',
            architecture='enhanced_encoder_ctc',
            version='1.0',
            config=self.config,
            total_parameters=self.model.get_num_trainable_params()
        )
        
        self.training_run_id = self.db.create_training_run(
            model_id=model_id,
            run_name=self.config.get('run_name', f'enhanced_run_{int(time.time())}'),
            config=self.config,
            batch_size=self.config.get('batch_size', 16),
            learning_rate=self.config.get('learning_rate', 1e-4),
            num_epochs=num_epochs,
            optimizer='AdamW',
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        )
        
        self.callback_manager.on_train_begin(self)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            self.callback_manager.on_epoch_begin(self, self.current_epoch)
            
            train_loss = self.train_epoch(train_loader)
            val_loss, wer, cer = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_wer = wer
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'wer': wer,
                'cer': cer,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            self.callback_manager.on_epoch_end(self, self.current_epoch, metrics)
            
            if self.should_stop:
                break
        
        total_time = time.time() - start_time
        
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
        
        self.callback_manager.on_train_end(self)
        
        self.logger.info(f'Enhanced training completed in {total_time:.2f}s')
        self.logger.info(f'Best WER: {self.best_wer:.4f}')

