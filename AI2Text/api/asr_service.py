"""
ASR Service for loading and running inference with trained models.
Supports ASRModelWithTimestamps and BPE tokenizer.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import yaml
import json
import logging

logger = logging.getLogger(__name__)


class ASRService:
    """Service for loading and running ASR inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize ASR service.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.audio_processor = None
        
        # Load model and components
        self._load_model()
        self._load_tokenizer()
        self._load_audio_processor()
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(str(self.checkpoint_path), map_location='cpu', weights_only=False)
        self.config = checkpoint.get('config', {})
        
        # Get model parameters from config
        input_dim = self.config.get('n_mels', 80)
        vocab_size = self.config.get('vocab_size', 2000)
        d_model = self.config.get('d_model', 320)
        num_encoder_layers = self.config.get('num_encoder_layers', 16)
        num_heads = self.config.get('num_heads', 4)
        d_ff = self.config.get('d_ff', 1280)
        dropout = self.config.get('dropout', 0.1)
        use_timestamps = self.config.get('use_timestamps', True)
        
        # Import model class
        from models.asr_with_timestamps import ASRModelWithTimestamps
        
        # Create model
        self.model = ASRModelWithTimestamps(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            predict_timestamps=use_timestamps
        )
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
        if state_dict:
            try:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                raise
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_tokenizer(self):
        """Load tokenizer based on config."""
        tokenizer_type = self.config.get('tokenizer_type', 'bpe')
        
        if tokenizer_type == 'bpe':
            from preprocessing.bpe_tokenizer import BPETokenizer
            bpe_path = self.config.get('bpe_vocab_path', 'models/bilingual_bpe_2k.json')
            
            # Resolve path relative to project root
            project_root = Path(__file__).parent.parent
            bpe_path = project_root / bpe_path
            
            if not bpe_path.exists():
                raise FileNotFoundError(f"BPE vocab not found: {bpe_path}")
            
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(str(bpe_path))
            logger.info(f"BPE tokenizer loaded from {bpe_path} ({len(self.tokenizer)} tokens)")
        else:
            from preprocessing.text_cleaning import Tokenizer
            
            # Try to load char vocab if specified
            char_vocab_path = self.config.get('char_vocab_path')
            if char_vocab_path:
                project_root = Path(__file__).parent.parent
                vocab_file = project_root / char_vocab_path
                if vocab_file.exists():
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
                    self.tokenizer = Tokenizer(vocab=vocab_list)
                    logger.info(f"Character tokenizer loaded from {vocab_file} ({len(self.tokenizer)} tokens)")
                else:
                    self.tokenizer = Tokenizer()
                    logger.info(f"Character vocab not found, using default tokenizer ({len(self.tokenizer)} tokens)")
            else:
                self.tokenizer = Tokenizer()
                logger.info(f"Character tokenizer initialized ({len(self.tokenizer)} tokens)")
    
    def _load_audio_processor(self):
        """Load audio processor."""
        from preprocessing.audio_processing import AudioProcessor
        
        sample_rate = self.config.get('sample_rate', 16000)
        n_mels = self.config.get('n_mels', 80)
        
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        logger.info(f"Audio processor initialized (sr={sample_rate}, n_mels={n_mels})")
    
    @torch.no_grad()
    def transcribe(self, 
                   audio_path: str,
                   return_timestamps: bool = False,
                   language_id: Optional[int] = None) -> Dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            return_timestamps: If True, return word-level timestamps
            language_id: Language ID (0=Vietnamese, 1=English), None for auto-detect
            
        Returns:
            Dictionary with:
                - text: Transcribed text
                - timestamps: Optional word-level timestamps (if return_timestamps=True)
                - confidence: Optional confidence score
        """
        # Load and process audio
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        
        # Extract mel spectrogram features
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio_data)
        
        # Prepare input: (time_frames, n_mels) -> (1, time_frames, n_mels)
        features = torch.from_numpy(mel_spec.T).unsqueeze(0).float().to(self.device)
        lengths = torch.tensor([features.size(1)]).to(self.device)
        
        # Prepare language IDs if specified
        language_ids = None
        if language_id is not None:
            language_ids = torch.tensor([language_id]).to(self.device)
        
        # Forward pass
        logits, output_lengths, timestamps = self.model(
            features,
            lengths,
            return_timestamps=return_timestamps,
            language_ids=language_ids
        )
        
        # Decode text
        predictions = torch.argmax(logits, dim=-1)
        pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
        
        # CTC decode: remove blanks and duplicates
        collapsed = []
        prev = None
        for token in pred_tokens:
            if token != prev and token != self.tokenizer.blank_token_id:
                collapsed.append(token)
            prev = token
        
        # Decode to text
        text = self.tokenizer.decode(collapsed)
        
        result = {
            'text': text,
            'confidence': None  # Can add confidence scoring if needed
        }
        
        # Add timestamps if requested and available
        if return_timestamps and timestamps is not None:
            word_timestamps = self._extract_word_timestamps(
                logits, timestamps, collapsed, output_lengths[0].item()
            )
            result['timestamps'] = word_timestamps
        
        return result
    
    def _extract_word_timestamps(self,
                                 logits: torch.Tensor,
                                 timestamps: torch.Tensor,
                                 tokens: List[int],
                                 output_length: int) -> List[Dict[str, Any]]:
        """Extract word-level timestamps.
        
        Args:
            logits: Model logits (1, time, vocab_size)
            timestamps: Frame-level timestamps (1, time, 2)
            tokens: Decoded token IDs
            output_length: Output sequence length
            
        Returns:
            List of dicts with 'word', 'start', 'end'
        """
        # Get frame-level timestamps
        frame_timestamps = timestamps[0, :output_length, :].cpu().numpy()  # (time, 2)
        
        # Map tokens to frames (simplified - assumes each token corresponds to a frame)
        # In practice, you'd need to align tokens to frames using CTC alignment
        word_timestamps = []
        
        # Simple approach: distribute timestamps evenly across tokens
        # For production, use proper CTC alignment
        if len(tokens) > 0:
            frames_per_token = max(1, output_length // len(tokens))
            
            for i, token_id in enumerate(tokens):
                start_frame = min(i * frames_per_token, output_length - 1)
                end_frame = min((i + 1) * frames_per_token, output_length)
                
                if start_frame < len(frame_timestamps) and end_frame <= len(frame_timestamps):
                    start_time = float(frame_timestamps[start_frame, 0])
                    end_time = float(frame_timestamps[end_frame - 1, 1])
                    
                    # Decode token to text
                    token_text = self.tokenizer.decode([token_id])
                    
                    if token_text.strip():  # Only add non-empty tokens
                        word_timestamps.append({
                            'word': token_text,
                            'start': start_time,
                            'end': end_time
                        })
        
        return word_timestamps
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of transcription results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                results.append({
                    'text': '',
                    'error': str(e)
                })
        return results

