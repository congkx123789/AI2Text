"""
SentencePiece BPE Tokenizer wrapper (kiểu Whisper/GPT).

Wrapper tương thích với interface của BPETokenizer để dễ dàng thay thế.
"""

import sentencepiece as spm
from typing import List, Optional
from pathlib import Path


class SentencePieceTokenizer:
    """
    SentencePiece BPE tokenizer wrapper.
    
    Tương thích với interface của BPETokenizer:
    - encode(text) -> List[int]
    - decode(token_ids, skip_special_tokens=True) -> str
    - save(filepath)
    - load(filepath)
    - __len__() -> int
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to .model file (e.g., "models/tokenizer_vi_en_3500.model")
        """
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        
        # Special tokens (mapping to SentencePiece IDs)
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.blank_token = '<blank>'  # For CTC (same as pad in SentencePiece)
        self.sos_token = '<s>'
        self.eos_token = '</s>'
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """
        Load tokenizer from .model file.
        
        Args:
            model_path: Path to .model file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp.load(str(model_path))
        self.model_path = str(model_path)
        
        # Get special token IDs from SentencePiece
        self.unk_token_id = self.sp.unk_id()
        self.pad_token_id = self.sp.pad_id()
        self.blank_token_id = self.sp.pad_id()  # Use pad_id for blank (CTC)
        self.sos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using SentencePiece BPE.
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        if not self.sp:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        
        # SentencePiece tự động normalize và tokenize
        # add_bos=False, add_eos=False: Không thêm BOS/EOS tokens (cho ASR)
        token_ids = self.sp.encode(text, out_type=int, add_bos=False, add_eos=False)
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens (pad, unk, etc.)
            
        Returns:
            text: Decoded text
        """
        if not self.sp:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            filtered_ids = [
                tid for tid in token_ids
                if tid not in [self.pad_token_id, self.blank_token_id, 
                               self.unk_token_id, self.sos_token_id, self.eos_token_id]
            ]
        else:
            filtered_ids = token_ids
        
        # Decode using SentencePiece
        # SentencePiece tự động xử lý ký tự '_' (U+2581) và chuyển thành space
        text = self.sp.decode(filtered_ids)
        return text
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        if not self.sp:
            return 0
        return self.sp.get_piece_size()
    
    def save(self, filepath: str):
        """
        Save tokenizer (just save model path reference).
        
        Note: SentencePiece model is already saved as .model file.
        This method is for compatibility with BPETokenizer interface.
        
        Args:
            filepath: Path to save (not used, model_path is the actual file)
        """
        # SentencePiece model is already saved as .model file
        # This is just for compatibility
        pass
    
    def get_vocab(self) -> List[str]:
        """
        Get vocabulary list.
        
        Returns:
            vocab: List of vocabulary tokens
        """
        if not self.sp:
            return []
        
        vocab = []
        for i in range(self.sp.get_piece_size()):
            vocab.append(self.sp.id_to_piece(i))
        return vocab
    
    def id_to_token(self, token_id: int) -> str:
        """
        Convert token ID to token string.
        
        Args:
            token_id: Token ID
            
        Returns:
            token: Token string
        """
        if not self.sp:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        
        return self.sp.id_to_piece(token_id)
    
    def token_to_id(self, token: str) -> int:
        """
        Convert token string to token ID.
        
        Args:
            token: Token string
            
        Returns:
            token_id: Token ID
        """
        if not self.sp:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        
        return self.sp.piece_to_id(token)


# Alias for backward compatibility
SentencePieceBPETokenizer = SentencePieceTokenizer

