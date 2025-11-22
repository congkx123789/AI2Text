"""
BPE (Byte Pair Encoding) tokenization for Vietnamese.

BPE is a subword tokenization method that breaks words into frequent subword units.
Better for handling out-of-vocabulary (OOV) words and rare words in Vietnamese.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path


class BPETokenizer:
    """
    BPE tokenizer for Vietnamese text.
    
    Implements Byte Pair Encoding algorithm to create subword vocabulary
    that can handle rare and OOV words better than character-level tokenization.
    """
    
    def __init__(self, vocab: Optional[List[str]] = None, merges: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Pre-built vocabulary (if None, builds from scratch)
            merges: Pre-computed BPE merges (if None, builds from scratch)
        """
        self.vocab = vocab or []
        self.merges = merges or []
        self.vocab_to_id = {token: idx for idx, token in enumerate(self.vocab)} if self.vocab else {}
        self.id_to_vocab = {idx: token for token, idx in self.vocab_to_id.items()}
        
        # Special tokens
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.blank_token = '<blank>'  # For CTC
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        self.unk_token_id = self.vocab_to_id.get(self.unk_token, 0)
        self.pad_token_id = self.vocab_to_id.get(self.pad_token, 1)
        self.blank_token_id = self.vocab_to_id.get(self.blank_token, 2)
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """
        Get word frequencies from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            word_freqs: Dictionary mapping words to frequencies
        """
        word_freqs = Counter()
        
        for text in texts:
            # Normalize and split
            words = text.lower().split()
            word_freqs.update(words)
        
        return dict(word_freqs)
    
    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get statistics of symbol pairs.
        
        Args:
            word_freqs: Word frequencies
            
        Returns:
            stats: Dictionary of (pair) -> frequency
        """
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            # Count adjacent pairs
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge a symbol pair in vocabulary.
        
        Args:
            pair: Pair to merge
            word_freqs: Word frequencies
            
        Returns:
            new_word_freqs: Updated word frequencies
        """
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        new_word_freqs = {}
        for word in word_freqs:
            new_word = pattern.sub(''.join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def train(self, texts: List[str], vocab_size: int = 1000, min_frequency: int = 2):
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for words
        """
        # Get word frequencies
        word_freqs = self._get_word_freqs(texts)
        
        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_frequency}
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            vocab.update(list(word))
        
        # Initialize vocabulary as character sequences
        word_freqs = {word: freq for word, freq in word_freqs.items()}
        
        # BPE training loop
        num_merges = vocab_size - len(vocab)
        merges = []
        
        for i in range(num_merges):
            # Get pair statistics
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge pair
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            vocab.add(''.join(best_pair))
            merges.append(best_pair)
        
        # Build final vocabulary
        self.vocab = sorted(list(vocab))
        
        # Add special tokens
        special_tokens = [self.unk_token, self.pad_token, self.blank_token, self.sos_token, self.eos_token]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab.insert(0, token)
        
        self.vocab_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_vocab = {idx: token for token, idx in self.vocab_to_id.items()}
        self.merges = merges
        
        # Set special token IDs
        self.unk_token_id = self.vocab_to_id.get(self.unk_token, 0)
        self.pad_token_id = self.vocab_to_id.get(self.pad_token, 1)
        self.blank_token_id = self.vocab_to_id.get(self.blank_token, 2)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using BPE.
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        # Normalize text
        text = text.lower().strip()
        
        # Split into words
        words = text.split()
        
        token_ids = []
        
        for word in words:
            # Apply BPE merges to word
            word_tokens = self._bpe_tokenize(word)
            
            # Convert tokens to IDs
            for token in word_tokens:
                token_id = self.vocab_to_id.get(token, self.unk_token_id)
                token_ids.append(token_id)
        
        return token_ids
    
    def _bpe_tokenize(self, word: str) -> List[str]:
        """
        Apply BPE tokenization to a word.
        
        Args:
            word: Input word
            
        Returns:
            tokens: List of subword tokens
        """
        # Start with characters
        word = ' '.join(list(word))
        tokens = word.split()
        
        # Apply merges
        for pair in self.merges:
            bigram = ' '.join(pair)
            if bigram in word:
                word = word.replace(bigram, ''.join(pair))
                tokens = word.split()
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            text: Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.id_to_vocab.get(token_id, self.unk_token)
            
            if skip_special_tokens and token in [self.unk_token, self.pad_token, 
                                                   self.blank_token, self.sos_token, self.eos_token]:
                continue
            
            tokens.append(token)
        
        # Join tokens
        text = ''.join(tokens)
        
        # Add spaces between words (heuristic)
        text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)
        
        return text
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        import json
        
        data = {
            'vocab': self.vocab,
            'merges': [list(pair) for pair in self.merges],
            'vocab_to_id': self.vocab_to_id
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load tokenizer from file."""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.vocab_to_id = {k: int(v) for k, v in data['vocab_to_id'].items()}
        self.id_to_vocab = {int(v): k for k, v in self.vocab_to_id.items()}
        
        # Set special token IDs
        self.unk_token_id = self.vocab_to_id.get(self.unk_token, 0)
        self.pad_token_id = self.vocab_to_id.get(self.pad_token, 1)
        self.blank_token_id = self.vocab_to_id.get(self.blank_token, 2)


if __name__ == "__main__":
    # Test BPE tokenizer
    texts = [
        "xin chào việt nam",
        "tôi là sinh viên",
        "hôm nay trời đẹp"
    ]
    
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=100, min_frequency=1)
    
    test_text = "xin chào"
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {len(tokenizer)}")

