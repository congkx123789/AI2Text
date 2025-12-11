"""
BPE (Byte Pair Encoding) tokenization for Vietnamese.

BPE is a subword tokenization method that breaks words into frequent subword units.
Better for handling out-of-vocabulary (OOV) words and rare words in Vietnamese.
"""

import re
import gc
import sys
import signal
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import numpy as np

# Global flag for graceful exit
INTERRUPT_RECEIVED = False

def signal_handler(sig, frame):
    global INTERRUPT_RECEIVED
    print("\n🛑 Đã nhận lệnh dừng! Đang lưu dữ liệu và thoát an toàn...")
    INTERRUPT_RECEIVED = True

signal.signal(signal.SIGINT, signal_handler)


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
        self.space_token = ' '  # Space token for word separation
        
        self.unk_token_id = self.vocab_to_id.get(self.unk_token, 0)
        self.pad_token_id = self.vocab_to_id.get(self.pad_token, 1)
        self.blank_token_id = self.vocab_to_id.get(self.blank_token, 2)
        self.space_token_id = self.vocab_to_id.get(self.space_token, None)
    
    def _process_text_batch(self, batch: List[str]) -> Dict[str, int]:
        """Process a batch of texts to get word frequencies (for multiprocessing)."""
        word_freqs = Counter()
        for text in batch:
            words = text.lower().split()
            word_freqs.update(words)
        return dict(word_freqs)
    
    def _get_word_freqs(self, texts: List[str], num_workers: int = None) -> Dict[str, int]:
        """
        Get word frequencies from texts (multiprocessing optimized for full CPU usage).
        
        Args:
            texts: List of text strings
            num_workers: Number of parallel workers (default: None = use all CPU cores)
            
        Returns:
            word_freqs: Dictionary mapping words to frequencies
        """
        if num_workers is None:
            num_workers = cpu_count()  # Use all available CPU cores
        
        if len(texts) < 10000:
            # For small datasets, use single process
            word_freqs = Counter()
            for text in texts:
                words = text.lower().split()
                word_freqs.update(words)
            return dict(word_freqs)
        
        # Optimized batch processing - use ProcessPoolExecutor for FULL CPU utilization
        # Use smaller batches to maximize parallelism and CPU usage
        batch_size = max(500, len(texts) // (num_workers * 8))  # More batches = better CPU utilization
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        print(f"   Processing {len(batches):,} batches with {num_workers} workers...")
        
        # Use ProcessPoolExecutor for CPU-bound tasks (better than Pool for large datasets)
        word_freqs = Counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches at once for maximum parallelism
            future_to_batch = {executor.submit(self._process_text_batch, batch): batch for batch in batches}
            
            # Collect results as they complete (parallel execution)
            completed = 0
            for future in as_completed(future_to_batch):
                if INTERRUPT_RECEIVED:
                    break
                try:
                    result = future.result()
                    word_freqs.update(result)
                    completed += 1
                    if completed % 100 == 0:
                        print(f"   Word frequency: {completed}/{len(batches)} batches processed", end='\r')
                except Exception as e:
                    print(f"⚠️  Error processing batch: {e}")
                    continue
        
        print()  # New line after progress
        return dict(word_freqs)
    
    def _process_pair_stats_batch(self, batch: List[Tuple[str, int]]) -> Dict[Tuple[str, str], int]:
        """Process a batch of words to get pair statistics (for multiprocessing)."""
        pairs = defaultdict(int)
        for word, freq in batch:
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pairs[(symbols[j], symbols[j + 1])] += freq
        return dict(pairs)
    
    def _get_stats(self, word_freqs: Dict[str, int], num_workers: int = None) -> Dict[Tuple[str, str], int]:
        """
        Get statistics of symbol pairs (multiprocessing optimized for full CPU usage).
        
        Args:
            word_freqs: Word frequencies (words are space-separated character sequences)
            num_workers: Number of parallel workers (default: None = use all CPU cores)
            
        Returns:
            stats: Dictionary of (pair) -> frequency
        """
        if num_workers is None:
            num_workers = cpu_count()  # Use all available CPU cores
        
        items = list(word_freqs.items())
        
        if len(items) < 5000:
            # For small datasets, use single process
            pairs = defaultdict(int)
            for word, freq in items:
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
            return dict(pairs)
        
        # Optimized batch processing - use ProcessPoolExecutor for FULL CPU utilization
        # Use smaller batches to maximize parallelism
        batch_size = max(200, len(items) // (num_workers * 4))  # More batches = better CPU utilization
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        pairs = defaultdict(int)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches at once for maximum parallelism
            future_to_batch = {executor.submit(self._process_pair_stats_batch, batch): batch for batch in batches}
            
            # Collect results as they complete (parallel execution)
            for future in as_completed(future_to_batch):
                if INTERRUPT_RECEIVED:
                    break
                try:
                    result = future.result()
                    for pair, freq in result.items():
                        pairs[pair] += freq
                except Exception as e:
                    print(f"⚠️  Error processing batch: {e}")
                    continue
        
        return dict(pairs)
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge a symbol pair in vocabulary (highly optimized).
        
        Args:
            pair: Pair to merge
            word_freqs: Word frequencies
            
        Returns:
            new_word_freqs: Updated word frequencies
        """
        bigram = ' '.join(pair)
        merged = ''.join(pair)
        
        # Highly optimized: use dict comprehension with string replace
        # Pre-compute replacement for better cache performance
        new_word_freqs = {
            word.replace(bigram, merged): freq 
            for word, freq in word_freqs.items()
        }
        
        return new_word_freqs
    
    def train(self, texts: List[str], vocab_size: int = 1000, min_frequency: int = 2, num_workers: int = None):
        """
        Train BPE tokenizer on texts (multiprocessing optimized for full CPU usage).
        
        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for words
            num_workers: Number of parallel workers (default: None = use all CPU cores)
        """
        if num_workers is None:
            num_workers = cpu_count()  # Use all available CPU cores
        
        # For maximum CPU utilization, we can use more workers than cores for I/O-bound parts
        # But for CPU-bound tasks, use exactly the number of cores
        actual_workers = num_workers
        
        print(f"🚀 Using {actual_workers} parallel workers (ALL {cpu_count()} CPU cores/threads)")
        print(f"⚡ Maximum CPU utilization enabled - all cores will be used")
        
        # Get word frequencies (parallelized)
        print("📊 Computing word frequencies (parallelized)...")
        word_freqs = self._get_word_freqs(texts, num_workers=num_workers)
        
        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_frequency}
        
        # Initialize vocabulary with characters (split words into characters)
        vocab = set()
        for word in word_freqs:
            vocab.update(list(word))
        
        # Convert words to character sequences for BPE
        # Format: "word" -> "w o r d" (space-separated characters)
        word_freqs_chars = {}
        for word, freq in word_freqs.items():
            word_freqs_chars[' '.join(list(word))] = freq
        
        # BPE training loop (optimized for Ryzen 9 9900X)
        num_merges = vocab_size - len(vocab)
        merges = []
        
        print(f"Starting BPE training: {num_merges:,} merges needed")
        print(f"Initial vocab size: {len(vocab)}")
        print(f"Processing {len(word_freqs_chars):,} unique words...")
        print(f"CPU cores available: {cpu_count()} (using {num_workers} workers)")
        print()
        
        start_time = time.time()
        last_report_time = start_time
        last_gc_time = start_time
        
        # Use tqdm for better progress display if available
        try:
            from tqdm import tqdm
            use_tqdm = True
            pbar = tqdm(total=num_merges, desc="BPE Training", unit="merge", ncols=100)
        except ImportError:
            use_tqdm = False
            pbar = None
        
        for i in range(num_merges):
            if INTERRUPT_RECEIVED:
                print(f"\n🛑 Training interrupted at merge {i}/{num_merges}")
                break
            
            # Progress reporting (every 50 merges or every 5 seconds)
            current_time = time.time()
            if i % 50 == 0 or (current_time - last_report_time) >= 5.0:
                progress = (i / num_merges * 100) if num_merges > 0 else 0
                elapsed = current_time - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (num_merges - i) / rate if rate > 0 else 0
                
                if use_tqdm:
                    pbar.set_postfix({
                        'vocab': len(vocab),
                        'speed': f'{rate:.1f}/s',
                        'eta': f'{eta/60:.1f}m'
                    })
                    pbar.update(50 if i > 0 else 0)
                else:
                    print(f"Progress: {i:,}/{num_merges:,} ({progress:.1f}%) | "
                          f"Vocab: {len(vocab):,} | "
                          f"Speed: {rate:.1f} merges/s | "
                          f"ETA: {eta/60:.1f} min", end='\r')
                last_report_time = current_time
            
            # Periodic garbage collection (every 1000 merges or every 30 seconds)
            if (i % 1000 == 0 and i > 0) or (current_time - last_gc_time) >= 30.0:
                gc.collect()
                last_gc_time = current_time
            
            # Get pair statistics (parallelized)
            pairs = self._get_stats(word_freqs_chars, num_workers=num_workers)
            
            if not pairs:
                if use_tqdm:
                    pbar.close()
                print(f"\nStopped early: no more pairs to merge at iteration {i}")
                break
            
            # Get most frequent pair (optimized - use max with key function)
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Merge pair
            merged_token = ''.join(best_pair)
            
            # Skip tokens with special space characters (Ġ, ▁, _ prefix)
            # We don't want these special characters in our vocabulary
            # Instead, we use regular space ' ' token
            if 'Ġ' in merged_token or '▁' in merged_token or (merged_token.startswith('_') and len(merged_token) > 1 and merged_token != '_'):
                # Skip this merge - remove from pairs and try next best
                pairs.pop(best_pair)
                if len(pairs) == 0:
                    break
                # Get next best pair
                best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                merged_token = ''.join(best_pair)
            
            word_freqs_chars = self._merge_vocab(best_pair, word_freqs_chars)
            vocab.add(merged_token)
            merges.append(best_pair)
        
        if use_tqdm:
            pbar.close()
        
        total_time = time.time() - start_time
        print(f"\n✅ BPE training completed in {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        
        # Final cleanup
        gc.collect()
        
        # Build final vocabulary
        self.vocab = sorted(list(vocab))
        
        # Filter out any tokens with special space characters (Ġ, ▁, _ prefix)
        # These are used by other tokenizers (GPT, SentencePiece) but we don't want them
        filtered_vocab = []
        for token in self.vocab:
            # Skip tokens with special space characters
            if 'Ġ' in token or '▁' in token or (token.startswith('_') and len(token) > 1):
                continue
            filtered_vocab.append(token)
        self.vocab = filtered_vocab
        
        # Add special tokens (including space token)
        # We use regular space ' ' instead of special characters like Ġ or _
        special_tokens = [self.unk_token, self.pad_token, self.blank_token, self.sos_token, self.eos_token, self.space_token]
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
        self.space_token_id = self.vocab_to_id.get(self.space_token, None)
    
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
        
        # Get space token ID (ensure it exists in vocab)
        if self.space_token_id is None:
            self.space_token_id = self.vocab_to_id.get(self.space_token)
            if self.space_token_id is None:
                # If space token doesn't exist, add it to vocab
                if self.space_token not in self.vocab:
                    self.vocab.append(self.space_token)
                    self.vocab_to_id[self.space_token] = len(self.vocab) - 1
                    self.id_to_vocab[len(self.vocab) - 1] = self.space_token
                self.space_token_id = self.vocab_to_id.get(self.space_token)
        
        for i, word in enumerate(words):
            # Apply BPE merges to word
            word_tokens = self._bpe_tokenize(word)
            
            # Convert tokens to IDs
            for token in word_tokens:
                token_id = self.vocab_to_id.get(token, self.unk_token_id)
                token_ids.append(token_id)
            
            # Insert space token after each word (except the last one)
            if i < len(words) - 1 and self.space_token_id is not None:
                token_ids.append(self.space_token_id)
        
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
            
            # Handle space token: convert to actual space character
            if token == self.space_token:
                tokens.append(' ')
            else:
                tokens.append(token)
        
        # Join tokens (spaces are now preserved as actual space characters)
        text = ''.join(tokens)
        
        # Clean up multiple spaces (just in case)
        text = re.sub(r' +', ' ', text).strip()
        
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
        self.space_token_id = self.vocab_to_id.get(self.space_token, None)
        
        # Ensure space token exists in vocab if loading from file
        if self.space_token_id is None and self.space_token not in self.vocab:
            # Add space token to vocab if it doesn't exist
            self.vocab.append(self.space_token)
            self.vocab_to_id[self.space_token] = len(self.vocab) - 1
            self.id_to_vocab[len(self.vocab) - 1] = self.space_token
            self.space_token_id = self.vocab_to_id.get(self.space_token)


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

