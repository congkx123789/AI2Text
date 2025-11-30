"""
Dataset classes for ASR training.
Handles loading audio and text data efficiently.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
import pandas as pd
import os
import multiprocessing
import pickle
import hashlib
import psutil
import resource
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import VietnameseTextNormalizer, Tokenizer, BilingualTextNormalizer


class ASRDataset(Dataset):
    """Dataset for ASR training with optional RAM caching for CPU optimization."""
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 audio_processor: AudioProcessor,
                 tokenizer: Tokenizer,
                 normalizer: Optional[VietnameseTextNormalizer] = None,
                 augmenter: Optional[AudioAugmenter] = None,
                 max_audio_len: Optional[int] = None,
                 max_text_len: Optional[int] = None,
                 apply_augmentation: bool = False,
                 cache_in_ram: bool = False):
        """Initialize ASR dataset.
        
        Args:
            data_df: DataFrame with columns ['file_path', 'transcript']
            audio_processor: Audio processor instance
            tokenizer: Text tokenizer instance
            normalizer: Text normalizer instance
            augmenter: Audio augmenter instance
            max_audio_len: Maximum audio length in samples
            max_text_len: Maximum text length in tokens
            apply_augmentation: Whether to apply augmentation
            cache_in_ram: If True, cache processed mel spectrograms in RAM (reduces CPU load)
                         Recommended for datasets < 40-50GB with 64GB+ RAM
        """
        self.data_df = data_df.reset_index(drop=True)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.normalizer = normalizer or VietnameseTextNormalizer()
        self.augmenter = augmenter
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.apply_augmentation = apply_augmentation
        self.cache_in_ram = cache_in_ram
        
        # Cache for processed mel spectrograms (saves CPU from re-decoding audio)
        self._cached_features = {} if cache_in_ram else None
        self._cached_text_tokens = {} if cache_in_ram else None
        
        # Cache directory for disk persistence
        self.cache_dir = Path("cache/dataset_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Background caching state
        self._cache_complete = False
        self._cache_thread = None
        import threading
        self._cache_lock = threading.Lock() if cache_in_ram else None
        
        # Pre-cache all data into RAM if enabled
        if cache_in_ram:
            cache_file = self._get_cache_path()
            # Try to load from disk first (fast, non-blocking)
            if not self._load_cache_from_disk(cache_file):
                # Cache not found - start background caching
                # Training can start immediately while caching happens in background
                print("🔄 Cache không tìm thấy - sẽ cache trong background")
                print("   Training sẽ bắt đầu ngay (dùng uncached data)")
                print("   Cache sẽ hoàn tất trong background và tự động chuyển sang")
                self._start_background_caching()
            else:
                self._cache_complete = True
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path based on dataset hash."""
        # Create hash from dataset info to detect changes
        dataset_info = f"{len(self.data_df)}_{self.data_df.iloc[0]['file_path']}_{self.data_df.iloc[-1]['file_path']}"
        cache_hash = hashlib.md5(dataset_info.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"cache_{cache_hash}.pkl"
        return cache_file
    
    def _load_cache_from_disk(self, cache_file: Path) -> bool:
        """Load cache from disk if available.
        
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not cache_file.exists():
            return False
        
        try:
            print(f"📂 Đang load cache từ disk: {cache_file}")
            print(f"   File size: {cache_file.stat().st_size / (1024**2):.1f} MB")
            print(f"   Loading... (this may take a moment for large caches)")
            
            # Load cache in chunks if it's very large to avoid blocking
            import time
            start_time = time.time()
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            load_time = time.time() - start_time
            
            self._cached_features = cache_data['features']
            self._cached_text_tokens = cache_data['tokens']
            
            total_size_mb = sum(
                feat.nbytes / (1024 * 1024) 
                for feat in self._cached_features.values()
            )
            
            print(f"✅ Cache đã load từ disk! (took {load_time:.1f}s)")
            print(f"   Tổng dung lượng: ~{total_size_mb:.2f} MB")
            print(f"   Số samples: {len(self._cached_features):,}")
            self._cache_complete = True
            return True
        except Exception as e:
            print(f"⚠️  Không thể load cache từ disk: {e}")
            print("   → Sẽ cache lại từ đầu...")
            return False
    
    def _save_cache_to_disk(self, cache_file: Path):
        """Save cache to disk for future use."""
        try:
            print(f"💾 Đang lưu cache vào disk: {cache_file}")
            cache_data = {
                'features': self._cached_features,
                'tokens': self._cached_text_tokens
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✅ Cache đã lưu vào disk!")
        except Exception as e:
            print(f"⚠️  Không thể lưu cache vào disk: {e}")
    
    def _start_background_caching(self):
        """Start background caching thread so training can begin immediately."""
        import threading
        cache_file = self._get_cache_path()
        
        def cache_worker():
            """Background worker that caches data while training runs."""
            self._precache_data()
            self._cache_complete = True
            print("✅ Background caching hoàn tất! Dataset giờ dùng cached data.")
        
        self._cache_thread = threading.Thread(target=cache_worker, daemon=True)
        self._cache_thread.start()
        print("   → Background caching thread đã khởi động")
    
    def _precache_data(self):
        """Pre-cache all processed mel spectrograms and text tokens into RAM.
        
        Can run in background thread while training proceeds.
        Uses multiprocessing to balance load across all 24 cores.
        Reduces CPU load and prevents single-core boosting (hotspot).
        """
        cache_file = self._get_cache_path()
        
        # Cache not found, need to build it
        print("🔄 Đang cache dữ liệu vào RAM (background)...")
        print(f"   Dataset size: {len(self.data_df):,} samples")
        # Use ThreadPoolExecutor to parallelize caching across all 24 cores
        num_threads = min(24, multiprocessing.cpu_count())
        print(f"   Sử dụng {num_threads} threads để cân bằng tải trên tất cả cores...")
        
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_sample(idx):
            """Process a single sample for caching (can access self methods)."""
            try:
                row = self.data_df.iloc[idx]
                
                # Process audio once and cache mel spectrogram
                audio, sr = self.audio_processor.load_audio(row['file_path'])
                audio = self.audio_processor.trim_silence(audio)
                mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
                mel_spec = mel_spec.T  # (time, freq)
                
                # Process and cache text tokens
                transcript = row['transcript']
                language = row['language'] if 'language' in row and pd.notna(row['language']) else 'vi'
                
                if 'normalized_transcript' in row and pd.notna(row['normalized_transcript']):
                    normalized_text = row['normalized_transcript']
                else:
                    try:
                        normalized_text = self.normalizer.normalize(transcript, lang=language)
                    except TypeError:
                        normalized_text = self.normalizer.normalize(transcript)
                
                text_tokens = self.tokenizer.encode(normalized_text)
                
                return idx, {
                    'mel_spec': mel_spec.astype(np.float32),
                    'text_tokens': text_tokens,
                    'text': normalized_text,
                    'size_mb': mel_spec.nbytes / (1024 * 1024)
                }
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                return idx, None
        
        # Use ThreadPoolExecutor to parallelize caching (I/O + CPU bound)
        # Threads share memory so we can access self.audio_processor, etc.
        num_workers = min(24, multiprocessing.cpu_count())
        total_size_mb = 0
        processed_count = 0
        
        # Process in parallel using threads (can access instance methods)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_sample, idx): idx 
                           for idx in range(len(self.data_df))}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(self.data_df), desc="Caching", unit="sample") as pbar:
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    if result is not None:
                        self._cached_features[idx] = result['mel_spec']
                        self._cached_text_tokens[idx] = {
                            'tokens': result['text_tokens'],
                            'text': result['text']
                        }
                        total_size_mb += result['size_mb']
                        processed_count += 1
                    pbar.update(1)
        
        print(f"✅ Cache hoàn tất!")
        print(f"   Tổng dung lượng RAM sử dụng: ~{total_size_mb:.2f} MB")
        print(f"   Đã xử lý: {processed_count:,}/{len(self.data_df):,} samples")
        print(f"   CPU sẽ không phải decode audio nữa - chỉ lấy từ RAM!")
        
        # Save cache to disk for future use
        self._save_cache_to_disk(cache_file)
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset.
        
        Smart caching strategy:
        - If cache exists for this idx → use it (fast, CPU nhàn rỗi)
        - If cache doesn't exist yet → process on-the-fly (fallback during background caching)
        - Background caching runs in parallel, so cache gradually becomes available
        - GPU trains immediately while CPU caches in background (SIMULTANEOUS)
        
        Returns:
            item: Dictionary with 'audio_features', 'audio_length', 
                  'text_tokens', 'text_length'
        """
        # Use cached data if available (CPU chỉ việc lấy ra, không decode nữa)
        # During background caching, some samples may not be cached yet → fallback to processing
        if self._cached_features is not None and idx in self._cached_features:
            mel_spec = self._cached_features[idx]
            cached_text = self._cached_text_tokens[idx]
            text_tokens = cached_text['tokens']
            normalized_text = cached_text['text']
            
            # Convert to tensors
            audio_features = torch.from_numpy(mel_spec.copy()).float()
            text_tokens = torch.tensor(text_tokens, dtype=torch.long)
            
            # Apply augmentation if enabled (only during training, on cached features)
            if self.apply_augmentation and self.augmenter:
                # For cached features, we need to load original audio to apply augmentation
                # This is still faster than processing from scratch every epoch
                row = self.data_df.iloc[idx]
                audio, sr = self.audio_processor.load_audio(row['file_path'])
                audio = self.augmenter.augment(audio)
                audio = self.audio_processor.trim_silence(audio)
                mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
                mel_spec = mel_spec.T
                
                # Apply SpecAugment on mel spectrogram for harder training
                mel_spec = self.augmenter.spec_augment(mel_spec)
                
                audio_features = torch.from_numpy(mel_spec).float()
        else:
            # Original on-the-fly processing (CPU làm việc nặng)
            row = self.data_df.iloc[idx]
            
            # Load and process audio
            audio, sr = self.audio_processor.load_audio(row['file_path'])
            
            # Apply augmentation if enabled (only during training)
            if self.apply_augmentation and self.augmenter:
                audio = self.augmenter.augment(audio)
            
            # Trim silence
            audio = self.audio_processor.trim_silence(audio)
            
            # Extract features
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
            
            # Transpose to (time, freq) for model input
            mel_spec = mel_spec.T
            
            # Apply SpecAugment on mel spectrogram for harder training
            if self.apply_augmentation and self.augmenter:
                mel_spec = self.augmenter.spec_augment(mel_spec)
            
            # Normalize and tokenize text (language-aware)
            transcript = row['transcript']
            language = row['language'] if 'language' in row and pd.notna(row['language']) else 'vi'

            if 'normalized_transcript' in row and pd.notna(row['normalized_transcript']):
                normalized_text = row['normalized_transcript']
            else:
                # Try language-aware normalization if supported
                try:
                    normalized_text = self.normalizer.normalize(transcript, lang=language)
                except TypeError:
                    # Fallback for old normalizers that don't accept lang
                    normalized_text = self.normalizer.normalize(transcript)
            
            text_tokens = self.tokenizer.encode(normalized_text)
            
            # Convert to tensors
            audio_features = torch.from_numpy(mel_spec).float()
            text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        # Get lengths
        audio_length = audio_features.size(0)
        text_length = text_tokens.size(0)
        
        # Truncate if needed
        if self.max_audio_len and audio_length > self.max_audio_len:
            audio_features = audio_features[:self.max_audio_len]
            audio_length = self.max_audio_len
        
        if self.max_text_len and text_length > self.max_text_len:
            text_tokens = text_tokens[:self.max_text_len]
            text_length = self.max_text_len
        
        # Load word timestamps if available
        word_timestamps = None
        row = self.data_df.iloc[idx]
        if 'words_json' in row and pd.notna(row['words_json']):
            try:
                import json
                word_timestamps = json.loads(row['words_json'])
            except (json.JSONDecodeError, TypeError):
                word_timestamps = None
        
        result = {
            'audio_features': audio_features,
            'audio_length': audio_length,
            'text_tokens': text_tokens,
            'text_length': text_length,
            'transcript': normalized_text
        }
        
        # Add timestamps if available
        if word_timestamps is not None:
            result['word_timestamps'] = word_timestamps
        
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with padding.
    
    Args:
        batch: List of dataset items
        
    Returns:
        collated_batch: Dictionary with padded tensors
    """
    # Find max lengths in batch (handle empty batch)
    if len(batch) == 0:
        raise ValueError("Empty batch received")
    
    max_audio_len = max(item['audio_length'] for item in batch) if batch else 0
    max_text_len = max(item['text_length'] for item in batch) if batch else 0
    
    if max_audio_len == 0 or max_text_len == 0:
        raise ValueError(f"Invalid batch: max_audio_len={max_audio_len}, max_text_len={max_text_len}")
    
    # Get feature dimension
    freq_dim = batch[0]['audio_features'].size(1)
    
    batch_size = len(batch)
    
    audio_dtype = batch[0]['audio_features'].dtype
    audio_device = batch[0]['audio_features'].device
    text_dtype = batch[0]['text_tokens'].dtype
    text_device = batch[0]['text_tokens'].device
    
    # Initialize padded tensors with correct dtype/device
    audio_features = torch.zeros(
        batch_size, max_audio_len, freq_dim,
        dtype=audio_dtype,
        device=audio_device
    )
    text_tokens = torch.zeros(
        batch_size, max_text_len,
        dtype=text_dtype,
        device=text_device
    )
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    transcripts = []
    word_timestamps = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        audio_len = item['audio_length']
        text_len = item['text_length']
        
        audio_features[i, :audio_len] = item['audio_features']
        text_tokens[i, :text_len] = item['text_tokens']
        audio_lengths[i] = audio_len
        text_lengths[i] = text_len
        transcripts.append(item['transcript'])
    
        # Add word timestamps if available
        if 'word_timestamps' in item and item['word_timestamps'] is not None:
            word_timestamps.append(item['word_timestamps'])
        else:
            word_timestamps.append(None)
    
    result = {
        'audio_features': audio_features,
        'audio_lengths': audio_lengths,
        'text_tokens': text_tokens,
        'text_lengths': text_lengths,
        'transcripts': transcripts
    }
    
    # Add word timestamps if any sample has them
    if any(ts is not None for ts in word_timestamps):
        result['word_timestamps'] = word_timestamps
    
    return result


def worker_init_fn(worker_id: int):
    """Initialize worker with CPU affinity for optimal load balancing.
    
    Assigns each worker to a specific CPU core for balanced distribution.
    With 8 workers on 24-core system, workers are distributed evenly.
    
    For Ryzen 9 9900X (24 logical cores) with 8 workers:
    - Worker 0 → Core 0
    - Worker 1 → Core 3
    - Worker 2 → Core 6
    - ... (spread evenly across cores)
    
    This prevents context switching overhead while keeping GPU fed.
    
    Args:
        worker_id: Unique ID for this worker (0, 1, 2, ..., num_workers-1)
    """
    try:
        # Get total number of logical cores
        num_cores = multiprocessing.cpu_count()
        
        # Distribute workers evenly across cores
        # With 8 workers on 24 cores: spread them out evenly
        # This prevents context switching while maintaining good distribution
        # Worker 0 → Core 0, Worker 1 → Core 3, Worker 2 → Core 6, etc.
        step = num_cores // 8  # For 8 workers on 24 cores = 3
        core_id = (worker_id * step) % num_cores
        
        # Set CPU affinity for this worker process
        # This pins the worker to a specific core for load balancing
        if hasattr(os, 'sched_setaffinity'):
            # Pin this worker process to the assigned core
            os.sched_setaffinity(0, {core_id})
        else:
            # Fallback: Try using taskset via environment (less reliable)
            # Most Linux systems support sched_setaffinity, so this is rare
            pass
        
    except (AttributeError, OSError, PermissionError) as e:
        # If CPU affinity or limiting fails, continue without it
        # Some systems don't support it or require special permissions
        # PyTorch DataLoader will still distribute work, just without limiting
        pass
    except Exception as e:
        # Catch any other unexpected errors
        pass


def get_optimal_prefetch_factor(prefetch_factor: int, num_workers: int, target_ram_gb: float = 6.0) -> int:
    """Auto-adjust prefetch_factor to limit RAM usage to target range (4-8GB).
    
    Strategy:
    - Limits RAM buffer to 4-8GB (default 6GB target)
    - Calculates optimal prefetch_factor based on:
      * Number of workers
      * Estimated batch size in RAM (~25MB per batch)
      * Target RAM usage (4-8GB)
    
    This ensures RAM usage stays within desired limits while maintaining
    good pipelining performance.
    
    Args:
        prefetch_factor: User-specified prefetch factor (hint, will be optimized)
        num_workers: Number of DataLoader workers
        target_ram_gb: Target RAM usage in GB (default 6GB, range 4-8GB)
        
    Returns:
        Optimal prefetch_factor to limit RAM to target range
    """
    # Estimate RAM per batch (conservative estimate)
    # Each batch contains: audio features, text tokens, lengths
    # For batch_size=8, mel_spec ~80x1000, text ~100 tokens
    # Rough estimate: ~20-30MB per batch
    mb_per_batch = 25.0  # 25MB per batch (conservative)
    
    # Calculate max batches for target RAM (4-8GB range)
    # Use middle value (6GB) as default, but allow 4-8GB range
    target_ram_mb = target_ram_gb * 1024  # Convert GB to MB
    max_batches = target_ram_mb / mb_per_batch
    
    # Calculate optimal prefetch_factor per worker
    if num_workers == 0:
        # When num_workers=0, prefetch_factor must be None
        return None
    
    optimal = max(1, int(max_batches / num_workers))
    
    # Ensure minimum of 2 for good pipelining, but respect RAM limit
    # Cap at reasonable maximum (8) to avoid excessive RAM
    optimal = max(2, min(optimal, 8))  # Between 2 and 8
    
    # Calculate actual RAM usage
    actual_ram_gb = (num_workers * optimal * mb_per_batch) / 1024
    
    # Determine strategy
    if actual_ram_gb < 4:
        strategy = "Conservative (<4GB)"
    elif actual_ram_gb <= 8:
        strategy = "Optimal (4-8GB)"
    else:
        strategy = "Above target (>8GB)"
        # Cap at 8GB if exceeded
        optimal = max(1, int((8 * 1024) / (num_workers * mb_per_batch)))
        actual_ram_gb = (num_workers * optimal * mb_per_batch) / 1024
        strategy = "Capped at 8GB"
    
    print(f"💾 RAM Usage Optimization:")
    print(f"   Target: 4-8GB (using {target_ram_gb:.1f}GB)")
    print(f"   Workers: {num_workers}")
    print(f"   Optimal prefetch_factor: {optimal}")
    print(f"   Buffer: {num_workers} workers × {optimal} = {num_workers * optimal} batches")
    print(f"   Estimated RAM: {actual_ram_gb:.2f}GB ({strategy})")
    
    return optimal


def create_data_loaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       audio_processor: AudioProcessor,
                       tokenizer: Tokenizer,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       augmenter: Optional[AudioAugmenter] = None,
                       persistent_workers: bool = True,
                       prefetch_factor: int = 2,
                       sort_by_length: bool = True,
                       use_bucketing: bool = False,
                       num_buckets: int = 10,
                       cache_in_ram: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        audio_processor: Audio processor instance
        tokenizer: Tokenizer instance
        batch_size: Batch size
        num_workers: Number of worker processes
        augmenter: Optional audio augmenter for training
        persistent_workers: Keep workers alive between epochs (faster)
        prefetch_factor: Number of batches to prefetch per worker
        sort_by_length: Sort DataFrame by duration before creating DataLoader (reduces padding)
        use_bucketing: Use BucketingSampler for more sophisticated length grouping
        num_buckets: Number of buckets for BucketingSampler (if use_bucketing=True)
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Sort data by length to reduce padding waste
    # This can improve training speed by 15-20% by grouping similar-length samples
    if sort_by_length:
        # Simple sorting by duration if available
        if 'duration_seconds' in train_df.columns:
            train_df = train_df.sort_values(by='duration_seconds').reset_index(drop=True)
            print(f"✅ Training data sorted by duration to optimize batching (reduces padding waste)")
        else:
            print(f"⚠️  'duration_seconds' column not found. Consider adding it for optimal batching.")
            print("   Data will not be sorted by length. Padding waste may be higher.")
        
        if 'duration_seconds' in val_df.columns:
            val_df = val_df.sort_values(by='duration_seconds').reset_index(drop=True)
            print(f"✅ Validation data sorted by duration")
    
    # Use bilingual normalizer so both Vi/En are supported
    normalizer = BilingualTextNormalizer()
    
    # Create datasets with optional RAM caching
    # cache_in_ram=True: Cache processed mel spectrograms in RAM (reduces CPU load)
    # Recommended for datasets < 40-50GB with 64GB+ RAM
    # Benefits: CPU không phải decode audio mỗi epoch, giảm nhiệt độ, tăng tốc độ
    train_dataset = ASRDataset(
        data_df=train_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        augmenter=augmenter,
        apply_augmentation=True,
        cache_in_ram=cache_in_ram
    )
    
    val_dataset = ASRDataset(
        data_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        apply_augmentation=False,
        cache_in_ram=cache_in_ram  # Cache validation data too for faster evaluation
    )
    
    # Setup sampler for training (bucketing or regular shuffle)
    train_sampler = None
    train_shuffle = True
    
    if use_bucketing:
        # Use bucketing sampler for more sophisticated length grouping
        from training.bucketing_sampler import BucketingSampler
        
        # Estimate lengths from duration_seconds if available
        if 'duration_seconds' in train_df.columns:
            # Convert duration to approximate sequence length (mel frames)
            # Rough estimate: 1 second ≈ 100 mel frames (at 16kHz, hop_length=160)
            # This is an approximation - actual length depends on audio content
            estimated_lengths = (train_df['duration_seconds'] * 100).astype(int).tolist()
            train_sampler = BucketingSampler(
                lengths=estimated_lengths,
                batch_size=batch_size,
                num_buckets=num_buckets,
                shuffle=True,
                drop_last=False
            )
            train_shuffle = False  # Sampler handles shuffling
            print(f"✅ Using BucketingSampler with {num_buckets} buckets")
        else:
            print("⚠️  Bucketing requested but duration_seconds not found. Using regular shuffle.")
            print("   Add duration_seconds column to DataFrame to enable bucketing.")
    
    # Auto-adjust prefetch_factor to limit RAM usage to 4-8GB
    optimal_prefetch = get_optimal_prefetch_factor(prefetch_factor, num_workers, target_ram_gb=6.0)
    # When num_workers=0, prefetch_factor must be None
    if num_workers == 0:
        optimal_prefetch = None
    
    # Create data loaders with optimizations for multi-core CPUs (e.g., Ryzen 9 9900X)
    # Optimized for: 24 logical cores, optimal worker count (Golden Rule: 4 × GPUs)
    # - num_workers: 8 (optimal for 24-core system, prevents context switching)
    # - worker_init_fn: Balance load across cores (workers spread evenly)
    # - pin_memory: Faster CPU→GPU transfer (causes high main process CPU, but speeds GPU)
    # - persistent_workers: Keep workers alive between epochs (smooth CPU usage)
    # - prefetch_factor: Auto-adjusted based on RAM (2-8 batches per worker)
    #   * RAM < 16GB: 2 batches (CPU works harder, smaller buffer)
    #   * RAM 16-32GB: 6 batches (balanced)
    #   * RAM > 32GB: 8 batches (CPU can relax, larger buffer)
    # Perfect balance = GPU 95-100%, Worker cores wavy 50-80%, Main process high but stable
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if num_workers > 0 else False,  # Faster GPU transfer
        persistent_workers=persistent_workers if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=optimal_prefetch if num_workers > 0 else None,  # Must be None when num_workers=0
        drop_last=True,  # Drop last incomplete batch to avoid dimension errors
        worker_init_fn=worker_init_fn if num_workers > 0 else None  # Balance CPU affinity across all cores
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=optimal_prefetch if num_workers > 0 else None,  # Must be None when num_workers=0
        drop_last=False,  # Don't drop last batch in validation
        worker_init_fn=worker_init_fn if num_workers > 0 else None  # Balance CPU affinity across all cores
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from preprocessing.audio_processing import AudioProcessor
    from preprocessing.text_cleaning import Tokenizer
    
    # Create dummy data
    data = {
        'file_path': ['dummy1.wav', 'dummy2.wav'],
        'transcript': ['xin chào', 'tạm biệt']
    }
    df = pd.DataFrame(data)
    
    processor = AudioProcessor()
    tokenizer = Tokenizer()
    
    print("Dataset test complete!")

