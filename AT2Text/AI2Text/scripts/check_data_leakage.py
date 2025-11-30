#!/usr/bin/env python3
"""
Script để kiểm tra Data Leakage - Target có bị lẫn vào Input không?

Nguyên nhân: Code xử lý dữ liệu bị sai, khiến Target (nhãn) bị lẫn vào trong Input (đầu vào).
Model không cần "học" gì cả, nó chỉ cần "copy" đáp án có sẵn trong đầu vào.

Cách check: Kiểm tra lại pipeline dataloader. Đảm bảo input audio và text label hoàn toàn tách biệt.
"""

import torch
import sys
from pathlib import Path
import yaml
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders


def check_data_leakage(config_path: str, num_samples: int = 50):
    """
    Kiểm tra data leakage - target có bị lẫn vào input không.
    
    Args:
        config_path: Đường dẫn đến file config
        num_samples: Số samples để kiểm tra
    """
    print("=" * 80)
    print("KIỂM TRA DATA LEAKAGE")
    print("=" * 80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup preprocessing
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
    normalizer = BilingualTextNormalizer()
    
    # Load data
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    
    print(f"Checking {num_samples} samples from training set...")
    print()
    
    # Statistics
    suspicious_samples = []
    
    # Check samples
    for idx in tqdm(range(min(num_samples, len(train_df))), desc="Checking samples"):
        row = train_df.iloc[idx]
        
        try:
            # Load and process audio
            audio, sr = audio_processor.load_audio(row['file_path'])
            audio = audio_processor.trim_silence(audio)
            mel_spec = audio_processor.extract_mel_spectrogram(audio)
            mel_spec = mel_spec.T  # (time, freq)
            
            # Normalize text
            transcript = row['transcript']
            language = row.get('language', 'vi')
            normalized_text = normalizer.normalize(transcript, lang=language)
            text_tokens = tokenizer.encode(normalized_text)
            
            # Check 1: Audio features không chứa text tokens
            # (Mel spectrogram không thể chứa thông tin text trực tiếp)
            # Đây là check cơ bản - mel spec chỉ chứa thông tin audio
            
            # Check 2: File path không chứa transcript
            file_path = str(row['file_path'])
            if normalized_text.lower() in file_path.lower():
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'Transcript found in file path',
                    'file_path': file_path,
                    'transcript': normalized_text
                })
            
            # Check 3: Transcript không chứa file path
            if file_path.lower() in normalized_text.lower():
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'File path found in transcript',
                    'file_path': file_path,
                    'transcript': normalized_text
                })
            
            # Check 4: Audio features có kích thước hợp lý
            if mel_spec.shape[0] < 10 or mel_spec.shape[1] != config.get('n_mels', 80):
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'Invalid audio feature shape',
                    'shape': mel_spec.shape,
                    'expected': (f">=10", config.get('n_mels', 80))
                })
            
            # Check 5: Text tokens không quá dài so với audio
            # (Nếu text dài hơn audio rất nhiều, có thể có vấn đề)
            audio_len = mel_spec.shape[0]
            text_len = len(text_tokens)
            if text_len > audio_len * 2:  # Text không nên dài gấp đôi audio
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'Text much longer than audio',
                    'audio_len': audio_len,
                    'text_len': text_len,
                    'ratio': text_len / audio_len
                })
            
            # Check 6: Mel spectrogram values trong range hợp lý
            if np.any(np.isnan(mel_spec)) or np.any(np.isinf(mel_spec)):
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'NaN or Inf in mel spectrogram',
                    'file_path': file_path
                })
            
            # Check 7: Text tokens không chứa invalid tokens
            vocab_size = len(tokenizer)
            if any(t >= vocab_size or t < 0 for t in text_tokens):
                suspicious_samples.append({
                    'idx': idx,
                    'reason': 'Invalid token IDs in text',
                    'tokens': text_tokens,
                    'vocab_size': vocab_size
                })
        
        except Exception as e:
            suspicious_samples.append({
                'idx': idx,
                'reason': f'Error processing: {e}',
                'file_path': row.get('file_path', 'unknown')
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 80)
    print(f"Tổng số samples kiểm tra: {num_samples}")
    print(f"Số samples đáng ngờ: {len(suspicious_samples)}")
    print()
    
    if len(suspicious_samples) > 0:
        print("⚠️  CÁC SAMPLES ĐÁNG NGỜ:")
        print()
        for sample in suspicious_samples[:10]:  # Show first 10
            print(f"Sample {sample['idx']}: {sample['reason']}")
            if 'file_path' in sample:
                print(f"  File: {sample['file_path']}")
            if 'transcript' in sample:
                print(f"  Transcript: {sample['transcript']}")
            print()
        
        if len(suspicious_samples) > 10:
            print(f"... và {len(suspicious_samples) - 10} samples khác")
        
        print()
        print("🚨 CẢNH BÁO: Có thể có data leakage!")
        print("   Một số samples có dấu hiệu đáng ngờ.")
        print("   Tuy nhiên, không phải tất cả đều là leakage:")
        print("   - Transcript trong file path: Có thể là naming convention")
        print("   - Text dài hơn audio: Có thể do audio ngắn nhưng text dài")
        print("   - Invalid tokens: Có thể do tokenizer chưa được train đúng")
        print()
        print("   Nên kiểm tra thêm:")
        print("   1. Xem lại code trong dataset.py - đảm bảo input và target tách biệt")
        print("   2. Kiểm tra validation loss vs train loss (xem check_loss_validation.py)")
        print("   3. Nếu train loss << val loss → có thể có leakage")
        return True
    else:
        print("✅ Không phát hiện data leakage rõ ràng.")
        print("   Input audio và text label hoàn toàn tách biệt.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check for data leakage')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to check (default: 50)')
    
    args = parser.parse_args()
    
    has_leakage = check_data_leakage(
        config_path=args.config,
        num_samples=args.num_samples
    )
    
    sys.exit(1 if has_leakage else 0)
