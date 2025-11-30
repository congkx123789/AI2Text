#!/usr/bin/env python3
"""
Script để kiểm tra Model Collapse - Model có đang output blank/silence không?

Hiện tượng: Model phát hiện ra "mẹo" để giảm loss nhanh nhất là không đoán gì cả
(hoặc đoán toàn ký tự trắng/blank/silence).

Cách check: In kết quả dự đoán (decode output) của model.
Nếu nó trả về chuỗi rỗng hoặc toàn ký tự giống nhau, model đã bị "sập" (collapse).
"""

import torch
import sys
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders


def check_model_collapse(config_path: str, checkpoint_path: str = None, num_samples: int = 20):
    """
    Kiểm tra model có bị collapse (output blank) không.
    
    Args:
        config_path: Đường dẫn đến file config
        checkpoint_path: Đường dẫn đến checkpoint (nếu None, dùng model mới khởi tạo)
        num_samples: Số samples để kiểm tra
    """
    print("=" * 80)
    print("KIỂM TRA MODEL COLLAPSE")
    print("=" * 80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
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
    
    # Setup model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load checkpoint nếu có
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("Using randomly initialized model (no checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Load data
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    
    print(f"\nChecking {num_samples} samples from training set...")
    print()
    
    # Statistics
    total_samples = 0
    blank_outputs = 0
    empty_outputs = 0
    repeated_outputs = 0
    unique_tokens = set()
    all_predictions = []
    
    # Check samples
    for idx in tqdm(range(min(num_samples, len(train_df))), desc="Checking samples"):
        row = train_df.iloc[idx]
        
        # Load and process audio
        try:
            audio, sr = audio_processor.load_audio(row['file_path'])
            audio = audio_processor.trim_silence(audio)
            mel_spec = audio_processor.extract_mel_spectrogram(audio)
            mel_spec = mel_spec.T  # (time, freq)
            
            # Normalize text
            transcript = row['transcript']
            language = row.get('language', 'vi')
            normalized_text = normalizer.normalize(transcript, lang=language)
            
            # Convert to tensor
            audio_features = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device)
            audio_lengths = torch.tensor([audio_features.size(1)], dtype=torch.long).to(device)
            
            # Forward pass
            with torch.no_grad():
                logits, output_lengths = model(audio_features, audio_lengths)
                
                # Greedy decode
                predictions = torch.argmax(logits, dim=-1)
                pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
            
            # CTC decode
            pred_text = ctc_decode(pred_tokens, tokenizer.blank_token_id, tokenizer)
            
            # Check for collapse
            is_blank = all(t == tokenizer.blank_token_id for t in pred_tokens)
            is_empty = len(pred_text.strip()) == 0
            is_repeated = len(set(pred_tokens)) <= 2  # Only 1-2 unique tokens
            
            # Statistics
            total_samples += 1
            if is_blank:
                blank_outputs += 1
            if is_empty:
                empty_outputs += 1
            if is_repeated:
                repeated_outputs += 1
            
            unique_tokens.update(pred_tokens)
            all_predictions.append(pred_text)
            
            # Print first few samples
            if idx < 5:
                print(f"\nSample {idx + 1}:")
                print(f"  Reference: '{normalized_text}'")
                print(f"  Prediction: '{pred_text}'")
                print(f"  Tokens: {pred_tokens[:20]}..." if len(pred_tokens) > 20 else f"  Tokens: {pred_tokens}")
                print(f"  Unique tokens: {set(pred_tokens)}")
                if is_blank:
                    print(f"  ⚠️  ALL BLANK TOKENS!")
                if is_empty:
                    print(f"  ⚠️  EMPTY OUTPUT!")
                if is_repeated:
                    print(f"  ⚠️  REPEATED OUTPUT!")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 80)
    print(f"Tổng số samples kiểm tra: {total_samples}")
    print(f"Số outputs toàn blank token: {blank_outputs} ({blank_outputs/total_samples*100:.1f}%)")
    print(f"Số outputs rỗng (sau decode): {empty_outputs} ({empty_outputs/total_samples*100:.1f}%)")
    print(f"Số outputs lặp lại (1-2 tokens): {repeated_outputs} ({repeated_outputs/total_samples*100:.1f}%)")
    print(f"Số unique tokens trong tất cả predictions: {len(unique_tokens)}")
    print()
    
    # Diagnosis
    if blank_outputs / total_samples > 0.5:
        print("🚨 CẢNH BÁO: Model đã bị COLLAPSE!")
        print("   Hơn 50% outputs là blank token.")
        print("   Nguyên nhân có thể:")
        print("   1. Learning rate quá cao → Model nhảy vọt vào local minima tồi tệ")
        print("   2. Loss function scale sai → Model học cách output blank để giảm loss")
        print("   3. CTC alignment issue → Output length > Input length sau subsampling")
        print()
        print("   Giải pháp:")
        print("   1. Giảm learning rate xuống 1/10 (ví dụ: 1e-4 → 1e-5)")
        print("   2. Kiểm tra lại loss function (xem check_loss_function.py)")
        print("   3. Kiểm tra output_lengths vs text_lengths trong CTC loss")
        return True
    elif empty_outputs / total_samples > 0.3:
        print("⚠️  CẢNH BÁO: Nhiều outputs rỗng!")
        print("   Model có thể đang học cách output blank để giảm loss.")
        print("   Nên giảm learning rate và kiểm tra lại loss function.")
        return True
    elif len(unique_tokens) < 10:
        print("⚠️  CẢNH BÁO: Model chỉ output rất ít unique tokens!")
        print(f"   Chỉ có {len(unique_tokens)} unique tokens trong tất cả predictions.")
        print("   Model có thể đang bị collapse hoặc learning rate quá cao.")
        return True
    else:
        print("✅ Model không bị collapse.")
        print("   Outputs có đa dạng tokens và không toàn blank.")
        return False


def ctc_decode(tokens: list, blank_token_id: int, tokenizer) -> str:
    """Simple CTC greedy decoding."""
    # Remove consecutive duplicates
    collapsed = []
    prev = None
    for token in tokens:
        if token != prev:
            collapsed.append(token)
            prev = token
    
    # Remove blank tokens
    filtered = [t for t in collapsed if t != blank_token_id]
    
    # Decode to text
    return tokenizer.decode(filtered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if model has collapsed (outputs blank)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to check (default: 20)')
    
    args = parser.parse_args()
    
    has_collapse = check_model_collapse(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples
    )
    
    sys.exit(1 if has_collapse else 0)

