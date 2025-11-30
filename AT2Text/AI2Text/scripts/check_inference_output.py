#!/usr/bin/env python3
"""
Script để kiểm tra inference output ngay lập tức.
Kiểm tra xem model có output blank/empty không.
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


def check_inference_output(config_path: str, checkpoint_path: str, num_samples: int = 10):
    """
    Kiểm tra inference output để debug WER=1.0.
    
    Args:
        config_path: Đường dẫn đến config file
        checkpoint_path: Đường dẫn đến checkpoint
        num_samples: Số samples để kiểm tra
    """
    print("=" * 80)
    print("KIỂM TRA INFERENCE OUTPUT (Debug WER=1.0)")
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
    
    # Load checkpoint (load to CPU first to avoid OOM if GPU is busy)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print()
    
    # Move model to device after loading
    model.to(device)
    model.eval()
    
    # Load data
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    split_version = config.get('split_version', 'v1')
    val_df = db.get_split_data('val', split_version)
    
    print(f"Checking {num_samples} samples from validation set...")
    print()
    
    # Statistics
    empty_outputs = 0
    blank_only_outputs = 0
    valid_outputs = 0
    
    # Check samples
    for idx in tqdm(range(min(num_samples, len(val_df))), desc="Checking"):
        row = val_df.iloc[idx]
        
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
            
            # Check
            is_empty = len(pred_text.strip()) == 0
            unique_tokens = set(pred_tokens)
            is_blank_only = len(unique_tokens) <= 1 and (len(unique_tokens) == 0 or list(unique_tokens)[0] == tokenizer.blank_token_id)
            
            if is_empty:
                empty_outputs += 1
            elif is_blank_only:
                blank_only_outputs += 1
            else:
                valid_outputs += 1
            
            # Print details
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}:")
            print(f"{'='*80}")
            print(f"📝 Reference:")
            print(f"   Text: '{normalized_text}'")
            print(f"   Tokens: {text_tokens[:20]}..." if len(text_tokens) > 20 else f"   Tokens: {text_tokens}")
            print(f"   Length: {len(text_tokens)} tokens")
            print(f"🤖 Prediction:")
            print(f"   Text: '{pred_text}'")
            print(f"   Tokens: {pred_tokens[:30]}..." if len(pred_tokens) > 30 else f"   Tokens: {pred_tokens}")
            print(f"   Length: {len(pred_tokens)} tokens")
            print(f"📊 Analysis:")
            print(f"   Output length: {output_lengths[0].item()}")
            print(f"   Text length: {len(text_tokens)}")
            print(f"   Output/Text ratio: {output_lengths[0].item() / len(text_tokens):.2f}x" if len(text_tokens) > 0 else "   N/A")
            print(f"   Unique pred tokens: {unique_tokens}")
            print(f"   Number of unique tokens: {len(unique_tokens)}")
            
            if is_empty:
                print(f"   🚨 PREDICTION IS EMPTY!")
            elif is_blank_only:
                print(f"   🚨 ALL PREDICTIONS ARE BLANK TOKEN!")
            elif len(unique_tokens) <= 3:
                print(f"   ⚠️  VERY FEW UNIQUE TOKENS ({len(unique_tokens)})")
            else:
                print(f"   ✅ Prediction has {len(unique_tokens)} unique tokens")
            
            if output_lengths[0] < len(text_tokens):
                print(f"   🚨 OUTPUT_LENGTH < TEXT_LENGTH! ({output_lengths[0].item()} < {len(text_tokens)})")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 80)
    print(f"Tổng số samples: {num_samples}")
    print(f"Empty outputs: {empty_outputs} ({empty_outputs/num_samples*100:.1f}%)")
    print(f"Blank-only outputs: {blank_only_outputs} ({blank_only_outputs/num_samples*100:.1f}%)")
    print(f"Valid outputs: {valid_outputs} ({valid_outputs/num_samples*100:.1f}%)")
    print()
    
    # Diagnosis
    if empty_outputs / num_samples > 0.5:
        print("🚨 CẢNH BÁO: Hơn 50% outputs là empty!")
        print("   Nguyên nhân có thể:")
        print("   1. Model đã collapse (output blank)")
        print("   2. Learning rate quá thấp (stuck in local minima)")
        print("   3. CTC alignment issue (output_lengths < text_lengths)")
        print()
        print("   Giải pháp:")
        print("   1. Tăng learning rate")
        print("   2. Kiểm tra output_lengths >= text_lengths")
        print("   3. Kiểm tra tokenizer encoding/decoding")
        return False
    elif blank_only_outputs / num_samples > 0.5:
        print("🚨 CẢNH BÁO: Hơn 50% outputs là blank token!")
        print("   Model đã bị collapse.")
        print("   Giải pháp: Tăng learning rate hoặc kiểm tra loss function")
        return False
    else:
        print("✅ Model có output đa dạng (không phải toàn blank/empty)")
        return True


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
    parser = argparse.ArgumentParser(description='Check inference output for debugging WER=1.0')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to check (default: 10)')
    
    args = parser.parse_args()
    
    success = check_inference_output(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples
    )
    
    sys.exit(0 if success else 1)

