#!/usr/bin/env python3
"""
Script nhanh để debug prediction của model trên 1 vài samples.
Sử dụng khi đang training để kiểm tra model có đang học đúng không.
"""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import BilingualTextNormalizer, Tokenizer
from training.dataset import ASRDataset, create_data_loaders


def quick_debug(model, val_loader, tokenizer, num_samples=5, device='cuda'):
    """
    Debug nhanh một vài predictions từ validation loader.
    
    Args:
        model: Model ASR
        val_loader: Validation DataLoader
        tokenizer: Tokenizer
        num_samples: Số samples để debug
        device: Device
    """
    model.eval()
    
    print("=" * 80)
    print("🔍 QUICK DEBUG PREDICTIONS")
    print("=" * 80)
    print()
    
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if count >= num_samples:
                break
            
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            transcripts = batch['transcripts']
            language_ids = batch.get('language_ids', torch.zeros(len(transcripts), dtype=torch.long)).to(device)
            
            # Get SOS, EOS, PAD token IDs
            if hasattr(tokenizer, 'sos_token_id'):
                sos_token_id = tokenizer.sos_token_id
            elif hasattr(tokenizer, 'sp_tokenizer'):
                sos_token_id = tokenizer.sp_tokenizer.bos_id()
            else:
                sos_token_id = 1
            
            if hasattr(tokenizer, 'eos_token_id'):
                eos_token_id = tokenizer.eos_token_id
            elif hasattr(tokenizer, 'sp_tokenizer'):
                eos_token_id = tokenizer.sp_tokenizer.eos_id()
            else:
                eos_token_id = 2
            
            if hasattr(tokenizer, 'pad_token_id'):
                pad_token_id = tokenizer.pad_token_id
            elif hasattr(tokenizer, 'sp_tokenizer'):
                pad_token_id = tokenizer.sp_tokenizer.pad_id()
            else:
                pad_token_id = 0
            
            # Generate predictions
            try:
                generated_tokens = model.generate(
                    audio_features,
                    lengths=audio_lengths,
                    language_ids=language_ids,
                    max_len=128,
                    sos_token_id=sos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    temperature=1.0
                )
            except Exception as e:
                print(f"❌ Lỗi khi generate: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Decode predictions
            batch_size = generated_tokens.size(0)
            for i in range(batch_size):
                if count >= num_samples:
                    break
                
                gen_seq = generated_tokens[i].cpu().tolist()
                decoded_tokens = []
                for token in gen_seq:
                    if token == eos_token_id:
                        break
                    if token != sos_token_id and token != pad_token_id:
                        decoded_tokens.append(token)
                
                # Decode với tokenizer
                if hasattr(tokenizer, 'sp_tokenizer'):
                    pred_text = tokenizer.sp_tokenizer.decode(decoded_tokens)
                elif hasattr(tokenizer, 'decode'):
                    pred_text = tokenizer.decode(decoded_tokens)
                else:
                    pred_text = ''.join([chr(t) if t < 256 else '?' for t in decoded_tokens])
                ref_text = transcripts[i]
                
                # Print result
                print("-" * 80)
                print(f"Sample #{count + 1}")
                print(f"🎯 Reference: {ref_text}")
                print(f"🤖 Prediction: {pred_text}")
                
                # Phân tích
                if pred_text == "":
                    print("🚨 KỊCH BẢN XẤU 1: Prediction rỗng!")
                    print("   → Dừng training, kiểm tra learning rate/tokenizer!")
                elif len(pred_text) < len(ref_text) * 0.3:
                    print("⚠️  Prediction quá ngắn (có thể do sample rate)")
                elif len(pred_text) > len(ref_text) * 3:
                    print("⚠️  Prediction quá dài (có thể do sample rate)")
                elif any(c * 2 in pred_text for c in ['x', 'a', 'i', 'e', 'o', 'u']):
                    print("✅ KỊCH BẢN TỐT: Model đang học (lặp ký tự - bình thường với CTC)")
                else:
                    # So sánh từ
                    ref_words = set(ref_text.lower().split())
                    pred_words = set(pred_text.lower().split())
                    if ref_words and pred_words:
                        match = len(ref_words & pred_words) / len(ref_words)
                        if match > 0.5:
                            print(f"✅ Prediction có {match*100:.1f}% từ khớp")
                        else:
                            print(f"⚠️  Prediction chỉ có {match*100:.1f}% từ khớp")
                
                print("-" * 80)
                print()
                
                count += 1
    
    print("=" * 80)
    print("✅ Debug hoàn tất!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description='Quick debug predictions của model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Debug với checkpoint
  python scripts/quick_debug_prediction.py \\
    --checkpoint checkpoints/best_model.pt \\
    --config configs/default.yaml \\
    --num_samples 10
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Đường dẫn đến checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Đường dẫn đến config file'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Số samples để debug (default: 5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    print(f"📦 Đang load checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load model (cần import model architecture)
    from models.asr_base import ASRModel
    from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
    
    # Setup tokenizer
    tokenizer_path = config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    vocab_size = config.get('vocab_size', 3500)
    
    # Setup audio processor
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80),
        n_fft=config.get('n_fft', 400),
        hop_length=config.get('hop_length', 160),
        win_length=config.get('win_length', 400)
    )
    
    # Load model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=vocab_size,
        d_model=config.get('d_model', 1024),
        num_encoder_layers=config.get('num_encoder_layers', 24),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_heads=config.get('num_heads', 16),
        d_ff=config.get('d_ff', 4096),
        dropout=0.0  # No dropout during inference
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    # Load validation data
    val_manifest = config.get('val_manifest', 'data/processed/full_merged_dataset/val/manifest.csv')
    if not Path(val_manifest).exists():
        # Try alternative path
        val_manifest = 'data/processed/full_merged_dataset/val/manifest.csv'
    
    val_df = pd.read_csv(val_manifest)
    
    # Create validation loader
    from training.dataset import create_data_loaders
    from preprocessing.text_cleaning import BilingualTextNormalizer
    
    normalizer = BilingualTextNormalizer()
    
    # Create simple tokenizer wrapper for dataset
    class SimpleTokenizer:
        def __init__(self, sp_tokenizer):
            self.sp_tokenizer = sp_tokenizer
            self.sos_token_id = sp_tokenizer.bos_id()
            self.eos_token_id = sp_tokenizer.eos_id()
            self.pad_token_id = sp_tokenizer.pad_id()
        
        def encode(self, text):
            return self.sp_tokenizer.encode(text, out_type=int)
        
        def decode(self, ids):
            return self.sp_tokenizer.decode(ids)
    
    dataset_tokenizer = SimpleTokenizer(tokenizer)
    
    _, val_loader = create_data_loaders(
        train_df=val_df.head(100),  # Dummy train
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=dataset_tokenizer,
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        cache_in_ram=False
    )
    
    # Debug
    quick_debug(model, val_loader, tokenizer, num_samples=args.num_samples, device=args.device)

