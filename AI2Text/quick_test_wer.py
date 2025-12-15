"""
Quick test to calculate WER/CER on a few samples from validation set.
"""

import torch
import yaml
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
from utils.metrics import calculate_wer, calculate_cer
from torch.cuda.amp import autocast


def quick_test(checkpoint_path: str, config_path: str = 'configs/default.yaml', num_samples: int = 10):
    """Quick test on a few validation samples."""
    print("="*80)
    print("🔍 QUICK WER/CER TEST")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')  # Use CPU to not interfere with training
    print(f"📱 Device: {device}")
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', 'N/A')
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    print(f"   Epoch: {epoch}, Best Val Loss: {best_val_loss:.4f}")
    
    # Setup components
    tokenizer_path = config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    vocab_size = config.get('vocab_size', 3500)
    
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    # Load model
    print("🤖 Loading model...")
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=vocab_size,
        d_model=config.get('d_model', 1024),
        num_encoder_layers=config.get('num_encoder_layers', 24),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_heads=config.get('num_heads', 16),
        d_ff=config.get('d_ff', 4096),
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Special tokens
    sos_token_id = getattr(tokenizer, 'sos_token_id', 2)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 3)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    # Load validation manifest
    print("📊 Loading validation samples...")
    dataset_root = config.get('dataset_root', 'data/processed/full_merged_dataset')
    manifest_path = Path(dataset_root) / 'val' / 'manifest.csv'
    val_df = pd.read_csv(manifest_path)
    print(f"   Total samples: {len(val_df):,}")
    
    # Test on first N samples
    test_df = val_df.head(num_samples)
    print(f"📈 Testing on {len(test_df)} samples...")
    print("-"*80)
    
    all_predictions = []
    all_references = []
    
    use_amp = config.get('use_amp', True)
    use_bf16 = config.get('use_bf16', True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            try:
                # Load audio - construct full path
                audio_path = row['audio_path']
                if not audio_path.startswith('/'):
                    # Relative path - construct full path
                    dataset_root = config.get('dataset_root', 'data/processed/full_merged_dataset')
                    audio_path = Path(dataset_root) / 'val' / audio_path
                else:
                    audio_path = Path(audio_path)
                
                if not audio_path.exists():
                    continue
                
                audio, sr = audio_processor.load_audio(str(audio_path))
                audio = audio_processor.trim_silence(audio)
                mel_spec = audio_processor.extract_mel_spectrogram(audio)
                
                # Prepare input
                features = torch.from_numpy(mel_spec.T).unsqueeze(0).float().to(device)
                lengths = torch.tensor([features.size(1)]).to(device)
                # Extract language from transcript if available (e.g., <|vi|> or <|en|>)
                transcript = row['transcript']
                if transcript.startswith('<|vi|>'):
                    language_id = 0  # Vietnamese
                    transcript = transcript.replace('<|vi|>', '').strip()
                elif transcript.startswith('<|en|>'):
                    language_id = 1  # English
                    transcript = transcript.replace('<|en|>', '').strip()
                else:
                    language_id = 0  # Default to Vietnamese
                
                language_ids = torch.tensor([language_id]).to(device)
                
                # Generate
                generated_tokens = model.generate(
                    features,
                    lengths=lengths,
                    language_ids=language_ids,
                    max_len=128,
                    sos_token_id=sos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    temperature=1.0
                )
                
                # Decode
                gen_seq = generated_tokens[0].cpu().tolist()
                decoded_tokens = []
                for token in gen_seq:
                    if token == eos_token_id:
                        break
                    if token != sos_token_id and token != pad_token_id:
                        decoded_tokens.append(token)
                
                pred_text = tokenizer.decode(decoded_tokens)
                ref_text = transcript  # Use cleaned transcript
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
            except Exception as e:
                print(f"Error processing {row.get('audio_path', 'unknown')}: {e}")
                continue
    
    if len(all_predictions) == 0:
        print("❌ No samples processed successfully")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    wer = calculate_wer(all_references, all_predictions)
    cer = calculate_cer(all_references, all_predictions)
    
    print(f"\n✅ Test Results (on {len(all_predictions)} samples):")
    print(f"📈 Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"📈 Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print("\n📝 Sample predictions:")
    print("-"*80)
    for i in range(min(3, len(all_predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Reference: {all_references[i][:100]}...")
        print(f"  Prediction: {all_predictions[i][:100]}...")
    print("="*80)
    
    return {'wer': wer, 'cer': cer, 'num_samples': len(all_predictions)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    
    quick_test(args.checkpoint, args.config, args.num_samples)

