"""
Standalone script to evaluate checkpoint and calculate WER/CER.
Runs independently without affecting training.
"""

import torch
import yaml
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
from training.dataset import ASRDataset, collate_fn, create_data_loaders
from utils.manifest_loader import load_merged_dataset
from utils.metrics import calculate_wer, calculate_cer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def evaluate_checkpoint(checkpoint_path: str, config_path: str = 'configs/default.yaml', 
                       max_batches: int = None):
    """Evaluate checkpoint on validation set and calculate WER/CER.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file
        max_batches: Maximum number of batches to evaluate (None = all)
    """
    print("="*80)
    print("🔍 EVALUATING CHECKPOINT")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use GPU if available for faster evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    print(f"   Epoch: {epoch}")
    print(f"   Best Val Loss: {best_val_loss}")
    
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
    print("🤖 Loading model...")
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Special token IDs
    sos_token_id = getattr(tokenizer, 'sos_token_id', 2)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 3)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    # Load validation dataset
    print("📊 Loading validation dataset...")
    dataset_root = config.get('dataset_root', 'data/processed/full_merged_dataset')
    language_filter = config.get('language_filter', None)
    val_df = load_merged_dataset('val', dataset_root, language=language_filter)
    print(f"   Validation samples: {len(val_df):,}")
    
    # Create dataset and loader
    val_dataset = ASRDataset(
        data_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        apply_augmentation=False,
        cache_in_ram=False
    )
    
    # Use smaller batch size for evaluation to avoid OOM
    eval_batch_size = min(config.get('val_batch_size', 128), 32)  # Cap at 32 for evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 12),  # Use more workers for faster loading
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()  # Enable pin_memory when using GPU
    )
    
    # Determine number of batches
    num_batches = len(val_loader)
    if max_batches:
        num_batches = min(num_batches, max_batches)
    else:
        # Evaluate on full validation set for comprehensive results
        num_batches = num_batches
    
    print(f"📈 Evaluating on {num_batches:,} batches (~{num_batches * config.get('val_batch_size', 128)} samples)...")
    print("-"*80)
    
    # Evaluation
    all_predictions = []
    all_references = []
    total_loss = 0.0
    num_samples = 0
    
    use_amp = config.get('use_amp', True)
    use_bf16 = config.get('use_bf16', True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", total=num_batches)):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move to device
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            language_ids = batch['language_ids'].to(device)
            transcripts = batch['transcripts']
            
            # Prepare target tokens
            target_tokens = text_tokens.clone()
            target_tokens[:, 0] = sos_token_id  # Replace first token with SOS
            
            # Forward pass
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits, _ = model(
                    x=audio_features,
                    tgt_tokens=target_tokens[:, :-1],
                    lengths=audio_lengths,
                    language_ids=language_ids
                )
            
            # Generate predictions using autoregressive generation
            generated_tokens = model.generate(
                audio_features,
                lengths=audio_lengths,
                language_ids=language_ids,
                max_len=config.get('val_max_len', 128),
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=1.0
            )
            
            # Decode predictions
            for i in range(generated_tokens.size(0)):
                gen_seq = generated_tokens[i].cpu().tolist()
                decoded_tokens = []
                for token in gen_seq:
                    if token == eos_token_id:
                        break
                    if token != sos_token_id and token != pad_token_id:
                        decoded_tokens.append(token)
                
                pred_text = tokenizer.decode(decoded_tokens)
                all_predictions.append(pred_text)
                all_references.append(transcripts[i])
                num_samples += 1
                
                # Debug: Print first few predictions
                if num_samples <= 5:
                    print("-" * 80)
                    print(f"🔍 Sample #{num_samples}")
                    print(f"🎯 Reference: {transcripts[i]}")
                    print(f"🤖 Prediction: {pred_text}")
                    
                    # Phân tích kết quả
                    if pred_text == "":
                        print("🚨 KỊCH BẢN XẤU 1: Prediction rỗng!")
                        print("   → Kiểm tra learning rate/tokenizer!")
                    elif len(pred_text) < len(transcripts[i]) * 0.3:
                        print("⚠️  Prediction quá ngắn (có thể do sample rate)")
                    elif len(pred_text) > len(transcripts[i]) * 3:
                        print("⚠️  Prediction quá dài (có thể do sample rate)")
                    elif any(c * 2 in pred_text for c in ['x', 'a', 'i', 'e', 'o', 'u']):
                        print("✅ KỊCH BẢN TỐT: Model đang học (lặp ký tự - bình thường)")
                    print("-" * 80)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("📊 CALCULATING METRICS")
    print("="*80)
    
    wer = calculate_wer(all_references, all_predictions)
    cer = calculate_cer(all_references, all_predictions)
    
    print(f"\n✅ EVALUATION RESULTS")
    print("-"*80)
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Samples evaluated: {num_samples:,}")
    print(f"📈 Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"📈 Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print("="*80)
    
    return {
        'wer': wer,
        'cer': cer,
        'num_samples': num_samples,
        'checkpoint': checkpoint_path,
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate checkpoint and calculate WER/CER')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (None = all)')
    
    args = parser.parse_args()
    
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        max_batches=args.max_batches
    )

