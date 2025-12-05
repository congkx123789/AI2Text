#!/usr/bin/env python3
"""
Quick script to check WER on validation set using merged_dataset.
"""

import torch
import yaml
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from models.asr_with_timestamps import ASRModelWithTimestamps
from preprocessing.audio_processing import AudioProcessor
from preprocessing.bpe_tokenizer import BPETokenizer
from utils.manifest_loader import load_merged_dataset
from training.dataset import ASRDataset, collate_fn
from torch.utils.data import DataLoader
from utils.metrics import calculate_wer, calculate_cer

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    checkpoints.extend(list(checkpoint_dir.rglob("*.pt")))
    checkpoints.extend(list(checkpoint_dir.rglob("*.pth")))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]

def evaluate_wer(config_path="configs/default.yaml", checkpoint_path=None, split="val"):
    """Evaluate WER on validation set."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoint found!")
            return
        print(f"📂 Using checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Use CPU for evaluation to avoid GPU memory conflicts
    device = torch.device('cpu')
    print(f"🖥️  Device: {device} (using CPU to avoid GPU conflicts)")
    
    # Load tokenizer
    tokenizer_type = config.get('tokenizer_type', 'bpe')
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer()
        tokenizer.load(config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json'))
        print(f"✅ Using BPE tokenizer ({len(tokenizer)} tokens)")
    else:
        from preprocessing.text_cleaning import Tokenizer
        tokenizer = Tokenizer()
        print(f"✅ Using character tokenizer ({len(tokenizer)} tokens)")
    
    # Load model
    model = ASRModelWithTimestamps(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 1024),
        num_encoder_layers=config.get('num_encoder_layers', 24),
        num_heads=config.get('num_heads', 16),
        d_ff=config.get('d_ff', 4096),
        dropout=0.0,
        predict_timestamps=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check vocab size mismatch and adjust if needed
    checkpoint_state = checkpoint['model_state_dict']
    if 'decoder.linear.weight' in checkpoint_state:
        checkpoint_vocab_size = checkpoint_state['decoder.linear.weight'].shape[0]
        current_vocab_size = len(tokenizer)
        
        if checkpoint_vocab_size != current_vocab_size:
            print(f"⚠️  Vocab size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}")
            print(f"   Using checkpoint vocab size: {checkpoint_vocab_size}")
            
            # Recreate model with checkpoint vocab size
            model = ASRModelWithTimestamps(
                input_dim=config.get('n_mels', 80),
                vocab_size=checkpoint_vocab_size,
                d_model=config.get('d_model', 1024),
                num_encoder_layers=config.get('num_encoder_layers', 24),
                num_heads=config.get('num_heads', 16),
                d_ff=config.get('d_ff', 4096),
                dropout=0.0,
                predict_timestamps=False
            )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('best_val_loss', 'N/A')
    wer = checkpoint.get('best_wer', 'N/A')
    print(f"📊 Model from epoch: {epoch}")
    if val_loss != 'N/A':
        print(f"   Best val_loss: {val_loss:.4f}")
    if wer != 'N/A':
        print(f"   Best WER: {wer:.4f}")
    
    # Load data
    dataset_root = config.get('dataset_root', 'data/processed/merged_dataset')
    print(f"📂 Loading {split} data from: {dataset_root}")
    val_df = load_merged_dataset(split, dataset_root)
    print(f"✅ Loaded {len(val_df)} samples")
    
    # Create dataset and loader
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    from preprocessing.text_cleaning import BilingualTextNormalizer
    normalizer = BilingualTextNormalizer()
    
    val_dataset = ASRDataset(
        data_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        apply_augmentation=False,
        cache_in_ram=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Evaluate
    print("\n🔄 Evaluating...")
    all_predictions = []
    all_references = []
    
    def ctc_decode(tokens, tokenizer):
        """CTC greedy decoding."""
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
            prev = token
        filtered = [t for t in collapsed if t != tokenizer.blank_token_id]
        return tokenizer.decode(filtered)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_tokens = batch['text_tokens']
            text_lengths = batch['text_lengths']
            transcripts = batch['transcripts']
            
            # Forward pass
            logits, output_lengths, _ = model(audio_features, audio_lengths, return_timestamps=False)
            
            # Decode predictions
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                pred_text = ctc_decode(pred_tokens, tokenizer)
                
                ref_text = transcripts[i]
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
    
    # Calculate metrics
    wer = calculate_wer(all_references, all_predictions)
    cer = calculate_cer(all_references, all_predictions)
    
    # Count empty predictions
    empty_preds = sum(1 for p in all_predictions if len(p.strip()) == 0)
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {split} ({len(all_predictions)} samples)")
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    print(f"Empty predictions: {empty_preds}/{len(all_predictions)} ({empty_preds/len(all_predictions)*100:.1f}%)")
    print("="*60)
    
    # Show some examples
    print("\n📝 Sample predictions:")
    for i in range(min(5, len(all_predictions))):
        ref = all_references[i][:60]
        pred = all_predictions[i][:60]
        print(f"\n  Reference: {ref}...")
        print(f"  Prediction: {pred}...")
    
    return wer, cer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    evaluate_wer(args.config, args.checkpoint, args.split)

