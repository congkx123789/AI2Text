#!/usr/bin/env python3
"""
Script to diagnose why WER = 1.0
Checks:
1. Are predictions all blank/empty?
2. Is CTC decoding working correctly?
3. Are predictions vs references being compared correctly?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer
from models.asr_base import ASRModel
from utils.metrics import calculate_wer, calculate_cer

def check_wer_issue():
    """Check why WER = 1.0"""
    
    # Load config
    config_path = 'configs/full_merged_dataset_test_1epoch.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load validation data
    val_df = db.get_split_data('val', config.get('split_version', 'v2_merged'))
    print(f"Validation samples: {len(val_df)}")
    
    # Setup tokenizer
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
    print(f"Tokenizer blank_token_id: {tokenizer.blank_token_id}")
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    # Create data loader
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()
    
    train_loader, val_loader = create_data_loaders(
        train_df=val_df.head(100),  # Use small subset for testing
        val_df=val_df.head(100),
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=2,
        augmenter=augmenter,
        persistent_workers=False,
        prefetch_factor=2,
        sort_by_length=False,
        use_bucketing=False,
        cache_in_ram=False
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/full_merged_test/best_model.pt'
    
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = ASRModel(
        vocab_size=len(tokenizer.vocab),
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_decoder_layers=config.get('num_decoder_layers', 0),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config.get('max_seq_length', 2000)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded. Running inference on validation samples...\n")
    
    # CTC decode function
    def ctc_decode(tokens, blank_id):
        """Simple CTC greedy decoding"""
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
                prev = token
        filtered = [t for t in collapsed if t != blank_id]
        return tokenizer.decode(filtered)
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 3:  # Only check first 3 batches
                break
                
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Forward pass
            logits, output_lengths = model(audio_features, audio_lengths)
            
            # Get predictions (greedy)
            predictions = torch.argmax(logits, dim=-1)
            
            print(f"\n{'='*80}")
            print(f"Batch {batch_idx + 1}")
            print(f"{'='*80}")
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                # Decode
                pred_text = ctc_decode(pred_tokens, tokenizer.blank_token_id)
                ref_text = tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                # Analyze
                unique_preds = set(pred_tokens)
                is_all_blank = len(unique_preds) == 1 and list(unique_preds)[0] == tokenizer.blank_token_id
                is_empty = len(pred_text.strip()) == 0
                
                print(f"\nSample {i+1}:")
                print(f"  Reference: '{ref_text}'")
                print(f"  Prediction: '{pred_text}'")
                print(f"  Pred tokens (first 20): {pred_tokens[:20]}")
                print(f"  Unique pred tokens: {unique_preds}")
                print(f"  Output length: {output_lengths[i].item()}, Text length: {text_lengths[i].item()}")
                print(f"  Is all blank: {is_all_blank}")
                print(f"  Is empty: {is_empty}")
                
                if is_all_blank or is_empty:
                    print(f"  ⚠️  PROBLEM: Prediction is blank/empty!")
    
    # Calculate WER/CER
    wer = calculate_wer(all_references, all_predictions)
    cer = calculate_cer(all_references, all_predictions)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples checked: {len(all_predictions)}")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    
    empty_count = sum(1 for p in all_predictions if len(p.strip()) == 0)
    print(f"Empty predictions: {empty_count}/{len(all_predictions)} ({empty_count/len(all_predictions)*100:.1f}%)")
    
    # Show first few comparisons
    print(f"\nFirst 5 predictions vs references:")
    for i in range(min(5, len(all_predictions))):
        print(f"  [{i+1}] Ref: '{all_references[i]}'")
        print(f"      Pred: '{all_predictions[i]}'")
        print(f"      Match: {all_references[i] == all_predictions[i]}")

if __name__ == '__main__':
    check_wer_issue()

