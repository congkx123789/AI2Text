#!/usr/bin/env python3
"""
Debug WER=1.0 ngay lập tức
Kiểm tra:
1. Output của model là gì (blank/empty/rubbish?)
2. Độ dài audio sau subsampling vs text length
3. CTC decode có hoạt động đúng không
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import numpy as np
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer
from models.asr_base import ASRModel

def debug_wer_immediate():
    """Debug WER=1.0 ngay lập tức"""
    
    print("="*80)
    print("DEBUG WER=1.0 - KIỂM TRA NGAY LẬP TỨC")
    print("="*80)
    
    # Load config
    config_path = 'configs/full_merged_dataset_test_1epoch.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load validation data (chỉ lấy 1 batch)
    val_df = db.get_split_data('val', config.get('split_version', 'v2_merged'))
    print(f"\nValidation samples: {len(val_df)}")
    print(f"Lấy 8 samples đầu tiên để test...\n")
    
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
    
    # Create data loader (chỉ 1 batch)
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    augmenter = AudioAugmenter()
    
    # Tạo dataset nhỏ
    small_val_df = val_df.head(8)
    
    train_loader, val_loader = create_data_loaders(
        train_df=small_val_df,
        val_df=small_val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=8,
        num_workers=1,  # Dùng 1 worker để tránh lỗi division by zero
        augmenter=augmenter,
        persistent_workers=False,
        prefetch_factor=2,
        sort_by_length=False,
        use_bucketing=False,
        cache_in_ram=False
    )
    
    # Load model trên CPU
    print(f"\nLoading model trên CPU...")
    device = torch.device('cpu')  # Dùng CPU để tránh OOM
    checkpoint_path = 'checkpoints/full_merged_test/best_model.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded. Running inference...\n")
    
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
    
    # Lấy 1 batch
    batch = next(iter(val_loader))
    
    audio_features = batch['audio_features'].to(device)
    audio_lengths = batch['audio_lengths'].to(device)
    text_tokens = batch['text_tokens'].to(device)
    text_lengths = batch['text_lengths'].to(device)
    
    print("="*80)
    print("INPUT CHECK")
    print("="*80)
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Audio lengths: {audio_lengths.tolist()}")
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Text lengths: {text_lengths.tolist()}")
    print()
    
    # Forward pass
    with torch.no_grad():
        logits, output_lengths = model(audio_features, audio_lengths)
    
    print("="*80)
    print("MODEL OUTPUT CHECK")
    print("="*80)
    print(f"Logits shape: {logits.shape}")  # (batch, time, vocab)
    print(f"Output lengths: {output_lengths.tolist()}")
    print()
    
    # Get predictions (greedy)
    predictions = torch.argmax(logits, dim=-1)
    
    print("="*80)
    print("PREDICTIONS vs REFERENCES")
    print("="*80)
    
    all_empty = 0
    all_blank = 0
    length_issues = 0
    
    for i in range(predictions.size(0)):
        pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
        ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
        
        # Decode
        pred_text = ctc_decode(pred_tokens, tokenizer.blank_token_id)
        ref_text = tokenizer.decode(ref_tokens)
        
        # Analyze
        unique_preds = set(pred_tokens)
        is_all_blank = len(unique_preds) == 1 and list(unique_preds)[0] == tokenizer.blank_token_id
        is_empty = len(pred_text.strip()) == 0
        output_len = output_lengths[i].item()
        text_len = text_lengths[i].item()
        has_length_issue = output_len < text_len
        
        if is_empty:
            all_empty += 1
        if is_all_blank:
            all_blank += 1
        if has_length_issue:
            length_issues += 1
        
        print(f"\nSample {i+1}:")
        print(f"  Reference: '{ref_text}'")
        print(f"  Prediction: '{pred_text}'")
        print(f"  Ref tokens (first 10): {ref_tokens[:10]}")
        print(f"  Pred tokens (first 20): {pred_tokens[:20]}")
        print(f"  Unique pred tokens: {unique_preds}")
        print(f"  Output length: {output_len}, Text length: {text_len}")
        print(f"  Audio length (original): {audio_lengths[i].item()}")
        print(f"  Subsampling ratio: {audio_lengths[i].item() / output_len:.2f}x")
        
        if is_empty:
            print(f"  🚨 PREDICTION IS EMPTY!")
        if is_all_blank:
            print(f"  🚨 ALL PREDICTIONS ARE BLANK TOKEN ({tokenizer.blank_token_id})!")
        if has_length_issue:
            print(f"  🚨 OUTPUT_LENGTH < TEXT_LENGTH! ({output_len} < {text_len})")
            print(f"     CTC sẽ fail hoặc output rỗng!")
        
        # Check logits distribution
        sample_logits = logits[i, :output_len, :]
        probs = torch.softmax(sample_logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_max_prob = max_probs.mean().item()
        blank_prob = probs[:, tokenizer.blank_token_id].mean().item()
        
        print(f"  Logits analysis:")
        print(f"    Avg max prob: {avg_max_prob:.4f}")
        print(f"    Avg blank prob: {blank_prob:.4f}")
        if blank_prob > 0.9:
            print(f"    ⚠️  Model đang output blank với xác suất rất cao!")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {predictions.size(0)}")
    print(f"Empty predictions: {all_empty}/{predictions.size(0)}")
    print(f"All-blank predictions: {all_blank}/{predictions.size(0)}")
    print(f"Length issues (output < text): {length_issues}/{predictions.size(0)}")
    
    if all_empty == predictions.size(0):
        print(f"\n🚨 TẤT CẢ PREDICTIONS ĐỀU RỖNG!")
        print(f"   Nguyên nhân có thể:")
        print(f"   1. Model output toàn blank token")
        print(f"   2. CTC decode bị lỗi")
        print(f"   3. output_lengths < text_lengths (CTC alignment fail)")
    elif all_blank == predictions.size(0):
        print(f"\n🚨 TẤT CẢ PREDICTIONS ĐỀU LÀ BLANK TOKEN!")
        print(f"   Model đã bị collapse - chỉ output blank để minimize loss")
    elif length_issues > 0:
        print(f"\n🚨 CÓ {length_issues} SAMPLES BỊ LỖI ĐỘ DÀI!")
        print(f"   Audio quá ngắn sau subsampling so với text dài")
        print(f"   Cần filter data hoặc tăng subsampling ratio")

if __name__ == '__main__':
    debug_wer_immediate()

