"""
Quick test to show validation output format with language embedding.
Runs only a few batches to demonstrate the output.
"""

import torch
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.asr_with_timestamps import ASRModelWithTimestamps
from preprocessing.audio_processing import AudioProcessor
from preprocessing.bpe_tokenizer import BPETokenizer
from training.dataset import ASRDataset, create_data_loaders
from utils.metrics import calculate_wer, calculate_cer
from utils.manifest_loader import load_merged_dataset
import pandas as pd

def quick_test():
    """Quick test showing validation output format."""
    
    print("=" * 80)
    print("⚡ QUICK VALIDATION TEST - Language Embedding")
    print("=" * 80)
    print()
    
    # Load config
    with open('configs/test_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    tokenizer = BPETokenizer()
    tokenizer.load(config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json'))
    print(f"✅ Tokenizer: {len(tokenizer)} tokens")
    
    # Load small subset of validation data using proper manifest loader
    dataset_root = Path(config.get('dataset_root', 'data/processed/merged_dataset'))
    val_manifest = dataset_root / 'val' / 'manifest.csv'
    
    if not val_manifest.exists():
        print(f"❌ Validation manifest not found: {val_manifest}")
        return
    
    # Use manifest_loader to properly handle paths
    from utils.manifest_loader import load_manifest_data
    df = load_manifest_data(str(val_manifest), base_audio_dir=str(dataset_root / 'val'))
    
    # Limit to 20 samples for quick test
    df = df.head(20)
    
    print(f"✅ Loaded {len(df)} validation samples (quick test)")
    
    # Create dataset
    dataset = ASRDataset(
        data_df=df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        apply_augmentation=False,
        cache_in_ram=False
    )
    
    # Create small model for test
    print("\n📦 Creating small test model...")
    model = ASRModelWithTimestamps(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1),
        predict_timestamps=False
    )
    model = model.to(device)
    model.eval()
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Language embedding: {sum(p.numel() for p in model.encoder.language_embedding.parameters())} parameters")
    
    # Test a few batches - use proper collate function
    from torch.utils.data import DataLoader
    from training.dataset import collate_fn
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print("\n" + "=" * 80)
    print("📊 VALIDATION OUTPUT FORMAT (First 10 samples)")
    print("=" * 80)
    
    all_predictions = []
    all_references = []
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if count >= 10:
                break
                
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_tokens = batch['text_tokens']
            transcripts = batch['transcripts']
            language_ids = batch.get('language_ids', None)
            
            if language_ids is not None:
                language_ids = language_ids.to(device)
                print(f"\n[Batch {batch_idx+1}] Language IDs: {language_ids.cpu().tolist()}")
            
            # Forward pass
            logits, output_lengths, _ = model(
                audio_features, audio_lengths, 
                return_timestamps=False,
                language_ids=language_ids
            )
            
            # Decode
            for i in range(len(transcripts)):
                if count >= 10:
                    break
                    
                # CTC decode
                logits_i = logits[i, :output_lengths[i]]
                pred_tokens = torch.argmax(logits_i, dim=-1).cpu().tolist()
                
                # Collapse CTC
                collapsed = []
                prev = None
                for token in pred_tokens:
                    if token != prev and token != tokenizer.blank_token_id:
                        collapsed.append(token)
                    prev = token
                
                pred_text = tokenizer.decode(collapsed)
                ref_text = transcripts[i]
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                # Calculate WER/CER
                wer = calculate_wer([ref_text], [pred_text])
                cer = calculate_cer([ref_text], [pred_text])
                match = "✅" if ref_text.strip().lower() == pred_text.strip().lower() else "❌"
                
                lang_info = ""
                if count < len(df):
                    lang_info = f"  Language: {df.iloc[count].get('language', 'N/A')}" if 'language' in df.columns else ""
                
                print(f"\n[{count+1}/10] {match}")
                if lang_info:
                    print(lang_info)
                print(f"  Ground Truth: {ref_text}")
                print(f"  Prediction:   {pred_text}")
                print(f"  WER: {wer:.4f} | CER: {cer:.4f}")
                
                count += 1
    
    # Overall metrics
    if all_references:
        overall_wer = calculate_wer(all_references, all_predictions)
        overall_cer = calculate_cer(all_references, all_predictions)
        print("\n" + "=" * 80)
        print(f"📊 OVERALL METRICS (First 10 samples)")
        print("=" * 80)
        print(f"WER: {overall_wer:.4f}")
        print(f"CER: {overall_cer:.4f}")
        print("=" * 80)
    
    print("\n✅ Quick test complete!")
    print("💡 This is the format you'll see after each epoch during training.")

if __name__ == '__main__':
    quick_test()

