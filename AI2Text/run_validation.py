"""
Run validation only - loads checkpoint and validates on validation set.
Shows first 10 validation outputs.
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.asr_with_timestamps import ASRModelWithTimestamps
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import create_data_loaders
from utils.metrics import calculate_wer, calculate_cer
from utils.manifest_loader import load_merged_dataset
from tqdm import tqdm


def validate_model(config_path: str, checkpoint_path: str):
    """Run validation on the model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Setup preprocessing
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    # Setup tokenizer
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe_18k.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
        print(f"✅ Using BPE tokenizer: {bpe_path} ({len(tokenizer)} tokens)")
    else:
        tokenizer = Tokenizer()
        print(f"✅ Using character-level tokenizer ({len(tokenizer)} tokens)")
    
    # Setup model
    use_timestamps = config.get('use_timestamps', True)
    model = ASRModelWithTimestamps(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 1024),
        num_encoder_layers=config.get('num_encoder_layers', 24),
        num_heads=config.get('num_heads', 16),
        d_ff=config.get('d_ff', 4096),
        dropout=0.0,  # No dropout during validation
        predict_timestamps=use_timestamps
    )
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()
    
    # Filter out keys that don't exist
    filtered_state = {}
    for key, value in model_state.items():
        if key in current_model_state:
            if current_model_state[key].shape == value.shape:
                filtered_state[key] = value
    
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Setup loss
    criterion = torch.nn.CTCLoss(
        blank=tokenizer.blank_token_id,
        zero_infinity=True,
        reduction='mean'
    )
    
    # Load validation data
    dataset_root = config.get('dataset_root', 'data/processed/merged_dataset')
    print(f"\n📂 Loading validation data from: {dataset_root}")
    val_df = load_merged_dataset('val', dataset_root, language=None)
    print(f"   Validation samples: {len(val_df)}")
    
    # Create data loader
    _, val_loader = create_data_loaders(
        train_df=val_df.head(1),  # Dummy train df
        val_df=val_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        augmenter=None,  # No augmentation for validation
        persistent_workers=config.get('persistent_workers', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        sort_by_length=config.get('sort_by_length', True),
        use_bucketing=False,
        cache_in_ram=False
    )
    
    # Run validation
    print(f"\n🔍 Running validation...")
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_references = []
    
    def _ctc_decode(tokens: list) -> str:
        """Simple CTC greedy decoding."""
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
                prev = token
        filtered = [t for t in collapsed if t != tokenizer.blank_token_id]
        return tokenizer.decode(filtered)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            audio_features = batch['audio_features'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Get language IDs from batch
            language_ids = batch.get('language_ids', None)
            if language_ids is not None:
                language_ids = language_ids.to(device)
            
            # Forward pass
            logits, output_lengths, timestamps = model(
                audio_features, audio_lengths, return_timestamps=use_timestamps,
                language_ids=language_ids
            )
            
            # Calculate loss
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Val batch {num_batches}: Loss is NaN/Inf, skipping")
                continue
            
            total_loss += loss.item()
            
            # Decode predictions
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                ref_tokens = text_tokens[i, :text_lengths[i]].cpu().tolist()
                
                pred_text = _ctc_decode(pred_tokens)
                ref_text = tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
            
            num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    wer = calculate_wer(all_references, all_predictions)
    cer = calculate_cer(all_references, all_predictions)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION RESULTS")
    print("=" * 80)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(f"Total samples: {len(all_predictions)}")
    
    empty_preds = sum(1 for p in all_predictions if len(p.strip()) == 0)
    if empty_preds > 0:
        print(f"⚠️  Empty predictions: {empty_preds}/{len(all_predictions)}")
    
    # Print first 10 validation outputs
    print("\n" + "=" * 80)
    print("📊 FIRST 10 VALIDATION OUTPUTS")
    print("=" * 80)
    num_to_show = min(10, len(all_predictions))
    for i in range(num_to_show):
        ref = all_references[i]
        pred = all_predictions[i]
        # Calculate individual WER/CER for this sample
        sample_wer = calculate_wer([ref], [pred])
        sample_cer = calculate_cer([ref], [pred])
        
        match_indicator = "✅" if ref.strip().lower() == pred.strip().lower() else "❌"
        print(f"\n[{i+1}/{num_to_show}] {match_indicator}")
        print(f"  Ground Truth: {ref}")
        print(f"  Prediction:   {pred}")
        print(f"  WER: {sample_wer:.4f} | CER: {sample_cer:.4f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run validation only')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    
    validate_model(args.config, args.checkpoint)


if __name__ == '__main__':
    main()

