#!/usr/bin/env python3
"""
Test model on full test dataset and calculate WER/CER metrics.
Usage: python3 scripts/test_full_dataset.py [--checkpoint PATH] [--batch_size N]
"""
import sys
import torch
import json
import csv
import argparse
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.model import CRNNCTC
from src.asr.dataset import ManifestDataset, collate_batch
from src.asr.decode import greedy_decode
from torch.utils.data import DataLoader

def load_model(model_path, vocab_path, device):
    """Load model and vocabulary."""
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    itos = {i: c for i, c in enumerate(vocab)}
    print(f"Vocabulary size: {len(vocab)}")
    
    print(f"Loading model from {model_path}...")
    model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return None, None, None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint format
    epoch = None
    loss = None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        print(f"Loaded checkpoint from epoch {epoch}")
        if loss is not None:
            print(f"Checkpoint loss: {loss:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    return model, itos, epoch, loss

def test_full_dataset(model, itos, test_manifest, audio_root, device, batch_size=16):
    """Test on full test dataset."""
    print(f"{'='*80}")
    print(f"Testing on FULL test dataset")
    print(f"{'='*80}\n")
    
    # Create dataset
    print(f"Loading test dataset from {test_manifest}...")
    dataset = ManifestDataset(
        manifest_path=test_manifest,
        vocab_path="data/processed/vocab.json",
        audio_root=audio_root
    )
    
    total_samples = len(dataset)
    print(f"Total test samples: {total_samples}\n")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_batch, 
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Collect all results and timing
    all_predictions = []
    all_ground_truths = []
    all_results = []
    total_inference_time = 0.0
    total_audio_duration = 0.0
    hop_length = 160  # must stay in sync with wav_to_logmelspec
    sample_rate = 16000
    
    print("Processing batches...")
    with torch.no_grad():
        for batch_idx, (X, Xlen, Y, Ylen) in enumerate(tqdm(dataloader, desc="Testing")):
            X, Xlen = X.to(device), Xlen.to(device)
            
            # Estimate total audio duration for this batch (seconds)
            # Each frame corresponds to hop_length samples.
            batch_duration = (Xlen.float().sum().item() * hop_length) / sample_rate
            total_audio_duration += batch_duration

            start_time = time.perf_counter()
            logits, out_lens = model(X, Xlen)
            predictions = greedy_decode(logits.cpu(), itos)
            total_inference_time += time.perf_counter() - start_time
            
            # Decode ground truth
            for y, ylen, pred in zip(Y, Ylen, predictions):
                gt = "".join([itos.get(int(idx), "") for idx in y[:ylen] if int(idx) != 0])
                
                # Remove language tags if present
                for tag in ['<|vi|>', '<|en|>', '<|ei|>']:
                    gt = gt.replace(tag, '').strip()
                    pred = pred.replace(tag, '').strip()
                
                all_ground_truths.append(gt)
                all_predictions.append(pred)
                all_results.append({
                    'ground_truth': gt,
                    'prediction': pred
                })
    
    return all_results, all_ground_truths, all_predictions, total_inference_time, total_audio_duration

def calculate_metrics(ground_truths, predictions, total_inference_time=None, total_audio_duration=None):
    """Calculate WER, CER and other metrics."""
    try:
        from jiwer import wer, cer
        
        # Calculate WER and CER
        word_error_rate = wer(ground_truths, predictions)
        char_error_rate = cer(ground_truths, predictions)
        
        # Calculate exact match accuracy
        exact_matches = sum(1 for gt, pred in zip(ground_truths, predictions) 
                          if gt.lower().replace(' ', '') == pred.lower().replace(' ', ''))
        exact_accuracy = exact_matches / len(ground_truths) * 100 if ground_truths else 0
        
        # Calculate word-level accuracy (case-insensitive)
        word_matches = sum(1 for gt, pred in zip(ground_truths, predictions) 
                          if gt.lower() == pred.lower())
        word_accuracy = word_matches / len(ground_truths) * 100 if ground_truths else 0

        # Sentence Error Rate (sentence counts as incorrect if any mismatch)
        sentence_errors = sum(1 for gt, pred in zip(ground_truths, predictions)
                              if gt.strip().lower() != pred.strip().lower())
        sentence_error_rate = sentence_errors / len(ground_truths) * 100 if ground_truths else 0

        # Real-Time Factor (RTF)
        rtf = None
        if total_inference_time is not None and total_audio_duration and total_audio_duration > 0:
            rtf = total_inference_time / total_audio_duration
        
        return {
            'wer': word_error_rate,
            'cer': char_error_rate,
            'exact_accuracy': exact_accuracy,
            'exact_matches': exact_matches,
            'word_accuracy': word_accuracy,
            'word_matches': word_matches,
            'total_samples': len(ground_truths),
            'sentence_error_rate': sentence_error_rate,
            'sentence_errors': sentence_errors,
            'rtf': rtf,
            'total_audio_seconds': total_audio_duration,
            'total_inference_seconds': total_inference_time
        }
    except ImportError:
        print("Error: jiwer not installed. Install with: pip install jiwer")
        return None

def print_results(metrics, epoch=None):
    """Print test results."""
    print(f"\n{'='*80}")
    print("TEST RESULTS - FULL DATASET")
    print(f"{'='*80}\n")
    
    if epoch is not None:
        print(f"Model Epoch: {epoch}")
    
    print(f"Total samples tested: {metrics['total_samples']:,}\n")
    
    print("Metrics:")
    print(f"  Word Error Rate (WER):     {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"  Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"  Exact Match Accuracy:      {metrics['exact_accuracy']:.2f}% ({metrics['exact_matches']}/{metrics['total_samples']})")
    print(f"  Word-level Accuracy:        {metrics['word_accuracy']:.2f}% ({metrics['word_matches']}/{metrics['total_samples']})")
    print(f"  Sentence Error Rate (SER):  {metrics['sentence_error_rate']:.2f}% ({metrics['sentence_errors']}/{metrics['total_samples']})")
    if metrics.get('rtf') is not None:
        print(f"  Real-Time Factor (RTF):     {metrics['rtf']:.4f}")
        print(f"    Total audio seconds:      {metrics['total_audio_seconds']:.2f}")
        print(f"    Total inference seconds:  {metrics['total_inference_seconds']:.2f}")
    
    print(f"\n{'='*80}\n")

def save_all_predictions(results, output_file):
    """Save all predictions to JSON file."""
    import json
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll predictions saved to: {output_path}")
    print(f"Total predictions: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Test ASR model on full test dataset")
    parser.add_argument("--checkpoint", default="data/results/asr_ctc.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", default="data/processed/vocab.json",
                       help="Path to vocabulary file")
    parser.add_argument("--test_manifest", default="data/processed/test/manifest.csv",
                       help="Path to test manifest")
    parser.add_argument("--audio_root", default="data/processed/test",
                       help="Root directory for audio files")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for testing")
    parser.add_argument("--output", default="experiments/reports/all_predictions.json",
                       help="Output file to save all predictions")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    model, itos, epoch, loss = load_model(args.checkpoint, args.vocab, device)
    if model is None:
        sys.exit(1)
    
    # Test on full dataset
    results, ground_truths, predictions, total_inference_time, total_audio_duration = test_full_dataset(
        model, itos, args.test_manifest, args.audio_root, device, args.batch_size
    )
    
    # Save all predictions
    save_all_predictions(results, args.output)
    
    # Calculate metrics
    metrics = calculate_metrics(
        ground_truths, predictions,
        total_inference_time=total_inference_time,
        total_audio_duration=total_audio_duration
    )
    if metrics:
        print_results(metrics, epoch)
        
        # Save metrics to file
        metrics_file = Path(args.output).parent / "metrics.json"
        metrics_to_save = {
            'checkpoint': args.checkpoint,
            'epoch': epoch,
            'loss': float(loss) if loss is not None else None,
            'metrics': {
                'wer': float(metrics['wer']),
                'cer': float(metrics['cer']),
                'exact_accuracy': float(metrics['exact_accuracy']),
                'exact_matches': int(metrics['exact_matches']),
                'word_accuracy': float(metrics['word_accuracy']),
                'word_matches': int(metrics['word_matches']),
                'total_samples': int(metrics['total_samples']),
                'sentence_error_rate': float(metrics['sentence_error_rate']),
                'sentence_errors': int(metrics['sentence_errors']),
                'rtf': float(metrics['rtf']) if metrics.get('rtf') is not None else None,
                'total_audio_seconds': float(metrics['total_audio_seconds']) if metrics.get('total_audio_seconds') is not None else None,
                'total_inference_seconds': float(metrics['total_inference_seconds']) if metrics.get('total_inference_seconds') is not None else None
            }
        }
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {metrics_file}")
    else:
        print("Could not calculate metrics. Install jiwer: pip install jiwer")
    
    # Show some sample results
    print("\nSample Results (first 10):")
    print("-" * 80)
    for i, r in enumerate(results[:10], 1):
        print(f"\n[{i}]")
        print(f"  GT: {r['ground_truth']}")
        print(f"  PR: {r['prediction']}")
        match = "✓" if r['ground_truth'].lower() == r['prediction'].lower() else "✗"
        print(f"  {match}")

if __name__ == "__main__":
    main()

