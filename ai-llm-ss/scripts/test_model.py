#!/usr/bin/env python3
"""
Test model performance on test dataset.
Usage: python scripts/test_model.py [--num_samples N] [--checkpoint PATH]
"""
import sys
import torch
import torchaudio
import json
import csv
import argparse
from pathlib import Path
from src.asr.model import CRNNCTC
from src.asr.features import wav_to_logmelspec, ensure_mono16k
from src.asr.decode import greedy_decode
from src.asr.dataset import ManifestDataset, collate_batch
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
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'loss' in checkpoint:
            print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    return model, itos

def transcribe_audio(model, itos, audio_path, device):
    """Transcribe a single audio file."""
    try:
        # Load audio with fallback
        try:
            wav, sr = torchaudio.load(audio_path)
        except Exception:
            # Fallback to soundfile
            try:
                import soundfile as sf
                wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
                wav = torch.from_numpy(wav).transpose(0, 1)  # to (channels, time)
            except ImportError:
                raise RuntimeError("Need soundfile for audio loading. Install: pip install soundfile")
        
        wav, sr = ensure_mono16k(wav, sr)
        
        # Extract features
        feats = wav_to_logmelspec(wav, sr)  # (T, 80)
        feats = feats.unsqueeze(0).to(device)  # (1, T, 80)
        
        # Transcribe
        with torch.no_grad():
            logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
            text = greedy_decode(logits.cpu(), itos)[0]
        
        return text, wav.shape[-1] / sr  # Return text and duration
    except Exception as e:
        return f"<ERROR: {str(e)}>", 0.0

def test_single_files(model, itos, test_manifest, audio_root, num_samples, device):
    """Test on individual files from manifest."""
    print(f"\n{'='*80}")
    print(f"Testing on {num_samples} samples from test set")
    print(f"{'='*80}\n")
    
    # Read manifest
    samples = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            samples.append(row)
    
    results = []
    for i, sample in enumerate(samples, 1):
        audio_path = Path(audio_root) / sample['audio_path']
        ground_truth = sample['transcript'].strip()
        
        # Remove language tags if present
        if ground_truth.startswith('<|vi|>'):
            ground_truth = ground_truth.replace('<|vi|>', '').strip()
        elif ground_truth.startswith('<|en|>'):
            ground_truth = ground_truth.replace('<|en|>', '').strip()
        
        print(f"\n[{i}/{len(samples)}] {audio_path.name}")
        print(f"  Duration: {sample.get('duration', 'N/A')}s")
        print(f"  Ground Truth: {ground_truth}")
        
        # Transcribe
        prediction, duration = transcribe_audio(model, itos, str(audio_path), device)
        print(f"  Prediction:   {prediction}")
        print(f"  Audio length: {duration:.2f}s")
        
        # Simple comparison
        gt_clean = ground_truth.lower().replace(' ', '')
        pred_clean = prediction.lower().replace(' ', '')
        match = "✓" if gt_clean == pred_clean else "✗"
        print(f"  Match: {match}")
        
        results.append({
            'file': audio_path.name,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'duration': duration
        })
    
    return results

def test_batch(model, itos, test_manifest, audio_root, num_samples, device):
    """Test using DataLoader for batch processing."""
    print(f"\n{'='*80}")
    print(f"Batch testing on {num_samples} samples")
    print(f"{'='*80}\n")
    
    # Create dataset
    dataset = ManifestDataset(
        manifest_path=test_manifest,
        vocab_path="data/processed/vocab.json",
        audio_root=audio_root
    )
    
    # Limit dataset size
    if num_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_batch, num_workers=2)
    
    results = []
    total_samples = 0
    
    for batch_idx, (X, Xlen, Y, Ylen) in enumerate(dataloader):
        X, Xlen = X.to(device), Xlen.to(device)
        
        # Transcribe
        with torch.no_grad():
            logits, out_lens = model(X, Xlen)
            predictions = greedy_decode(logits.cpu(), itos)
        
        # Decode ground truth
        for y, ylen, pred in zip(Y, Ylen, predictions):
            gt = "".join([itos.get(int(idx), "") for idx in y[:ylen] if int(idx) != 0])
            results.append({
                'ground_truth': gt,
                'prediction': pred
            })
            total_samples += 1
            
            if total_samples >= num_samples:
                break
        
        if total_samples >= num_samples:
            break
    
    # Print first few results
    print("\nFirst 10 results:")
    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}]")
        print(f"  GT: {result['ground_truth']}")
        print(f"  PR: {result['prediction']}")
        match = "✓" if result['ground_truth'].lower() == result['prediction'].lower() else "✗"
        print(f"  {match}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test ASR model")
    parser.add_argument("--checkpoint", default="data/results/asr_ctc.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", default="data/processed/vocab.json",
                       help="Path to vocabulary file")
    parser.add_argument("--test_manifest", default="data/processed/full_merged_dataset/test/manifest.csv",
                       help="Path to test manifest")
    parser.add_argument("--audio_root", default="data/processed/full_merged_dataset/test",
                       help="Root directory for audio files")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to test")
    parser.add_argument("--batch", action="store_true",
                       help="Use batch processing")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, itos = load_model(args.checkpoint, args.vocab, device)
    if model is None:
        sys.exit(1)
    
    # Test
    if args.batch:
        results = test_batch(model, itos, args.test_manifest, args.audio_root, args.num_samples, device)
    else:
        results = test_single_files(model, itos, args.test_manifest, args.audio_root, args.num_samples, device)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples tested: {len(results)}")
    
    # Calculate accuracy (exact match)
    exact_matches = sum(1 for r in results 
                       if r['ground_truth'].lower().replace(' ', '') == r['prediction'].lower().replace(' ', ''))
    accuracy = exact_matches / len(results) * 100 if results else 0
    print(f"Exact match accuracy: {accuracy:.2f}% ({exact_matches}/{len(results)})")
    
    # Calculate WER and CER if jiwer is available
    try:
        from jiwer import wer, cer
        
        # Clean predictions and ground truth (remove language tags)
        clean_gts = []
        clean_preds = []
        for r in results:
            gt = r['ground_truth']
            pred = r['prediction']
            
            # Remove language tags
            for tag in ['<|vi|>', '<|en|>', '<|ei|>']:
                gt = gt.replace(tag, '').strip()
                pred = pred.replace(tag, '').strip()
            
            clean_gts.append(gt)
            clean_preds.append(pred)
        
        word_error_rate = wer(clean_gts, clean_preds)
        char_error_rate = cer(clean_gts, clean_preds)
        
        print(f"\nWord Error Rate (WER): {word_error_rate:.4f} ({word_error_rate*100:.2f}%)")
        print(f"Character Error Rate (CER): {char_error_rate:.4f} ({char_error_rate*100:.2f}%)")
    except ImportError:
        print("\n(Install 'jiwer' for WER/CER metrics: pip install jiwer)")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

