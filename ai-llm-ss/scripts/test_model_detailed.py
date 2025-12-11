#!/usr/bin/env python3
"""
Detailed model testing with analysis.
"""
import sys
import torch
import json
import csv
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.model import CRNNCTC
from src.asr.features import wav_to_logmelspec, ensure_mono16k
from src.asr.decode import greedy_decode
import torchaudio

def load_model(model_path, vocab_path, device):
    """Load model and vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    itos = {i: c for i, c in enumerate(vocab)}
    
    model = CRNNCTC(n_mels=80, vocab_size=len(vocab))
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        loss = 'unknown'
    
    model.to(device).eval()
    return model, itos, epoch, loss

def transcribe_audio(model, itos, audio_path, device):
    """Transcribe audio with error handling."""
    try:
        try:
            wav, sr = torchaudio.load(audio_path)
        except Exception:
            import soundfile as sf
            wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav).transpose(0, 1)
        
        wav, sr = ensure_mono16k(wav, sr)
        feats = wav_to_logmelspec(wav, sr).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
            text = greedy_decode(logits.cpu(), itos)[0]
        
        return text, True
    except Exception as e:
        return f"<ERROR: {str(e)}>", False

def analyze_results(results):
    """Analyze test results."""
    from jiwer import wer, cer
    
    # Separate by language
    vi_results = []
    en_results = []
    
    for r in results:
        gt = r['ground_truth']
        pred = r['prediction']
        
        # Remove language tags
        for tag in ['<|vi|>', '<|en|>', '<|ei|>']:
            gt = gt.replace(tag, '').strip()
            pred = pred.replace(tag, '').strip()
        
        if '<|vi|>' in r['prediction'] or any(c in r['ground_truth'] for c in 'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'):
            vi_results.append({'gt': gt, 'pred': pred})
        else:
            en_results.append({'gt': gt, 'pred': pred})
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}\n")
    
    # Overall metrics
    all_gts = [r['gt'] for r in vi_results + en_results]
    all_preds = [r['pred'] for r in vi_results + en_results]
    
    overall_wer = wer(all_gts, all_preds)
    overall_cer = cer(all_gts, all_preds)
    
    print(f"Overall Metrics (all {len(all_gts)} samples):")
    print(f"  Word Error Rate (WER): {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"  Character Error Rate (CER): {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    
    # By language
    if vi_results:
        vi_gts = [r['gt'] for r in vi_results]
        vi_preds = [r['pred'] for r in vi_results]
        vi_wer = wer(vi_gts, vi_preds)
        vi_cer = cer(vi_gts, vi_preds)
        print(f"\nVietnamese ({len(vi_results)} samples):")
        print(f"  WER: {vi_wer:.4f} ({vi_wer*100:.2f}%)")
        print(f"  CER: {vi_cer:.4f} ({vi_cer*100:.2f}%)")
    
    if en_results:
        en_gts = [r['gt'] for r in en_results]
        en_preds = [r['pred'] for r in en_results]
        en_wer = wer(en_gts, en_preds)
        en_cer = cer(en_gts, en_preds)
        print(f"\nEnglish ({len(en_results)} samples):")
        print(f"  WER: {en_wer:.4f} ({en_wer*100:.2f}%)")
        print(f"  CER: {en_cer:.4f} ({en_cer*100:.2f}%)")
    
    # Best and worst examples
    print(f"\n{'='*80}")
    print("SAMPLE RESULTS")
    print(f"{'='*80}\n")
    
    # Show first 5 examples
    for i, r in enumerate(results[:5], 1):
        gt = r['ground_truth']
        pred = r['prediction']
        for tag in ['<|vi|>', '<|en|>', '<|ei|>']:
            gt = gt.replace(tag, '').strip()
            pred = pred.replace(tag, '').strip()
        
        print(f"[{i}] {r['file']}")
        print(f"  GT:  {gt}")
        print(f"  PR:  {pred}")
        print()

def main():
    model_path = "data/results/checkpoints/checkpoint_epoch_3.pt"
    vocab_path = "data/processed/vocab.json"
    test_manifest = "data/processed/full_merged_dataset/test/manifest.csv"
    audio_root = "data/processed/full_merged_dataset/test"
    num_samples = 20
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    model, itos, epoch, loss = load_model(model_path, vocab_path, device)
    print(f"Model loaded:")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Load test samples
    samples = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            samples.append(row)
    
    # Test
    results = []
    for sample in samples:
        audio_path = Path(audio_root) / sample['audio_path']
        gt = sample['transcript'].strip()
        
        pred, success = transcribe_audio(model, itos, str(audio_path), device)
        
        results.append({
            'file': audio_path.name,
            'ground_truth': gt,
            'prediction': pred,
            'success': success
        })
    
    # Analyze
    analyze_results(results)

if __name__ == "__main__":
    main()

