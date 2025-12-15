#!/usr/bin/env python3
"""
Script để kiểm tra coverage của tokenizer trên toàn bộ dataset (train, val, test).
"""

import pandas as pd
import sentencepiece as spm
from pathlib import Path
import sys
import re
from collections import Counter
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.text_cleaning import BilingualTextNormalizer


def extract_text_from_transcript(transcript: str) -> Tuple[str, str]:
    """Extract text and language from transcript."""
    if '<|vi|>' in transcript:
        lang = 'vi'
        text = re.sub(r'<\|vi\|>', '', transcript)
    elif '<|en|>' in transcript:
        lang = 'en'
        text = re.sub(r'<\|en\|>', '', transcript)
    else:
        lang = 'vi'
        text = transcript
    return text.strip(), lang


def load_texts_from_manifest(manifest_path: str) -> Tuple[List[str], List[str]]:
    """Load texts and languages from manifest."""
    print(f"📖 Loading from {manifest_path}...")
    
    df = pd.read_csv(manifest_path)
    texts = []
    languages = []
    
    for transcript in df['transcript']:
        text, lang = extract_text_from_transcript(transcript)
        texts.append(text)
        languages.append(lang)
    
    print(f"   Loaded {len(texts):,} texts")
    return texts, languages


def check_coverage(tokenizer: spm.SentencePieceProcessor, texts: List[str], 
                   normalizer: BilingualTextNormalizer = None) -> dict:
    """Check tokenizer coverage on texts."""
    print(f"\n🔍 Checking coverage on {len(texts):,} texts...")
    
    total_chars = 0
    total_tokens = 0
    unk_count = 0
    byte_fallback_count = 0
    unique_chars = set()
    char_freq = Counter()
    
    unk_id = tokenizer.unk_id()
    
    for i, text in enumerate(texts):
        if normalizer:
            text = normalizer.normalize(text)
        
        # Count characters
        for char in text:
            unique_chars.add(char)
            char_freq[char] += 1
        total_chars += len(text)
        
        # Tokenize
        token_ids = tokenizer.encode(text, out_type=int, add_bos=False, add_eos=False)
        total_tokens += len(token_ids)
        
        # Check for UNK and byte fallback
        for tid in token_ids:
            if tid == unk_id:
                unk_count += 1
            # Check if it's a byte fallback token (<0x00> to <0xFF>)
            token_str = tokenizer.id_to_piece(tid)
            if token_str.startswith('<0x') and token_str.endswith('>'):
                byte_fallback_count += 1
        
        if (i + 1) % 10000 == 0:
            print(f"   Processed {i+1:,}/{len(texts):,} texts...", end='\r')
    
    print(f"\n✅ Coverage analysis completed")
    
    # Calculate statistics
    avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    unk_rate = unk_count / total_tokens if total_tokens > 0 else 0
    byte_fallback_rate = byte_fallback_count / total_tokens if total_tokens > 0 else 0
    
    return {
        'total_texts': len(texts),
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'unique_chars_count': len(unique_chars),
        'unique_chars_set': unique_chars,
        'unk_count': unk_count,
        'unk_rate': unk_rate,
        'byte_fallback_count': byte_fallback_count,
        'byte_fallback_rate': byte_fallback_rate,
        'avg_tokens_per_char': avg_tokens_per_char,
        'char_freq': char_freq
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check tokenizer coverage")
    parser.add_argument(
        '--model',
        type=str,
        default='models/tokenizer_vi_en_3500.model',
        help='Path to tokenizer model'
    )
    parser.add_argument(
        '--train-manifest',
        type=str,
        default='data/processed/full_merged_dataset/train/manifest.csv',
        help='Training manifest path'
    )
    parser.add_argument(
        '--val-manifest',
        type=str,
        default='data/processed/full_merged_dataset/val/manifest.csv',
        help='Validation manifest path'
    )
    parser.add_argument(
        '--test-manifest',
        type=str,
        default='data/processed/full_merged_dataset/test/manifest.csv',
        help='Test manifest path'
    )
    parser.add_argument(
        '--skip-normalize',
        action='store_true',
        help='Skip text normalization'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Tokenizer Coverage Check")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer from {args.model}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.model)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.get_piece_size()})")
    
    # Load normalizer
    normalizer = None if args.skip_normalize else BilingualTextNormalizer()
    
    # Check each dataset
    results = {}
    
    # Training set
    if Path(args.train_manifest).exists():
        train_texts, train_langs = load_texts_from_manifest(args.train_manifest)
        results['train'] = check_coverage(tokenizer, train_texts, normalizer)
        print(f"   Vietnamese: {train_langs.count('vi'):,} ({train_langs.count('vi')/len(train_langs)*100:.1f}%)")
        print(f"   English: {train_langs.count('en'):,} ({train_langs.count('en')/len(train_langs)*100:.1f}%)")
    
    # Validation set
    if Path(args.val_manifest).exists():
        val_texts, val_langs = load_texts_from_manifest(args.val_manifest)
        results['val'] = check_coverage(tokenizer, val_texts, normalizer)
        print(f"   Vietnamese: {val_langs.count('vi'):,} ({val_langs.count('vi')/len(val_langs)*100:.1f}%)")
        print(f"   English: {val_langs.count('en'):,} ({val_langs.count('en')/len(val_langs)*100:.1f}%)")
    
    # Test set
    if Path(args.test_manifest).exists():
        test_texts, test_langs = load_texts_from_manifest(args.test_manifest)
        results['test'] = check_coverage(tokenizer, test_texts, normalizer)
        print(f"   Vietnamese: {test_langs.count('vi'):,} ({test_langs.count('vi')/len(test_langs)*100:.1f}%)")
        print(f"   English: {test_langs.count('en'):,} ({test_langs.count('en')/len(test_langs)*100:.1f}%)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    
    for split_name, stats in results.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Texts: {stats['total_texts']:,}")
        print(f"  Total characters: {stats['total_chars']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Unique characters: {stats['unique_chars_count']:,}")
        print(f"  Avg tokens/char: {stats['avg_tokens_per_char']:.3f}")
        print(f"  UNK tokens: {stats['unk_count']:,} ({stats['unk_rate']*100:.4f}%)")
        print(f"  Byte fallback tokens: {stats['byte_fallback_count']:,} ({stats['byte_fallback_rate']*100:.4f}%)")
    
    # Overall statistics
    if len(results) > 1:
        print(f"\nOVERALL:")
        total_texts = sum(s['total_texts'] for s in results.values())
        total_chars = sum(s['total_chars'] for s in results.values())
        total_tokens = sum(s['total_tokens'] for s in results.values())
        total_unk = sum(s['unk_count'] for s in results.values())
        total_byte_fallback = sum(s['byte_fallback_count'] for s in results.values())
        all_unique_chars = set()
        for s in results.values():
            all_unique_chars.update(s['unique_chars_set'])
        
        print(f"  Total texts: {total_texts:,}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Unique characters: {len(all_unique_chars):,}")
        print(f"  UNK rate: {total_unk/total_tokens*100:.4f}%")
        print(f"  Byte fallback rate: {total_byte_fallback/total_tokens*100:.4f}%")
    
    print("\n" + "=" * 80)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    overall_unk_rate = sum(s['unk_count'] for s in results.values()) / sum(s['total_tokens'] for s in results.values()) if results else 0
    
    if overall_unk_rate > 0.01:  # > 1%
        print("  ⚠️  UNK rate is high (>1%). Consider:")
        print("     - Training on train+val sets together")
        print("     - Increasing vocabulary size")
    else:
        print("  ✅ UNK rate is low. Tokenizer coverage is good!")
    
    if 'val' in results or 'test' in results:
        print("  💡 For best coverage, consider training on train+val sets")
        print("     (test set should remain unseen for final evaluation)")


if __name__ == "__main__":
    main()

