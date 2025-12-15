#!/usr/bin/env python3
"""
Script để train SentencePiece tokenizer với 3500 vocabulary cho seq2seq training.

Sử dụng dữ liệu từ training manifest để train tokenizer bilingual (Vietnamese + English).
"""

import pandas as pd
import sentencepiece as spm
from pathlib import Path
import sys
import re
from typing import List, Tuple
import multiprocessing

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.text_cleaning import BilingualTextNormalizer


def extract_text_from_transcript(transcript: str) -> Tuple[str, str]:
    """
    Extract text and language from transcript.
    
    Args:
        transcript: Transcript with language tags like <|vi|> or <|en|>
        
    Returns:
        Tuple of (text, language) where language is 'vi' or 'en'
    """
    # Detect language
    if '<|vi|>' in transcript:
        lang = 'vi'
        text = re.sub(r'<\|vi\|>', '', transcript)
    elif '<|en|>' in transcript:
        lang = 'en'
        text = re.sub(r'<\|en\|>', '', transcript)
    else:
        # Default to Vietnamese if no tag found
        lang = 'vi'
        text = transcript
    
    text = text.strip()
    return text, lang


def load_training_texts(manifest_path: str, max_samples: int = None) -> Tuple[List[str], List[str]]:
    """
    Load training texts from manifest CSV.
    
    Args:
        manifest_path: Path to manifest CSV file
        max_samples: Maximum number of samples to load (None = all)
        
    Returns:
        Tuple of (texts, languages) lists
    """
    print(f"📖 Loading texts from {manifest_path}...")
    
    # Read CSV in chunks to handle large files
    texts = []
    languages = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(manifest_path, chunksize=chunk_size):
        transcripts = chunk['transcript'].tolist()
        for t in transcripts:
            text, lang = extract_text_from_transcript(t)
            texts.append(text)
            languages.append(lang)
        
        if max_samples and len(texts) >= max_samples:
            texts = texts[:max_samples]
            languages = languages[:max_samples]
            break
        
        print(f"   Loaded {len(texts):,} texts...", end='\r')
    
    print(f"\n✅ Loaded {len(texts):,} texts")
    print(f"   Vietnamese: {languages.count('vi'):,} ({languages.count('vi')/len(languages)*100:.1f}%)")
    print(f"   English: {languages.count('en'):,} ({languages.count('en')/len(languages)*100:.1f}%)")
    return texts, languages


def normalize_texts(texts: List[str], languages: List[str], normalizer: BilingualTextNormalizer) -> List[str]:
    """
    Normalize texts using BilingualTextNormalizer with language awareness.
    
    Args:
        texts: List of raw texts
        languages: List of language codes ('vi' or 'en')
        normalizer: Text normalizer instance
        
    Returns:
        List of normalized texts
    """
    print("🔧 Normalizing texts...")
    
    num_workers = multiprocessing.cpu_count()
    batch_size = max(1000, len(texts) // (num_workers * 4))
    
    normalized_texts = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_langs = languages[i:i + batch_size]
        normalized_batch = [
            normalizer.normalize(text, lang=lang) 
            for text, lang in zip(batch_texts, batch_langs)
        ]
        normalized_texts.extend(normalized_batch)
        
        if (i // batch_size) % 10 == 0:
            print(f"   Normalized {len(normalized_texts):,}/{len(texts):,} texts...", end='\r')
    
    print(f"\n✅ Normalized {len(normalized_texts):,} texts")
    return normalized_texts


def train_sentencepiece_tokenizer(
    texts: List[str],
    vocab_size: int = 3500,
    model_prefix: str = "models/tokenizer_vi_en_3500",
    character_coverage: float = 0.9995,
    model_type: str = "bpe"
):
    """
    Train SentencePiece tokenizer.
    
    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        model_prefix: Prefix for output files (.model and .vocab will be added)
        character_coverage: Character coverage (0.9995 = 99.95%)
        model_type: Model type ("bpe", "unigram", "char", "word")
    """
    print(f"\n🚀 Training SentencePiece tokenizer...")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Model type: {model_type}")
    print(f"   Character coverage: {character_coverage}")
    print(f"   Training texts: {len(texts):,}")
    
    # Create output directory if it doesn't exist
    output_path = Path(model_prefix).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save texts to temporary file for SentencePiece training
    temp_text_file = output_path / "temp_training_texts.txt"
    print(f"\n💾 Saving texts to temporary file: {temp_text_file}")
    
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print(f"✅ Saved {len(texts):,} texts to temporary file")
    
    # SentencePiece training parameters
    # Optimized for Ryzen 9 9990X with 64GB RAM
    spm.SentencePieceTrainer.train(
        input=str(temp_text_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,  # BPE for subword tokenization
        character_coverage=character_coverage,
        num_threads=multiprocessing.cpu_count(),  # Use all CPU cores
        input_sentence_size=len(texts),  # Use all sentences
        shuffle_input_sentence=True,  # Shuffle for better training
        seed_sentencepiece_size=1000000,  # Large seed for better coverage
        shrinking_factor=0.75,
        max_sentence_length=4192,  # Maximum sentence length
        # Special tokens for seq2seq
        user_defined_symbols=['<|vi|>', '<|en|>'],  # Language tags
        # BPE-specific parameters
        byte_fallback=True,  # Handle unknown characters
        split_by_unicode_script=True,  # Better handling of multilingual text
        split_by_whitespace=True,  # Split by whitespace
        normalization_rule_name='nmt_nfkc_cf',  # Normalization for multilingual
        remove_extra_whitespaces=True,
    )
    
    # Clean up temporary file
    temp_text_file.unlink()
    print(f"\n🗑️  Removed temporary file")
    
    # Verify the model was created
    model_file = Path(f"{model_prefix}.model")
    vocab_file = Path(f"{model_prefix}.vocab")
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not created: {model_file}")
    
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not created: {vocab_file}")
    
    # Load and verify tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    
    actual_vocab_size = sp.get_piece_size()
    print(f"\n✅ Tokenizer training completed!")
    print(f"   Model file: {model_file}")
    print(f"   Vocab file: {vocab_file}")
    print(f"   Actual vocabulary size: {actual_vocab_size}")
    
    # Test tokenizer
    test_texts = [
        "xin chào việt nam",
        "hello world",
        "tôi là sinh viên",
        "this is a test"
    ]
    
    print(f"\n🧪 Testing tokenizer:")
    for text in test_texts:
        token_ids = sp.encode(text, out_type=int, add_bos=False, add_eos=False)
        decoded = sp.decode(token_ids)
        print(f"   '{text}' -> {len(token_ids)} tokens -> '{decoded}'")
    
    return model_file, vocab_file


def main():
    """Main function to train tokenizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for seq2seq")
    parser.add_argument(
        '--manifest',
        type=str,
        default='data/processed/full_merged_dataset/train/manifest.csv',
        help='Path to training manifest CSV'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=3500,
        help='Vocabulary size (default: 3500)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/tokenizer_vi_en_3500',
        help='Output model prefix (default: models/tokenizer_vi_en_3500)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (None = all)'
    )
    parser.add_argument(
        '--skip-normalize',
        action='store_true',
        help='Skip text normalization (use raw texts)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SentencePiece Tokenizer Training for Seq2Seq")
    print("=" * 80)
    print(f"Manifest: {args.manifest}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Output prefix: {args.output}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Skip normalization: {args.skip_normalize}")
    print("=" * 80)
    
    # Load texts
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    texts, languages = load_training_texts(str(manifest_path), max_samples=args.max_samples)
    
    # Normalize texts (optional)
    if not args.skip_normalize:
        normalizer = BilingualTextNormalizer()
        texts = normalize_texts(texts, languages, normalizer)
    
    # Train tokenizer
    model_file, vocab_file = train_sentencepiece_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        model_prefix=args.output
    )
    
    print("\n" + "=" * 80)
    print("✅ Tokenizer training completed successfully!")
    print(f"   Model: {model_file}")
    print(f"   Vocab: {vocab_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

