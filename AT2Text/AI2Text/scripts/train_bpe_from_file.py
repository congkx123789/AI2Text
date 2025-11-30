"""
Train BPE tokenizer from a text file containing all transcripts.
This ensures we train on the complete vocabulary from all datasets.

Usage:
    python scripts/train_bpe_from_file.py --input all_texts_for_bpe.txt --vocab-size 16000
"""

import argparse
import sys
import signal
import gc
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.bpe_tokenizer import BPETokenizer, INTERRUPT_RECEIVED

# Global flag for graceful exit
def signal_handler(sig, frame):
    global INTERRUPT_RECEIVED
    print("\n🛑 Đã nhận lệnh dừng! Đang lưu dữ liệu và thoát an toàn...")
    INTERRUPT_RECEIVED = True

signal.signal(signal.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer from text file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file (one text per line)')
    parser.add_argument('--vocab-size', type=int, default=16000,
                       help='Vocabulary size (default: 16000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: models/bilingual_bpe_{vocab_size}.json)')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum frequency for BPE merges (default: 2)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: None = use ALL CPU cores)')
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        output_path = f"models/bilingual_bpe_{args.vocab_size}.json"
    
    # Ensure models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)
    
    # Read texts from file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    print("=" * 60)
    print("Training BPE Tokenizer from Text File")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Output: {output_path}")
    print()
    
    print("📖 Reading texts...")
    texts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:  # Skip empty lines
                texts.append(text)
    
    print(f"✓ Read {len(texts):,} texts")
    print()
    
    if not texts:
        print("❌ No texts found in input file.")
        return 1
    
    # Train BPE tokenizer
    import multiprocessing
    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()
    
    print("🔧 Training BPE tokenizer (FULL CPU optimization)...")
    print(f"   Vocabulary size: {args.vocab_size:,}")
    print(f"   Parallel workers: {args.num_workers} (ALL CPU cores/threads)")
    print()
    
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, num_workers=args.num_workers)
    
    # Save tokenizer
    print()
    print("💾 Saving tokenizer...")
    tokenizer.save(output_path)
    
    print()
    print("=" * 60)
    print("✅ BPE Tokenizer Training Complete!")
    print("=" * 60)
    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Saved to: {output_path}")
    print()
    print("📝 Next steps:")
    print("   1. Update all config files to use this vocabulary:")
    print(f"      bpe_vocab_path: \"{output_path}\"")
    print(f"      vocab_size: {args.vocab_size}")
    print("   2. Start training with this FIXED vocabulary")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

