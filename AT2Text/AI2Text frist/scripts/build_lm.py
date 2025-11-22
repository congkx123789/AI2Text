"""
Build KenLM language model from transcripts.

Trains KenLM language model from database transcripts for improved ASR accuracy.
This provides significant WER improvement (typically 10-30%).
"""

import argparse
import sqlite3
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.text_cleaning import VietnameseTextNormalizer
from database.db_utils import ASRDatabase


def extract_transcripts(db_path: str, output_file: str):
    """
    Extract transcripts from database and write to text file.
    
    Args:
        db_path: Path to SQLite database
        output_file: Path to output text file (one transcript per line)
    """
    con = sqlite3.connect(db_path)
    normalizer = VietnameseTextNormalizer()
    
    try:
        cur = con.execute("SELECT transcript FROM Transcripts WHERE transcript IS NOT NULL")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for (txt,) in cur.fetchall():
                if txt:
                    # Normalize text
                    normalized = normalizer.normalize(txt)
                    f.write(normalized + '\n')
        
        print(f"[OK] Extracted transcripts to {output_file}")
        
    finally:
        con.close()


def train_kenlm(corpus_file: str,
               output_arpa: str,
               ngram_order: int = 3,
               memory: str = "80%",
               prune: Optional[List[int]] = None):
    """
    Train KenLM language model.
    
    Args:
        corpus_file: Path to text corpus (one sentence per line)
        output_arpa: Path to output .arpa file
        ngram_order: N-gram order (default: 3 for trigram)
        memory: Memory limit for training (default: "80%")
        prune: Pruning thresholds for each order (optional)
    """
    from typing import Optional, List
    
    # Check if kenlm is installed
    try:
        result = subprocess.run(['lmplz', '--version'], 
                               capture_output=True, 
                               text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Error: KenLM tools not found!")
        print("Install KenLM with:")
        print("  Ubuntu/Debian: sudo apt-get install libkenlm-dev")
        print("  macOS: brew install kenlm")
        print("  Or build from source: https://github.com/kpu/kenlm")
        return False
    
    # Build command
    cmd = [
        'lmplz',
        '-o', str(ngram_order),
        '--memory', memory,
        '<', corpus_file,
        '>', output_arpa
    ]
    
    # Handle pruning
    if prune:
        for i, threshold in enumerate(prune):
            if i < ngram_order - 1:
                cmd.extend(['--prune', str(i+2), str(threshold)])
    
    print(f"Training KenLM model...")
    print(f"  Corpus: {corpus_file}")
    print(f"  Output: {output_arpa}")
    print(f"  Order: {ngram_order}")
    
    # Execute (using shell for input redirection)
    try:
        with open(corpus_file, 'r') as f:
            result = subprocess.run(
                ['lmplz', '-o', str(ngram_order), '--memory', memory],
                stdin=f,
                stdout=open(output_arpa, 'w'),
                stderr=subprocess.PIPE,
                text=True
            )
        
        if result.returncode == 0:
            print(f"[OK] KenLM model saved to {output_arpa}")
            return True
        else:
            print(f"Error training KenLM: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def build_binary_lm(arpa_file: str, output_bin: str):
    """
    Build binary KenLM model for faster loading.
    
    Args:
        arpa_file: Path to .arpa file
        output_bin: Path to output .bin file
    """
    try:
        result = subprocess.run(['build_binary', arpa_file, output_bin],
                               capture_output=True,
                               text=True)
        
        if result.returncode == 0:
            print(f"[OK] Binary model saved to {output_bin}")
            return True
        else:
            print(f"Error building binary: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Warning: build_binary not found. Using .arpa file.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Build KenLM language model from database')
    parser.add_argument('--db', type=str, default='database/asr_training.db',
                       help='Path to database')
    parser.add_argument('--output', type=str, default='models/lm.arpa',
                       help='Output .arpa file path')
    parser.add_argument('--order', type=int, default=3,
                       help='N-gram order (default: 3)')
    parser.add_argument('--memory', type=str, default='80%',
                       help='Memory limit (default: 80%%)')
    parser.add_argument('--binary', action='store_true',
                       help='Also build binary model')
    args = parser.parse_args()
    
    print("="*70)
    print("Building KenLM Language Model")
    print("="*70)
    print(f"Database: {args.db}")
    print(f"Output: {args.output}")
    print()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract transcripts
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        corpus_file = f.name
    
    try:
        extract_transcripts(args.db, corpus_file)
        
        # Train KenLM
        success = train_kenlm(
            corpus_file=corpus_file,
            output_arpa=args.output,
            ngram_order=args.order,
            memory=args.memory
        )
        
        if success:
            # Build binary if requested
            if args.binary:
                binary_path = str(output_path.with_suffix('.bin'))
                build_binary_lm(args.output, binary_path)
            
            print("="*70)
            print("[OK] Language model training completed!")
            print(f"Model saved to: {args.output}")
            print()
            print("Usage:")
            print(f"  from decoding.lm_decoder import LMBeamSearchDecoder")
            print(f"  decoder = LMBeamSearchDecoder(vocab=vocab, lm_path='{args.output}')")
            print("="*70)
        else:
            print("[ERROR] Language model training failed!")
            return 1
            
    finally:
        # Cleanup
        Path(corpus_file).unlink(missing_ok=True)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

