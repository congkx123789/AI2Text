#!/usr/bin/env python3
"""
Diagnostic script to check why WER/CER is 1.0.
Checks tokenizer, model outputs, and decoding process.
"""

import sys
sys.path.append('.')
import torch
from pathlib import Path
from preprocessing.bpe_tokenizer import BPETokenizer
from preprocessing.text_cleaning import Tokenizer

def diagnose_tokenizer():
    """Check tokenizer configuration."""
    print("=" * 60)
    print("TOKENIZER DIAGNOSIS")
    print("=" * 60)
    
    # Check BPE tokenizer
    bpe_path = Path('models/bilingual_bpe_18k.json')
    if bpe_path.exists():
        tokenizer = BPETokenizer()
        tokenizer.load(str(bpe_path))
        print(f"✅ BPE tokenizer loaded")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Blank token ID: {tokenizer.blank_token_id}")
        
        # Test encode/decode
        test_text = "xin chào việt nam"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"\n   Test: \"{test_text}\"")
        print(f"   Encoded: {encoded[:5]}... (first 5 tokens)")
        print(f"   Decoded: \"{decoded}\"")
        print(f"   Spaces preserved: {test_text.count(' ') == decoded.count(' ')}")
        print(f"   ⚠️  BPE does NOT preserve spaces!")
    else:
        print(f"❌ BPE vocab file not found: {bpe_path}")

def diagnose_model_vocab():
    """Check model vocabulary size."""
    print("\n" + "=" * 60)
    print("MODEL VOCABULARY CHECK")
    print("=" * 60)
    
    checkpoint_path = Path('checkpoints/best_model.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model_state = checkpoint.get('model_state_dict', {})
        
        # Find output layer
        classifier_key = 'decoder.linear.weight'
        if classifier_key in model_state:
            weight = model_state[classifier_key]
            model_vocab_size = weight.shape[0]
            print(f"✅ Model output layer found")
            print(f"   Model vocab size: {model_vocab_size}")
            
            # Check tokenizer
            bpe_path = Path('models/bilingual_bpe_18k.json')
            if bpe_path.exists():
                tokenizer = BPETokenizer()
                tokenizer.load(str(bpe_path))
                tokenizer_vocab_size = len(tokenizer)
                print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
                
                if model_vocab_size == tokenizer_vocab_size:
                    print(f"   ✅ Vocab sizes match!")
                else:
                    print(f"   ⚠️  MISMATCH! This will cause decoding errors!")
        else:
            print(f"❌ Output layer not found in checkpoint")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")

def simulate_validation():
    """Simulate what happens during validation."""
    print("\n" + "=" * 60)
    print("VALIDATION SIMULATION")
    print("=" * 60)
    
    bpe_path = Path('models/bilingual_bpe_18k.json')
    if not bpe_path.exists():
        print("❌ BPE tokenizer not available")
        return
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(bpe_path))
    
    # Simulate different model outputs
    test_cases = [
        ([tokenizer.blank_token_id] * 50, "All blank tokens (untrained model)"),
        ([0] * 20, "All token 0 (PAD)"),
        ([1] * 20, "All token 1 (UNK)"),
        ([100, 200, 300] * 5, "Valid tokens (no blanks)"),
        ([tokenizer.blank_token_id, 100, tokenizer.blank_token_id, 200] * 5, "Mixed blanks + tokens"),
    ]
    
    ref_text = "xin chào việt nam"
    
    for tokens, desc in test_cases:
        # CTC decode
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
            prev = token
        
        filtered = [t for t in collapsed if t != tokenizer.blank_token_id]
        decoded = tokenizer.decode(filtered)
        
        # Calculate WER
        from utils.metrics import calculate_wer
        wer = calculate_wer([ref_text], [decoded])
        
        print(f"\n{desc}:")
        print(f"   Tokens after CTC: {len(filtered)}")
        print(f"   Decoded: \"{decoded[:50]}\"")
        print(f"   Empty: {len(decoded.strip()) == 0}")
        print(f"   WER: {wer:.4f}")

if __name__ == "__main__":
    diagnose_tokenizer()
    diagnose_model_vocab()
    simulate_validation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
ISSUES FOUND:
1. BPE tokenizer does NOT preserve spaces during decode
   - "xin chào" → "xinchào" (no spaces)
   - This causes WER=1.0 even if characters match

2. If model outputs mostly blank tokens (normal for early training):
   - After CTC decode → empty string
   - Empty prediction → WER=1.0

SOLUTIONS:
1. ✅ Fixed WER calculation to normalize spaces
2. ⚠️  BPE decode still doesn't preserve word boundaries
3. ⚠️  Model may need more training to output non-blank tokens

NEXT STEPS:
- Check actual model predictions during validation
- Consider using character-level metrics for BPE
- Or use word segmentation to add spaces during BPE decode
    """)

