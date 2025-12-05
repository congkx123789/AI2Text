"""
Simple test runner that doesn't require pytest.
Can be used as an alternative to pytest for basic testing.
"""

import sys
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_func, test_name):
    """Run a single test function."""
    try:
        test_func()
        print(f"[PASS] {test_name}")
        return True
    except Exception as e:
        print(f"[FAIL] {test_name}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        return False


def test_database_basic():
    """Basic database test."""
    from database.db_utils import ASRDatabase
    import tempfile
    import os
    
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    try:
        db = ASRDatabase(db_path)
        assert db.db_path.exists()
        
        # Test adding audio file
        audio_id = db.add_audio_file(
            file_path="test/audio.wav",
            filename="audio.wav",
            duration=5.0,
            sample_rate=16000
        )
        assert audio_id is not None
        
        # Test retrieving audio file
        audio_file = db.get_audio_file(audio_id)
        assert audio_file is not None
        assert audio_file['file_path'] == "test/audio.wav"
        
    finally:
        os.unlink(db_path)


def test_metrics():
    """Test metrics calculation."""
    from utils.metrics import calculate_wer, calculate_cer, calculate_accuracy
    
    # Test WER
    references = ["xin chào việt nam"]
    hypotheses = ["xin chào việt nam"]
    wer = calculate_wer(references, hypotheses)
    assert wer == 0.0, f"Expected 0.0, got {wer}"
    
    # Test CER
    cer = calculate_cer(references, hypotheses)
    assert cer == 0.0, f"Expected 0.0, got {cer}"
    
    # Test accuracy
    accuracy = calculate_accuracy(references, hypotheses)
    assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"


def test_text_normalizer():
    """Test text normalizer."""
    from preprocessing.text_cleaning import VietnameseTextNormalizer
    
    normalizer = VietnameseTextNormalizer()
    
    # Test normalization
    text = "Xin Chào VIỆT NAM"
    normalized = normalizer.normalize(text)
    
    assert normalized.islower(), "Should be lowercase"
    assert "việt nam" in normalized.lower(), "Should contain Vietnamese text"


def test_tokenizer():
    """Test tokenizer."""
    from preprocessing.text_cleaning import Tokenizer
    
    tokenizer = Tokenizer()
    
    # Test encoding
    text = "xin chào"
    token_ids = tokenizer.encode(text)
    
    assert isinstance(token_ids, list), "Should return list"
    assert len(token_ids) > 0, "Should have tokens"
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    assert isinstance(decoded, str), "Should return string"
    assert len(decoded) > 0, "Should have decoded text"


def test_model_initialization():
    """Test model initialization."""
    from models.asr_base import ASRModel
    
    model = ASRModel(
        input_dim=80,
        vocab_size=100,
        d_model=128,
        num_encoder_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1
    )
    
    assert model is not None, "Model should be initialized"
    num_params = model.get_num_params()
    assert num_params > 0, "Model should have parameters"
    
    # Test forward pass
    import torch
    x = torch.randn(1, 50, 80)
    lengths = torch.tensor([50])
    
    logits, output_lengths = model(x, lengths)
    
    assert logits is not None, "Should produce logits"
    assert output_lengths is not None, "Should produce output lengths"


def main():
    """Run all tests."""
    print("="*70)
    print("Simple Test Runner")
    print("="*70)
    print()
    
    tests = [
        (test_database_basic, "Database Basic Operations"),
        (test_metrics, "Metrics Calculation"),
        (test_text_normalizer, "Text Normalization"),
        (test_tokenizer, "Tokenizer"),
        (test_model_initialization, "Model Initialization"),
    ]
    
    passed = 0
    failed = 0
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1
        print()
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

