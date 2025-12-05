"""
Tests for evaluation metrics.
"""

import pytest

from utils.metrics import (
    levenshtein_distance,
    calculate_wer,
    calculate_cer,
    calculate_accuracy
)


class TestLevenshteinDistance:
    """Test Levenshtein distance calculation."""
    
    def test_identical_strings(self):
        """Test with identical strings."""
        ref = list("hello")
        hyp = list("hello")
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == 0
    
    def test_different_strings(self):
        """Test with different strings."""
        ref = list("hello")
        hyp = list("world")
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance > 0
        assert distance == 4  # All characters different
    
    def test_one_insertion(self):
        """Test with one character insertion."""
        ref = list("hello")
        hyp = list("helloo")
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == 1
    
    def test_one_deletion(self):
        """Test with one character deletion."""
        ref = list("hello")
        hyp = list("hell")
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == 1
    
    def test_one_substitution(self):
        """Test with one character substitution."""
        ref = list("hello")
        hyp = list("hallo")
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == 1
    
    def test_empty_strings(self):
        """Test with empty strings."""
        ref = []
        hyp = []
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == 0
    
    def test_one_empty_string(self):
        """Test with one empty string."""
        ref = list("hello")
        hyp = []
        
        distance = levenshtein_distance(ref, hyp)
        
        assert distance == len(ref)


class TestWER:
    """Test Word Error Rate calculation."""
    
    def test_perfect_match(self):
        """Test with perfect transcription."""
        references = ["xin chào việt nam"]
        hypotheses = ["xin chào việt nam"]
        
        wer = calculate_wer(references, hypotheses)
        
        assert wer == 0.0
    
    def test_one_word_error(self):
        """Test with one word error."""
        references = ["xin chào việt nam"]
        hypotheses = ["xin chào việt bắc"]
        
        wer = calculate_wer(references, hypotheses)
        
        assert wer > 0
        assert wer < 1.0
    
    def test_multiple_samples(self):
        """Test with multiple samples."""
        references = [
            "xin chào việt nam",
            "tôi là sinh viên",
            "hôm nay trời đẹp"
        ]
        hypotheses = [
            "xin chào việt nam",
            "tôi là học sinh",
            "hôm nay trời đẹp"
        ]
        
        wer = calculate_wer(references, hypotheses)
        
        assert wer >= 0
        assert wer <= 1.0
    
    def test_completely_different(self):
        """Test with completely different transcriptions."""
        references = ["xin chào việt nam"]
        hypotheses = ["hello world"]
        
        wer = calculate_wer(references, hypotheses)
        
        assert wer > 0
        assert wer <= 1.0
    
    def test_empty_references(self):
        """Test with empty references."""
        references = [""]
        hypotheses = ["xin chào"]
        
        # Should handle gracefully
        wer = calculate_wer(references, hypotheses)
        assert wer >= 0


class TestCER:
    """Test Character Error Rate calculation."""
    
    def test_perfect_match(self):
        """Test with perfect transcription."""
        references = ["xin chào"]
        hypotheses = ["xin chào"]
        
        cer = calculate_cer(references, hypotheses)
        
        assert cer == 0.0
    
    def test_one_character_error(self):
        """Test with one character error."""
        references = ["xin chào"]
        hypotheses = ["xin chảo"]
        
        cer = calculate_cer(references, hypotheses)
        
        assert cer > 0
        assert cer < 1.0
    
    def test_multiple_samples(self):
        """Test with multiple samples."""
        references = [
            "xin chào",
            "việt nam",
            "tôi là"
        ]
        hypotheses = [
            "xin chào",
            "việt bắc",
            "tôi là"
        ]
        
        cer = calculate_cer(references, hypotheses)
        
        assert cer >= 0
        assert cer <= 1.0
    
    def test_vietnamese_characters(self):
        """Test with Vietnamese characters."""
        references = ["xin chào việt nam"]
        hypotheses = ["xin chao viet nam"]
        
        cer = calculate_cer(references, hypotheses)
        
        assert cer > 0  # Missing diacritics


class TestAccuracy:
    """Test accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test with perfect transcriptions."""
        references = [
            "xin chào",
            "việt nam",
            "tôi là sinh viên"
        ]
        hypotheses = [
            "xin chào",
            "việt nam",
            "tôi là sinh viên"
        ]
        
        accuracy = calculate_accuracy(references, hypotheses)
        
        assert accuracy == 1.0
    
    def test_zero_accuracy(self):
        """Test with all incorrect."""
        references = [
            "xin chào",
            "việt nam"
        ]
        hypotheses = [
            "hello",
            "world"
        ]
        
        accuracy = calculate_accuracy(references, hypotheses)
        
        assert accuracy == 0.0
    
    def test_partial_accuracy(self):
        """Test with partial matches."""
        references = [
            "xin chào",
            "việt nam",
            "tôi là"
        ]
        hypotheses = [
            "xin chào",
            "hello",
            "tôi là"
        ]
        
        accuracy = calculate_accuracy(references, hypotheses)
        
        assert accuracy > 0
        assert accuracy < 1.0
        assert accuracy == pytest.approx(2/3, rel=0.1)

