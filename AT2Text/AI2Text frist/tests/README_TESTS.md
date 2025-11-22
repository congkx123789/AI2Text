# Testing Guide for ASR Training System

## 🧪 Test Suite Overview

The test suite covers all major components of the ASR training system:

- ✅ **Database Tests** (`test_database.py`) - Database operations, validation, statistics
- ✅ **Preprocessing Tests** (`test_preprocessing.py`) - Audio and text preprocessing
- ✅ **Model Tests** (`test_models.py`) - ASR model architecture and forward pass
- ✅ **Metrics Tests** (`test_metrics.py`) - WER, CER, accuracy calculations
- ✅ **Training Tests** (`test_training.py`) - Dataset, data loaders, training components

## 🚀 Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_database.py
pytest tests/test_preprocessing.py
pytest tests/test_models.py
```

### Run Specific Test Class or Function
```bash
pytest tests/test_database.py::TestASRDatabase
pytest tests/test_database.py::TestASRDatabase::test_add_audio_file
```

### Run with Coverage Report
```bash
pytest --cov=. --cov-report=html
```

### Run in Parallel (Faster)
```bash
pytest -n auto
```

### Run Only Fast Tests (Skip Slow Ones)
```bash
pytest -m "not slow"
```

### Verbose Output
```bash
pytest -v
```

### Show Print Statements
```bash
pytest -s
```

## 📊 Test Coverage

### Current Coverage:
- Database operations: ✅ Complete
- Audio preprocessing: ✅ Complete
- Text preprocessing: ✅ Complete
- Model architecture: ✅ Complete
- Metrics calculation: ✅ Complete
- Training components: ✅ Complete

### Test Statistics:
- Total test files: 5
- Total test classes: 6+
- Total test functions: 50+

## 🔧 Test Fixtures

The test suite includes comprehensive fixtures (`conftest.py`):
- `temp_db` - Temporary database for testing
- `temp_dir` - Temporary directory for files
- `sample_audio_file` - Sample WAV file
- `audio_processor` - AudioProcessor instance
- `audio_augmenter` - AudioAugmenter instance
- `text_normalizer` - VietnameseTextNormalizer instance
- `tokenizer` - Tokenizer instance
- `asr_model` - ASR model instance
- `sample_transcripts` - Vietnamese transcript samples

## 📝 Writing New Tests

### Test File Structure
```python
"""
Tests for your_module.
"""

import pytest
from your_module import YourClass

class TestYourClass:
    """Test YourClass."""
    
    def test_something(self, fixture_name):
        """Test something."""
        # Arrange
        obj = YourClass()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result is not None
```

### Best Practices:
1. **Use descriptive test names**: `test_add_audio_file_duplicate_skip` not `test_1`
2. **One assertion per test**: Focus on one behavior
3. **Use fixtures**: Reuse setup code
4. **Test edge cases**: Empty inputs, None values, etc.
5. **Use parametrize**: Test multiple inputs efficiently

### Example with Parametrize:
```python
@pytest.mark.parametrize("input,expected", [
    ("Xin Chào", "xin chào"),
    ("VIỆT NAM", "việt nam"),
])
def test_normalization(input, expected, text_normalizer):
    result = text_normalizer.normalize(input)
    assert result == expected
```

## 🐛 Debugging Tests

### Run with Debugger
```bash
pytest --pdb
```

### Show Local Variables on Failure
```bash
pytest -l
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

## ✅ Continuous Integration

Tests can be integrated into CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=. --cov-report=xml
```

## 📈 Improving Test Coverage

To see what's not covered:
```bash
pytest --cov=. --cov-report=term-missing
```

Focus on:
1. Error handling paths
2. Edge cases
3. Integration between modules
4. Real-world scenarios

---

**Happy Testing!** 🎉

