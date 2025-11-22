"""
Integration tests for the complete ASR system.

Tests end-to-end workflows:
1. Data preparation → Embeddings training → Model training
2. Training → Evaluation with beam search
3. Beam search → N-best rescoring with embeddings
4. Complete inference pipeline
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import sqlite3
import numpy as np

from database.db_utils import ASRDatabase
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from preprocessing.bpe_tokenizer import BPETokenizer
from models.lstm_asr import LSTMASRModel
from models.asr_base import ASRModel
from models.enhanced_asr import EnhancedASRModel
from training.dataset import ASRDataset, create_data_loaders
from decoding.beam_search import BeamSearchDecoder, generate_nbest
from decoding.rescoring import rescore_nbest, contextual_biasing
from utils.metrics import calculate_wer, calculate_cer


class TestIntegration:
    """Integration tests for complete ASR pipeline."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_db.db"
        db = ASRDatabase(str(db_path))
        
        yield db_path, db
        
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return [
            {"audio_file": "test1.wav", "transcript": "xin chào việt nam"},
            {"audio_file": "test2.wav", "transcript": "tôi là sinh viên"},
            {"audio_file": "test3.wav", "transcript": "hôm nay trời đẹp"}
        ]
    
    def test_database_data_preparation(self, temp_db):
        """Test database initialization and data preparation."""
        db_path, db = temp_db
        
        # Add sample audio file
        audio_id = db.add_audio_file(
            file_path="test_audio.wav",
            dataset_name="test_dataset",
            duration=5.0,
            sample_rate=16000
        )
        assert audio_id is not None
        
        # Add transcript
        transcript_id = db.add_transcript(
            audio_id=audio_id,
            transcript="xin chào việt nam",
            language="vi"
        )
        assert transcript_id is not None
        
        # Verify data
        transcript = db.get_transcript(audio_id)
        assert transcript == "xin chào việt nam"
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        # Audio processor
        processor = AudioProcessor(sample_rate=16000, n_mels=80)
        
        # Create dummy audio (random)
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second
        
        # Extract features
        features = processor.extract_features(dummy_audio)
        assert features.shape[1] == 80  # n_mels
        
        # Text normalization
        normalizer = VietnameseTextNormalizer()
        text = "Xin chào VIỆT NAM"
        normalized = normalizer.normalize(text)
        assert normalized == "xin chào việt nam"
        
        # Tokenization
        tokenizer = Tokenizer()
        tokens = tokenizer.encode(normalized)
        assert len(tokens) > 0
        
        # BPE tokenization
        bpe_tokenizer = BPETokenizer()
        texts = ["xin chào", "việt nam", "tôi là sinh viên"]
        bpe_tokenizer.train(texts, vocab_size=100, min_frequency=1)
        bpe_tokens = bpe_tokenizer.encode("xin chào")
        assert len(bpe_tokens) > 0
    
    def test_model_architectures(self):
        """Test all model architectures."""
        vocab_size = 100
        batch_size = 2
        seq_len = 100
        input_dim = 80
        
        # LSTM model
        lstm_model = LSTMASRModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_size=128
        )
        x_lstm = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len, seq_len])
        logits_lstm, _ = lstm_model(x_lstm, lengths)
        assert logits_lstm.shape == (batch_size, seq_len, vocab_size)
        
        # Transformer model
        transformer_model = ASRModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=128
        )
        x_trans = torch.randn(batch_size, seq_len, input_dim)
        logits_trans, _ = transformer_model(x_trans, lengths)
        assert logits_trans.shape == (batch_size, seq_len, vocab_size)
        
        # Enhanced model
        enhanced_model = EnhancedASRModel(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=128,
            use_contextual_embeddings=True,
            use_cross_modal_attention=True
        )
        x_enhanced = torch.randn(batch_size, seq_len, input_dim)
        text_context = torch.randint(0, vocab_size, (batch_size, 20))
        output = enhanced_model(x_enhanced, lengths, text_context=text_context)
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_beam_search_decoding(self):
        """Test beam search decoding."""
        vocab_size = 100
        blank_token_id = 2
        
        decoder = BeamSearchDecoder(
            vocab_size=vocab_size,
            blank_token_id=blank_token_id,
            beam_width=5
        )
        
        # Create dummy logits
        batch_size = 2
        seq_len = 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([seq_len, seq_len - 10])
        
        # Decode
        results = decoder.decode(logits, lengths)
        assert len(results) == batch_size
        assert len(results[0]) > 0  # Has hypotheses
        
        # Check N-best generation
        nbest = generate_nbest(logits, decoder, n=5, lengths=lengths)
        assert len(nbest) == batch_size
        assert len(nbest[0]) <= 5
    
    def test_embeddings_integration(self, temp_db):
        """Test embeddings training and usage (mock)."""
        db_path, db = temp_db
        
        # Note: Full Word2Vec/Phon2Vec training requires gensim
        # This test verifies the interface
        
        # Test phonetic tokenization (used by Phon2Vec)
        from preprocessing.phonetic import phonetic_tokens
        
        text = "xin chào việt nam"
        ph_tokens = phonetic_tokens(text, telex=True, tone_token=True)
        assert len(ph_tokens) > 0
        assert isinstance(ph_tokens, list)
    
    def test_nbest_rescoring_mock(self):
        """Test N-best rescoring with mock embeddings."""
        # Create mock N-best list
        nbest = [
            {"text": "toi muon dat ban", "am_score": -12.5, "lm_score": -1.1},
            {"text": "toi muon dat banh", "am_score": -12.8, "lm_score": -1.0}
        ]
        
        # Rescore without embeddings (baseline)
        rescored_baseline = rescore_nbest(
            nbest,
            semantic_kv=None,
            phon_kv=None,
            alpha=1.0,
            beta=0.0
        )
        assert len(rescored_baseline) == len(nbest)
        assert "re_score" in rescored_baseline[0]
        
        # With context text (still works without embeddings)
        rescored_context = rescore_nbest(
            nbest,
            semantic_kv=None,
            phon_kv=None,
            context_text="đặt bánh gato",
            alpha=1.0,
            beta=0.0,
            gamma=0.5,
            delta=0.5
        )
        assert len(rescored_context) == len(nbest)
    
    def test_contextual_biasing_mock(self):
        """Test contextual biasing (mock)."""
        nbest = [
            {"text": "toi muon dat ban", "am_score": -12.5},
            {"text": "toi muon dat banh", "am_score": -12.8}
        ]
        
        bias_list = ["bánh", "gato"]
        
        # Biasing works even without embeddings (uses text matching)
        biased = contextual_biasing(
            nbest,
            bias_list=bias_list,
            semantic_kv=None,
            phon_kv=None,
            bias_weight=0.3
        )
        assert len(biased) == len(nbest)
    
    def test_metrics_calculation(self):
        """Test WER/CER calculation."""
        references = [
            "xin chào việt nam",
            "tôi là sinh viên"
        ]
        predictions = [
            "xin chào việt nam",
            "tôi là sinh viên học"
        ]
        
        wer = calculate_wer(references, predictions)
        cer = calculate_cer(references, predictions)
        
        assert 0 <= wer <= 1
        assert 0 <= cer <= 1
        assert wer == 0.0  # First matches
        assert cer > 0  # Second has extra characters
    
    def test_end_to_end_workflow(self, temp_db):
        """Test complete end-to-end workflow (simplified)."""
        db_path, db = temp_db
        
        # 1. Add data to database
        audio_id = db.add_audio_file(
            file_path="test.wav",
            dataset_name="test",
            duration=5.0,
            sample_rate=16000
        )
        transcript_id = db.add_transcript(audio_id, "xin chào", "vi")
        
        # 2. Preprocessing
        processor = AudioProcessor()
        tokenizer = Tokenizer()
        dummy_audio = np.random.randn(16000).astype(np.float32)
        features = processor.extract_features(dummy_audio)
        
        # 3. Model inference
        model = LSTMASRModel(input_dim=80, vocab_size=len(tokenizer), hidden_size=128)
        model.eval()
        x = torch.from_numpy(features).unsqueeze(0)
        logits, lengths = model(x)
        
        # 4. Decoding
        decoder = BeamSearchDecoder(
            vocab_size=len(tokenizer),
            blank_token_id=tokenizer.blank_token_id,
            beam_width=5
        )
        results = decoder.decode(logits, lengths)
        
        assert len(results) == 1
        assert len(results[0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

