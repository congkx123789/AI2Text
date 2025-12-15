"""
Quick test training script - trains on a single audio file for 20 epochs.
Useful for testing the training pipeline without using the full dataset.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import sys
import os
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
from utils.logger import setup_logger

def create_test_data(audio_file_path: str, transcript: str, audio_processor, tokenizer, device):
    """Create a single test sample from an audio file."""
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_file_path, sr=16000)
    
    # Process audio - extract mel spectrogram
    # Returns (n_mels, time), need to transpose to (time, n_mels)
    mel_spec = audio_processor.extract_mel_spectrogram(audio)
    # Transpose: (n_mels, time) -> (time, n_mels)
    mel_spec = mel_spec.T
    audio_features = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # (1, time, freq)
    audio_lengths = torch.tensor([audio_features.size(1)], dtype=torch.long)
    
    # Process text
    text_tokens = tokenizer.encode(transcript)
    text_tokens = torch.tensor([text_tokens], dtype=torch.long)  # (1, seq_len)
    text_lengths = torch.tensor([len(text_tokens[0])], dtype=torch.long)
    
    # Language ID (0 = Vietnamese, 1 = English)
    language_ids = torch.tensor([0], dtype=torch.long)  # Assume Vietnamese
    
    return {
        'audio_features': audio_features.to(device),
        'audio_lengths': audio_lengths.to(device),
        'text_tokens': text_tokens.to(device),
        'text_lengths': text_lengths.to(device),
        'language_ids': language_ids.to(device),
        'transcripts': [transcript]
    }

def main():
    """Quick test training on a single audio file."""
    print("="*80)
    print("QUICK TEST TRAINING - Single Audio File")
    print("="*80)
    
    # Load config
    config_path = 'configs/default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config for quick test
    config['num_epochs'] = 20
    config['batch_size'] = 1
    config['val_batch_size'] = 1
    config['use_amp'] = True
    config['use_bf16'] = True
    config['gradient_accumulation_steps'] = 1
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('QuickTest', 'logs/quick_test.log')
    
    logger.info("="*60)
    logger.info("Quick Test Training")
    logger.info(f"Device: {device}")
    logger.info("="*60)
    
    # Load tokenizer
    tokenizer_path = config.get('bpe_vocab_path', 'models/tokenizer_vi_en_3500.model')
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    vocab_size = config.get('vocab_size', 3500)
    
    # Audio processor
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80),
        n_fft=config.get('n_fft', 400),
        hop_length=config.get('hop_length', 160),
        win_length=config.get('win_length', 400)
    )
    
    # Create model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 14),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 2048),
        dropout=config.get('dropout', 0.2),
        num_languages=2,
        use_gradient_checkpointing=True
    )
    model.to(device)
    
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.0003),
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    # Mixed precision
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16 if config.get('use_bf16', True) and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler() if use_amp else None
    
    logger.info(f"Mixed Precision: {use_amp}")
    if use_amp:
        dtype_str = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        logger.info(f"AMP Dtype: {dtype_str}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    # Special token IDs
    sos_token_id = getattr(tokenizer, 'sos_token_id', 2)
    eos_token_id = getattr(tokenizer, 'eos_token_id', 3)
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    
    # Find a test audio file
    test_audio_dir = Path('data/processed/full_merged_dataset/train/audio')
    audio_files = list(test_audio_dir.glob('*.wav')) if test_audio_dir.exists() else []
    
    if not audio_files:
        logger.error("No audio files found! Please provide an audio file path.")
        print("\n❌ ERROR: No audio files found in data/processed/full_merged_dataset/train/audio")
        print("Please provide an audio file path or ensure the dataset is available.")
        return
    
    # Use first audio file
    audio_file = audio_files[0]
    logger.info(f"Using test audio: {audio_file}")
    print(f"\n📁 Test Audio: {audio_file}")
    
    # Get transcript from manifest if available
    manifest_path = Path('data/processed/full_merged_dataset/train/manifest.csv')
    transcript = "test transcript"
    
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        audio_name = audio_file.name
        # Try different column names
        matching = df[df['audio_path'].str.contains(audio_name, na=False) if 'audio_path' in df.columns else df[df.iloc[:, 0].str.contains(audio_name, na=False)]]
        if not matching.empty:
            # Try different transcript column names
            if 'transcript' in matching.columns:
                transcript = matching.iloc[0]['transcript']
            elif 'text' in matching.columns:
                transcript = matching.iloc[0]['text']
            elif len(matching.columns) > 1:
                transcript = str(matching.iloc[0].iloc[1])  # Second column
            else:
                transcript = "test transcript"
    
    print(f"📝 Transcript: {transcript}")
    logger.info(f"Transcript: {transcript}")
    
    # Create test data
    print("\n🔄 Creating test data...")
    test_data = create_test_data(str(audio_file), transcript, audio_processor, tokenizer, device)
    
    # Training loop
    num_epochs = 20
    model.train()
    
    print("\n" + "="*80)
    print("🚀 STARTING QUICK TEST TRAINING")
    print("="*80)
    print(f"📊 Epochs: {num_epochs}")
    print(f"📁 Audio: {audio_file.name}")
    print(f"📝 Transcript: {transcript[:50]}..." if len(transcript) > 50 else f"📝 Transcript: {transcript}")
    print("="*80 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        # Prepare target tokens (shift right, add SOS)
        text_tokens = test_data['text_tokens']
        batch_size = text_tokens.size(0)
        sos_tokens = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        target_tokens = torch.cat([sos_tokens, text_tokens], dim=1)
        
        # Forward pass
        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits, _ = model(
                x=test_data['audio_features'],
                tgt_tokens=target_tokens[:, :-1],
                lengths=test_data['audio_lengths'],
                language_ids=test_data['language_ids']
            )
            
            # Compute loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = target_tokens[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        total_loss = loss.item()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update best loss
        if total_loss < best_loss:
            best_loss = total_loss
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {total_loss:.6f} | Avg: {total_loss:.6f} | "
              f"LR: {current_lr:.2e} | Best: {best_loss:.6f} | Time: {epoch_time:.2f}s")
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.6f} | LR: {current_lr:.2e}")
    
    print("\n" + "="*80)
    print("✅ QUICK TEST TRAINING COMPLETE")
    print("="*80)
    print(f"📊 Final Loss: {total_loss:.6f}")
    print(f"🏆 Best Loss: {best_loss:.6f}")
    print("="*80)
    
    logger.info("Quick test training completed")
    logger.info(f"Final Loss: {total_loss:.6f}")
    logger.info(f"Best Loss: {best_loss:.6f}")
    
    # Save checkpoint
    print("\n💾 Saving checkpoint...")
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'final_loss': total_loss,
        'config': config,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    checkpoint_path = checkpoint_dir / 'quick_test_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Checkpoint saved: {checkpoint_path}")
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Also save as best_model for consistency
    best_model_path = checkpoint_dir / 'quick_test_best_model.pt'
    torch.save(checkpoint, best_model_path)
    print(f"✅ Best model saved: {best_model_path}")
    logger.info(f"Best model saved: {best_model_path}")
    
    # Test inference
    print("\n🧪 Testing inference...")
    model.eval()
    with torch.no_grad():
        with autocast(enabled=use_amp, dtype=amp_dtype):
            generated = model.generate(
                test_data['audio_features'],
                lengths=test_data['audio_lengths'],
                language_ids=test_data['language_ids'],
                max_len=128,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                temperature=1.0
            )
            
            # Decode
            gen_seq = generated[0].cpu().tolist()
            decoded_tokens = []
            for token in gen_seq:
                if token == eos_token_id:
                    break
                if token != sos_token_id and token != pad_token_id:
                    decoded_tokens.append(token)
            
            predicted_text = tokenizer.decode(decoded_tokens)
            
            print(f"📝 Original:  {transcript}")
            print(f"🎯 Predicted: {predicted_text}")
            logger.info(f"Original: {transcript}")
            logger.info(f"Predicted: {predicted_text}")
    
    print("\n✅ Test completed successfully!")

if __name__ == '__main__':
    main()

