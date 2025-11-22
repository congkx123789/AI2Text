"""
Example training script for AI2Text ASR model.

This is a simplified example showing how to train a model.
For production use, see training/train.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from training.dataset import ASRDataset, collate_fn
from database.db_utils import ASRDatabase


def simple_train_example():
    """Simple training example."""
    
    print("=" * 70)
    print("AI2Text - Simple Training Example")
    print("=" * 70)
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 0.0001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    print(f"  Device: {config['device']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    # Initialize database
    print("\nInitializing database...")
    db = ASRDatabase('database/asr_training.db')
    
    # Get training data
    print("Loading training data...")
    train_files = db.get_audio_files_by_split('train', 'v1')
    
    if len(train_files) == 0:
        print("\nERROR: No training data found!")
        print("\nPlease add training data first:")
        print("  1. Create metadata.csv with your audio files")
        print("  2. Run: python scripts/prepare_data.py")
        print("  3. See TRAINING_GUIDE.md for details")
        return
    
    print(f"  Found {len(train_files)} training files")
    
    # Create DataFrame
    data_list = []
    for file_data in train_files[:100]:  # Limit for example
        transcript = db.get_transcript_by_audio_id(file_data['id'])
        if transcript:
            data_list.append({
                'file_path': file_data['file_path'],
                'transcript': transcript['transcript']
            })
    
    if len(data_list) == 0:
        print("\nERROR: No transcripts found!")
        return
    
    df = pd.DataFrame(data_list)
    print(f"  Prepared {len(df)} samples")
    
    # Initialize components
    print("\nInitializing model components...")
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    tokenizer = Tokenizer()
    normalizer = VietnameseTextNormalizer()
    
    # Create dataset
    print("Creating dataset...")
    dataset = ASRDataset(
        data_df=df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        apply_augmentation=False
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    # Initialize model
    print("Initializing model...")
    model = ASRModel(
        input_dim=80,
        vocab_size=len(tokenizer),
        d_model=128,  # Small for example
        num_encoder_layers=2,
        num_heads=2,
        d_ff=256,
        dropout=0.1
    )
    model = model.to(config['device'])
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and loss
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            # Move to device
            audio_features = batch['audio_features'].to(config['device'])
            audio_lengths = batch['audio_lengths'].to(config['device'])
            text_tokens = batch['text_tokens'].to(config['device'])
            text_lengths = batch['text_lengths'].to(config['device'])
            
            # Forward pass
            logits, output_lengths = model(audio_features, audio_lengths)
            
            # Reshape for CTC loss
            logits = logits.transpose(0, 1)  # (T, N, C)
            logits = logits.log_softmax(dim=2)
            
            # Compute loss
            loss = ctc_loss(
                logits,
                text_tokens,
                output_lengths,
                text_lengths
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'checkpoints/example_epoch_{epoch+1}.pt'
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: checkpoints/")
    print("\nNext steps:")
    print("  1. Evaluate: python training/evaluate.py --model_path checkpoints/example_epoch_5.pt")
    print("  2. Use API: uvicorn api.app:app --reload")
    print("  3. See TRAINING_GUIDE.md for advanced training")


if __name__ == "__main__":
    try:
        simple_train_example()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("\nFor help, see:")
        print("  - TRAINING_GUIDE.md")
        print("  - QUICK_TRAIN.md")
        import traceback
        traceback.print_exc()

