#!/usr/bin/env python3
"""
Sequential training script: Train English first, then Vietnamese.

This script automates the sequential training process:
1. Train on English data
2. Resume from English checkpoint
3. Continue training on Vietnamese data

Usage:
    python scripts/train_sequential.py \
        --english-config configs/english_training.yaml \
        --vietnamese-config configs/vietnamese_training.yaml \
        --english-epochs 30 \
        --vietnamese-epochs 20
"""

import argparse
import yaml
import sys
from pathlib import Path
import subprocess
import time

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase


def check_data_availability(db: ASRDatabase, split_version: str, language: str):
    """Check if data is available for the specified language."""
    train_df = db.get_split_data('train', split_version, language=language)
    val_df = db.get_split_data('val', split_version, language=language)
    
    print(f"\n{'='*60}")
    print(f"Data Check for Language: {language.upper()}")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    if len(train_df) == 0:
        print(f"⚠️  WARNING: No training data found for language '{language}'")
        return False
    
    if len(val_df) == 0:
        print(f"⚠️  WARNING: No validation data found for language '{language}'")
        return False
    
    print(f"✅ Data available for {language}")
    return True


def run_training(config_path: str, language: str, resume: str = None, epochs: int = None):
    """Run training with specified parameters."""
    print(f"\n{'='*60}")
    print(f"Starting Training: {language.upper()}")
    print(f"{'='*60}")
    
    # Load config to check epochs
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs if specified
    if epochs is not None:
        config['num_epochs'] = epochs
        # Save temporary config
        temp_config = Path(config_path).parent / f"temp_{language}_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        config_path = str(temp_config)
    
    # Build command
    cmd = [
        sys.executable,
        "training/train.py",
        "--config", config_path,
        "--language", language
    ]
    
    if resume:
        cmd.extend(["--resume", resume])
        print(f"Resuming from checkpoint: {resume}")
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print(f"Language: {language}")
    print(f"Epochs: {config.get('num_epochs', 'default')}")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Training failed for {language}")
        return None
    
    print(f"\n✅ Training completed for {language} in {elapsed_time/60:.1f} minutes")
    
    # Clean up temp config if created
    if epochs is not None:
        temp_config = Path(config_path)
        if temp_config.exists():
            temp_config.unlink()
    
    return "checkpoints/best_model.pt"


def main():
    parser = argparse.ArgumentParser(
        description='Sequential training: English first, then Vietnamese',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sequential training
  python scripts/train_sequential.py \\
      --english-config configs/english_training.yaml \\
      --vietnamese-config configs/vietnamese_training.yaml

  # Custom epochs
  python scripts/train_sequential.py \\
      --english-config configs/english_training.yaml \\
      --vietnamese-config configs/vietnamese_training.yaml \\
      --english-epochs 30 \\
      --vietnamese-epochs 20

  # Skip English training (if already done)
  python scripts/train_sequential.py \\
      --vietnamese-config configs/vietnamese_training.yaml \\
      --resume checkpoints/best_model.pt \\
      --vietnamese-epochs 20
        """
    )
    
    parser.add_argument('--english-config', type=str, default=None,
                       help='Path to English training config file')
    parser.add_argument('--vietnamese-config', type=str, required=True,
                       help='Path to Vietnamese training config file')
    parser.add_argument('--english-epochs', type=int, default=None,
                       help='Number of epochs for English training (overrides config)')
    parser.add_argument('--vietnamese-epochs', type=int, default=None,
                       help='Number of epochs for Vietnamese training (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume from (skips English training if provided)')
    parser.add_argument('--skip-english', action='store_true',
                       help='Skip English training and go straight to Vietnamese')
    
    args = parser.parse_args()
    
    # Load Vietnamese config to get database path and split version
    with open(args.vietnamese_config, 'r') as f:
        vi_config = yaml.safe_load(f)
    
    db_path = vi_config.get('database_path', 'database/asr_training.db')
    split_version = vi_config.get('split_version', 'v1')
    
    # Initialize database
    db = ASRDatabase(db_path)
    
    print("="*60)
    print("Sequential Training: English → Vietnamese")
    print("="*60)
    
    # Check data availability
    print("\nChecking data availability...")
    en_available = check_data_availability(db, split_version, 'en')
    vi_available = check_data_availability(db, split_version, 'vi')
    
    if not vi_available:
        print("\n❌ ERROR: Vietnamese data not available. Cannot proceed.")
        sys.exit(1)
    
    # Step 1: Train English (if not skipped)
    english_checkpoint = None
    if not args.skip_english and not args.resume and args.english_config:
        if not en_available:
            print("\n⚠️  WARNING: English data not available. Skipping English training.")
        else:
            print("\n" + "="*60)
            print("STEP 1: Training on English Data")
            print("="*60)
            
            english_checkpoint = run_training(
                config_path=args.english_config,
                language='en',
                epochs=args.english_epochs
            )
            
            if english_checkpoint is None:
                print("\n❌ English training failed. Cannot proceed to Vietnamese.")
                sys.exit(1)
    elif args.resume:
        english_checkpoint = args.resume
        print(f"\n✅ Using provided checkpoint: {english_checkpoint}")
    else:
        print("\n⚠️  Skipping English training (no config provided or --skip-english)")
    
    # Step 2: Train Vietnamese
    print("\n" + "="*60)
    print("STEP 2: Training on Vietnamese Data")
    print("="*60)
    
    vietnamese_checkpoint = run_training(
        config_path=args.vietnamese_config,
        language='vi',
        resume=english_checkpoint,
        epochs=args.vietnamese_epochs
    )
    
    if vietnamese_checkpoint is None:
        print("\n❌ Vietnamese training failed.")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("Sequential Training Complete!")
    print("="*60)
    print(f"✅ English training: {'Completed' if english_checkpoint else 'Skipped'}")
    print(f"✅ Vietnamese training: Completed")
    print(f"\nFinal model checkpoint: {vietnamese_checkpoint}")
    print("\nNext steps:")
    print("1. Evaluate the model:")
    print(f"   python training/evaluate.py \\")
    print(f"       --config {args.vietnamese_config} \\")
    print(f"       --checkpoint {vietnamese_checkpoint} \\")
    print(f"       --split test")
    print("\n2. Test on individual files:")
    print(f"   python training/evaluate.py \\")
    print(f"       --config {args.vietnamese_config} \\")
    print(f"       --checkpoint {vietnamese_checkpoint} \\")
    print(f"       --audio path/to/audio.wav")


if __name__ == '__main__':
    main()

