"""
Overfit test script - Test model's ability to memorize a few samples.
This is a sanity check to verify the model architecture is correct.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from models.asr_base import ASRModel
from utils.manifest_loader import load_merged_dataset


class OverfitDataset(Dataset):
    """Dataset that repeats a few samples many times for overfitting test."""

    def __init__(self, rows: pd.DataFrame, config: dict, repeats_per_sample: int = 100):
        self.rows = rows
        self.repeats_per_sample = repeats_per_sample
        self.total_samples = len(rows) * repeats_per_sample

        self.audio_processor = AudioProcessor(
            sample_rate=config.get("sample_rate", 16000),
            n_mels=config.get("n_mels", 80),
            n_fft=config.get("n_fft", 400),
            hop_length=config.get("hop_length", 160),
            win_length=config.get("win_length", 400),
        )
        self.normalizer = BilingualTextNormalizer()
        self.tokenizer = Tokenizer()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Map index to actual row
        row_idx = idx % len(self.rows)
        row = self.rows.iloc[row_idx]
        
        file_path = row["file_path"]
        transcript = row["transcript"]
        language = row.get("language", "vi")

        # Load and process audio
        audio, sr = self.audio_processor.load_audio(file_path)
        audio = self.audio_processor.trim_silence(audio)
        mel = self.audio_processor.extract_mel_spectrogram(audio).T  # (T, F)

        # Normalize + tokenize text
        text_norm = self.normalizer.normalize(transcript, lang=language)
        tokens = self.tokenizer.encode(text_norm)

        language_id = 0 if language == "vi" else 1

        return {
            "audio_features": torch.from_numpy(mel).float(),
            "audio_length": mel.shape[0],
            "text_tokens": torch.tensor(tokens, dtype=torch.long),
            "text_length": len(tokens),
            "language_id": torch.tensor(language_id, dtype=torch.long),
            "text": text_norm,
            "file_path": file_path,
        }


def collate_fn(batch):
    """Collate function for overfit test."""
    B = len(batch)
    max_T = max(b["audio_length"] for b in batch)
    max_L = max(b["text_length"] for b in batch)
    F = batch[0]["audio_features"].shape[1]

    feats = torch.zeros(B, max_T, F, dtype=torch.float32)
    feat_lens = torch.zeros(B, dtype=torch.long)
    texts = torch.zeros(B, max_L, dtype=torch.long)
    text_lens = torch.zeros(B, dtype=torch.long)
    lang_ids = torch.zeros(B, dtype=torch.long)
    texts_raw = []
    file_paths = []

    for i, b in enumerate(batch):
        t = b["audio_length"]
        l = b["text_length"]
        feats[i, :t] = b["audio_features"]
        feat_lens[i] = t
        texts[i, :l] = b["text_tokens"]
        text_lens[i] = l
        lang_ids[i] = b["language_id"]
        texts_raw.append(b["text"])
        file_paths.append(b["file_path"])

    return {
        "audio_features": feats,
        "audio_lengths": feat_lens,
        "text_tokens": texts,
        "text_lengths": text_lens,
        "language_ids": lang_ids,
        "texts": texts_raw,
        "file_paths": file_paths,
    }


def ctc_greedy_decode(logits, lengths, blank_id=0):
    """Simple CTC greedy decoding."""
    batch_size = logits.size(0)
    predictions = []
    
    for b in range(batch_size):
        seq_len = lengths[b].item()
        pred_ids = logits[b, :seq_len].argmax(dim=-1).tolist()
        
        # CTC collapse: remove repeats and blanks
        collapsed = []
        prev = None
        for t in pred_ids:
            if t == prev:
                continue
            if t == blank_id:
                prev = t
                continue
            collapsed.append(t)
            prev = t
        
        predictions.append(collapsed)
    
    return predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config_path = "configs/overfit_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 80)
    print("OVERFIT TEST - Model Sanity Check")
    print("=" * 80)

    # Load a few samples from dataset
    df = load_merged_dataset(
        split="train",
        dataset_root=config.get("dataset_root", "data/processed/merged_dataset"),
        language=None,
    )
    
    if len(df) == 0:
        raise RuntimeError("No samples found in dataset")

    # Select 3-5 samples for overfitting
    num_samples = min(5, len(df))
    test_samples = df.sample(num_samples, random_state=42)
    
    print(f"\nSelected {num_samples} samples for overfitting:")
    for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
        print(f"  {idx}. {row.get('file_path', 'N/A')}")
        print(f"     Transcript: {row['transcript'][:60]}...")
        print(f"     Language: {row.get('language', 'N/A')}")

    # Create dataset with repeats
    repeats_per_sample = config.get("repeats_per_sample", 100)
    dataset = OverfitDataset(test_samples, config, repeats_per_sample=repeats_per_sample)
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("num_workers", 0),
    )

    print(f"\nDataset: {len(dataset)} samples ({num_samples} unique × {repeats_per_sample} repeats)")

    # Build model
    vocab_size = len(Tokenizer())  # Get actual vocab size
    model = ASRModel(
        input_dim=config.get("n_mels", 80),
        vocab_size=vocab_size,
        d_model=config.get("d_model", 128),
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_heads=config.get("num_heads", 4),
        d_ff=config.get("d_ff", 512),
        dropout=config.get("dropout", 0.0),
        use_gradient_checkpointing=False,  # Disable for overfit test
    ).to(device)

    print(f"\nModel parameters: {model.get_num_params():,}")
    print(f"Model architecture:")
    print(f"  d_model: {config.get('d_model', 128)}")
    print(f"  num_layers: {config.get('num_encoder_layers', 2)}")
    print(f"  num_heads: {config.get('num_heads', 4)}")
    print(f"  d_ff: {config.get('d_ff', 512)}")

    # Loss and optimizer
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0)
    )

    tokenizer = Tokenizer()

    # Training loop
    num_epochs = config.get("num_epochs", 20)
    print(f"\n{'=' * 80}")
    print(f"Training for {num_epochs} epochs...")
    print(f"{'=' * 80}\n")

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        
        for batch in loader:
            feats = batch["audio_features"].to(device)
            feat_lens = batch["audio_lengths"].to(device)
            texts = batch["text_tokens"].to(device)
            text_lens = batch["text_lengths"].to(device)
            lang_ids = batch["language_ids"].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits, out_lens = model(feats, feat_lens, language_ids=lang_ids)

            # CTC loss expects (T, B, V)
            logits_t = logits.transpose(0, 1)  # (T, B, V)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            
            loss = ctc_loss_fn(log_probs, texts, out_lens, text_lens)
            
            loss.backward()
            
            # Gradient clipping
            grad_clip = config.get("grad_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        best_loss = min(best_loss, avg_loss)
        
        print(f"Epoch {epoch:03d}/{num_epochs} - Loss: {avg_loss:.4f} (best: {best_loss:.4f})")

    # Evaluation - Decode predictions
    print(f"\n{'=' * 80}")
    print("Evaluation - Decoding predictions")
    print(f"{'=' * 80}\n")
    
    model.eval()
    with torch.no_grad():
        # Get first batch
        for batch in loader:
            feats = batch["audio_features"].to(device)
            feat_lens = batch["audio_lengths"].to(device)
            lang_ids = batch["language_ids"].to(device)
            texts = batch["text_tokens"]
            text_lens = batch["text_lengths"]
            
            logits, out_lens = model(feats, feat_lens, language_ids=lang_ids)
            
            # Decode predictions
            pred_token_ids = ctc_greedy_decode(logits, out_lens, blank_id=0)
            
            # Print results
            for i in range(len(batch["texts"])):
                gt_text = batch["texts"][i]
                gt_tokens = texts[i, :text_lens[i]].tolist()
                pred_tokens = pred_token_ids[i]
                
                pred_text = tokenizer.decode(pred_tokens)
                
                # Calculate accuracy
                gt_str = tokenizer.decode(gt_tokens)
                match = (pred_text == gt_str)
                
                print(f"Sample {i+1}:")
                print(f"  File: {batch['file_paths'][i]}")
                print(f"  GT:   {gt_text}")
                print(f"  Pred: {pred_text}")
                print(f"  Match: {'✅' if match else '❌'}")
                print()
            
            break  # Only evaluate first batch

    print(f"{'=' * 80}")
    print("Overfit test completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

