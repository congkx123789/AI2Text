import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml

from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from models.asr_with_timestamps import ASRModelWithTimestamps
from utils.manifest_loader import load_merged_dataset


class OneSampleDataset(Dataset):
    """Tiny dataset that repeats a single (audio, text) pair many times."""

    def __init__(self, row: pd.Series, config: dict, repeats: int = 200):
        self.row = row
        self.repeats = repeats

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
        return self.repeats

    def __getitem__(self, idx):
        row = self.row
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
        }


def collate_fn(batch):
    """Simple collate for the one-sample overfit test."""
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

    for i, b in enumerate(batch):
        t = b["audio_length"]
        l = b["text_length"]
        feats[i, :t] = b["audio_features"]
        feat_lens[i] = t
        texts[i, :l] = b["text_tokens"]
        text_lens[i] = l
        lang_ids[i] = b["language_id"]
        texts_raw.append(b["text"])

    return {
        "audio_features": feats,
        "audio_lengths": feat_lens,
        "text_tokens": texts,
        "text_lengths": text_lens,
        "language_ids": lang_ids,
        "texts": texts_raw,
    }


def ctc_greedy_collapse(token_ids, blank_id: int = 0):
    """
    Simple CTC greedy post-processing:
    - collapse repeated tokens
    - remove blanks
    """
    collapsed = []
    prev = None
    for t in token_ids:
        # skip exact repeats
        if t == prev:
            continue
        # skip blanks
        if t == blank_id:
            prev = t
            continue
        collapsed.append(t)
        prev = t
    return collapsed


def main():
    # Use GPU if available for speed, but keep model small to avoid OOM.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load train split from merged_dataset via helper (builds file_path, language, etc.)
    df = load_merged_dataset(
        split="train",
        dataset_root="data/processed/merged_dataset",
        language=None,
    )
    if len(df) == 0:
        raise RuntimeError("merged_dataset/train manifest has no valid samples")

    # Pick a random sample
    row = df.sample(1).iloc[0]
    print("=== OVERFIT ONE RANDOM SAMPLE (FAST DEBUG) ===")
    print("file_path :", row["file_path"])
    print("transcript:", row["transcript"])
    print("language  :", row.get("language", "N/A"))

    # Fewer repeats for faster debug
    dataset = OneSampleDataset(row, config, repeats=50)
    loader = DataLoader(
        dataset,
        batch_size=4,  # small batch for speed
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Build model
    vocab_size = config["vocab_size"]
    # Smaller debug model for fast overfitting
    debug_d_model = 384
    debug_num_layers = 6
    debug_num_heads = 6
    debug_d_ff = 1536

    model = ASRModelWithTimestamps(
        input_dim=config.get("n_mels", 80),
        vocab_size=vocab_size,
        d_model=debug_d_model,
        num_encoder_layers=debug_num_layers,
        num_heads=debug_num_heads,
        d_ff=debug_d_ff,
        dropout=config.get("dropout", 0.1),
        predict_timestamps=False,  # only care about CTC here
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tokenizer = Tokenizer()

    # Fewer epochs for quicker convergence
    epochs = 10
    for epoch in range(1, epochs + 1):
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
            logits, out_lens, _ = model(
                feats,
                feat_lens,
                return_timestamps=False,
                language_ids=lang_ids,
            )

            # CTC expects (T, B, V)
            logits_t = logits.transpose(0, 1)
            log_probs = torch.log_softmax(logits_t, dim=-1)
            loss = ctc_loss_fn(log_probs, texts, out_lens, text_lens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch:03d} - CTC loss: {avg_loss:.4f}")

    # Decode once
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feats = batch["audio_features"].to(device)
            feat_lens = batch["audio_lengths"].to(device)
            lang_ids = batch["language_ids"].to(device)
            logits, out_lens, _ = model(
                feats,
                feat_lens,
                return_timestamps=False,
                language_ids=lang_ids,
            )
            # Raw argmax sequence
            pred_ids_raw = logits.argmax(dim=-1)[0, : out_lens[0]].tolist()
            print("PRED TOKEN IDS (raw):", pred_ids_raw)

            # CTC greedy collapse (remove repeats + blanks)
            pred_ids = ctc_greedy_collapse(pred_ids_raw, blank_id=0)
            print("PRED TOKEN IDS (collapsed):", pred_ids)

            pred_text = tokenizer.decode(pred_ids)
            print("PRED TEXT (Clean):", pred_text)
            print("GT TEXT  :", batch["texts"][0])
            break


if __name__ == "__main__":
    main()


