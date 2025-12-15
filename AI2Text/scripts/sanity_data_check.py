#!/usr/bin/env python3
"""
Quick sanity check for data/label alignment.

What it does:
- Loads a few samples from manifest (train/val).
- Prints file path, language, raw transcript (cleaned tags), and tokenizer round-trip.
- Builds a tiny DataLoader with the same dataset/normalizer/tokenizer as training
  to verify that the collate pipeline produces aligned tensors.
"""

import argparse
from pathlib import Path
import sys
import yaml
import torch

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.manifest_loader import load_merged_dataset
from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import BilingualTextNormalizer
from training.dataset import ASRDataset, collate_fn


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_sample_rows(df, num_samples: int):
    print(f"\n📑 Sample rows from manifest (n={num_samples}):")
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)
    for i, row in sample_df.reset_index(drop=True).iterrows():
        transcript_preview = str(row["transcript"])[:160].replace("\n", " ")
        print(
            f"[row {i}] lang={row.get('language','?')} "
            f"dur={row.get('duration_seconds','?')} "
            f"path={row['file_path']}"
        )
        print(f"       text: {transcript_preview}")


def check_dataloader(df, tokenizer, audio_processor, batch_size: int = 2):
    # Use a tiny subset to avoid heavy I/O
    subset_df = df.sample(min(batch_size * 2, len(df)), random_state=0).reset_index(drop=True)
    dataset = ASRDataset(
        data_df=subset_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=BilingualTextNormalizer(),
        augmenter=None,
        apply_augmentation=False,
        cache_in_ram=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    transcripts = batch["transcripts"]
    lang_ids = batch["language_ids"].tolist()
    text_tokens = batch["text_tokens"]

    print("\n🧪 DataLoader sanity (first batch):")
    print(f"  batch shape audio={batch['audio_features'].shape}, text={text_tokens.shape}")
    print(f"  language_ids: {lang_ids}")

    # Decode tokens back to text to ensure alignment
    for i in range(len(transcripts)):
        toks = text_tokens[i].tolist()
        # Remove padding (0) and stop at EOS if present
        trimmed = []
        for t in toks:
            if t == 0:  # pad
                break
            trimmed.append(t)
        decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
        print(f"  sample {i}: lang_id={lang_ids[i]}")
        print(f"    transcript (from dataset): {transcripts[i][:120]}")
        print(f"    decoded tokens           : {decoded[:120]}")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check data alignment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=6, help="Rows to preview from manifest")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    dataset_root = cfg.get("dataset_root", "data/processed/full_merged_dataset")

    print(f"Loading manifest split={args.split} from {dataset_root} ...")
    df = load_merged_dataset(split=args.split, dataset_root=dataset_root)
    print(f"✅ Loaded {len(df):,} samples")

    # Quick manifest inspection
    print_sample_rows(df, args.num_samples)

    # Instantiate tokenizer/audio processor
    tokenizer = SentencePieceTokenizer(cfg.get("bpe_vocab_path", "models/tokenizer_vi_en_3500.model"))
    audio_processor = AudioProcessor(
        sample_rate=cfg.get("sample_rate", 16000),
        n_mels=cfg.get("n_mels", 80),
        n_fft=cfg.get("n_fft", 400),
        hop_length=cfg.get("hop_length", 160),
        win_length=cfg.get("win_length", 400),
    )

    # Check dataloader alignment
    check_dataloader(df, tokenizer, audio_processor, batch_size=2)


if __name__ == "__main__":
    main()

