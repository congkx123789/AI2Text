import json
from pathlib import Path
from typing import List

from utils.manifest_loader import load_merged_dataset
from preprocessing.bpe_tokenizer import BPETokenizer


def collect_texts_from_merged_dataset(
    dataset_root: str = "data/processed/full_merged_dataset",
    split: str = "train",
    languages: List[str] | None = None,
) -> list[str]:
    """
    Load transcripts from merged_dataset to train BPE vocab.

    - Dùng chung pipeline clean transcript & language tag có sẵn.
    - Có thể lọc theo language: ["vi"], ["en"], hoặc ["vi", "en"].
    """
    df = load_merged_dataset(split=split, dataset_root=dataset_root)

    if languages:
        df = df[df["language"].isin(languages)].reset_index(drop=True)

    if "transcript" not in df.columns:
        raise ValueError("Manifest must contain a 'transcript' column")

    texts = df["transcript"].astype(str).tolist()
    print(f"✅ Collected {len(texts):,} transcripts from merged_dataset ({split})")
    if languages:
        print(f"   Languages: {', '.join(languages)}")
    return texts


def build_bpe_vocab(
    output_path: str = "models/bilingual_bpe_2k.json",
    dataset_root: str = "data/processed/full_merged_dataset",
    split: str = "train",
    vocab_size: int = 2000,
    min_frequency: int = 2,
    languages: list[str] | None = None,
):
    """
    Build a BPE vocab from merged_dataset for bilingual (Vi + En) ASR.

    Args:
        output_path: Where to save the BPE JSON file.
        dataset_root: Root of merged_dataset.
        split: Which split to use for vocab (usually 'train').
        vocab_size: Target vocabulary size (recommend ~2000 for 25M model).
        min_frequency: Minimum frequency for subwords.
        languages: Optional list of language codes to include, e.g. ["vi", "en"].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load texts
    texts = collect_texts_from_merged_dataset(
        dataset_root=dataset_root,
        split=split,
        languages=languages,
    )

    # 2) Train BPE tokenizer
    tokenizer = BPETokenizer()
    print(f"🚀 Training BPE tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency} ...")
    tokenizer.train(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    # 3) Save to JSON (reusing BPETokenizer.save for consistency)
    tokenizer.save(str(output_path))
    print(f"✅ Saved BPE vocab ({len(tokenizer)} tokens) to {output_path}")


if __name__ == "__main__":
    # Default: bilingual Vi+En, 2k vocab, train split of merged_dataset
    build_bpe_vocab(
        output_path="models/bilingual_bpe_2k.json",
        dataset_root="data/processed/full_merged_dataset",
        split="train",
        vocab_size=2000,
        min_frequency=2,
        languages=["vi", "en"],
    )


