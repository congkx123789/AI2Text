import json
from pathlib import Path

import pandas as pd

from utils.manifest_loader import load_manifest_data


def build_char_vocab(manifest_path: str, output_path: str):
    """
    Build a simple character-level vocabulary from a manifest.csv file.
    
    The manifest is loaded via load_manifest_data() so transcripts are:
    - cleaned of language tags (<|vi|>, <|en|>)
    - filtered to valid samples
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading manifest from: {manifest_path}")
    df = load_manifest_data(str(manifest_path))

    if 'transcript' not in df.columns:
        raise ValueError("Manifest must contain a 'transcript' column")

    # Concatenate all transcripts into one big string
    all_text = " ".join(df['transcript'].astype(str).tolist())

    # Unique characters present in the dataset
    unique_chars = set(all_text)

    # Base vocab:
    #  0: <pad>  (also used as CTC blank in our Tokenizer by default)
    #  1: <s>
    #  2: </s>
    #  3: <unk>
    #  4: '|'  (used as space token in some decoders)
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }

    idx = 5
    for char in sorted(unique_chars):
        # Skip normal space - we can optionally map it to '|' at encode time
        if char == " ":
            continue
        if char in vocab:
            continue
        vocab[char] = idx
        idx += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo vocab mới với {len(vocab)} tokens tại {output_path}")


if __name__ == "__main__":
    default_manifest = "data/processed/merged_dataset/train/manifest.csv"
    default_output = "models/vocab_char_level.json"
    build_char_vocab(default_manifest, default_output)


