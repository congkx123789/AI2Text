import pandas as pd
from pathlib import Path


def sort_manifest_by_text_length(manifest_path: str, output_path: str):
    """
    Sort a manifest.csv file by transcript length (short → long).
    
    This is a lightweight alternative to bucketing that helps reduce
    padding waste, especially for the first epoch.
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Đang đọc manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    if "transcript" not in df.columns:
        raise ValueError("Manifest must contain a 'transcript' column")

    # Approximate sequence length by transcript length
    df["length"] = df["transcript"].astype(str).str.len()

    # Sort from short to long
    df_sorted = df.sort_values(by="length").reset_index(drop=True)

    # Drop helper column before saving
    df_sorted = df_sorted.drop(columns=["length"])

    print(f"💾 Lưu manifest đã sắp xếp tại: {output_path}")
    df_sorted.to_csv(output_path, index=False)
    print("✅ Đã sắp xếp manifest. Hãy dùng file 'manifest_sorted.csv' để train!")


if __name__ == "__main__":
    default_manifest = "data/processed/merged_dataset/train/manifest.csv"
    default_output = "data/processed/merged_dataset/train/manifest_sorted.csv"
    sort_manifest_by_text_length(default_manifest, default_output)


