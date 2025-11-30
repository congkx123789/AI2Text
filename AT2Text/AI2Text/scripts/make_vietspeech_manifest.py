#!/usr/bin/env python3
"""
Create a CSV manifest for the VietSpeech dataset stored locally at
/home/alida/datasets/VietSpeech.

Output CSV format (compatible with existing manifests):
    file_path,transcript,split,speaker_id

For VietSpeech the audio is stored inside parquet shards as bytes,
so `file_path` will reference the parquet file and row index in the form:

    /home/alida/datasets/VietSpeech/data/data/train-00000-of-00027.parquet:12345

Your dataloader can then open the parquet and fetch that row to get audio+text.
"""

import csv
import os
from pathlib import Path

import pyarrow.parquet as pq


BASE_DIR = Path("/home/alida/datasets/VietSpeech/data/data")
OUTPUT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "manifests"
    / "vietspeech_manifest.csv"
)


def main() -> None:
    parquet_files = sorted(
        [p for p in BASE_DIR.glob("train-*.parquet") if p.is_file()]
    )
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {BASE_DIR}")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "transcript", "split", "speaker_id"])

        for parquet_path in parquet_files:
            print(f"Reading (streaming) {parquet_path} ...")
            pf = pq.ParquetFile(parquet_path)

            if "transcription" not in pf.schema.names:
                raise SystemExit(
                    f"'transcription' column not found in {parquet_path}"
                )

            global_row_idx = 0
            # Iterate row-groups to keep memory low
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["transcription"])
                col = table["transcription"]

                for local_idx, value in enumerate(col):
                    transcript = str(value.as_py()).strip()
                    ref_path = f"{parquet_path}:{global_row_idx + local_idx}"
                    writer.writerow(
                        [ref_path, transcript, "train", "vietspeech"]
                    )
                    total_rows += 1

                global_row_idx += len(col)

    print(f"✅ Wrote {total_rows} rows to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()




