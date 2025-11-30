#!/usr/bin/env python3
"""
Create a CSV manifest for the GigaSpeech M subset using the Hugging Face datasets API.

This ONLY exports text metadata (no audio arrays), so it should be light on RAM
and network usage once your account has access to `speechcolab/gigaspeech`.

Output CSV format:
    segment_id,transcript,split
"""

import csv
from pathlib import Path

from datasets import load_dataset


# Where to save the manifest (next to your other manifests)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "gigaspeech_m_manifest.csv"


def main() -> None:
    print("=" * 70)
    print("📚 Exporting GigaSpeech M subset transcriptions")
    print("=" * 70)
    print("⚠️  Requires that you have access to `speechcolab/gigaspeech`")
    print("    and are logged in with `hf auth login`.")
    print("")

    # Use streaming to avoid loading everything into RAM
    print("🔄 Loading dataset (streaming=True) ...")
    ds = load_dataset(
        "speechcolab/gigaspeech",
        "m",
        split="train",
        streaming=True,
    )

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "transcript", "split"])

        for sample in ds:
            seg_id = str(sample.get("segment_id") or "").strip()
            text = str(sample.get("text") or "").strip()
            if not seg_id or not text:
                continue

            writer.writerow([seg_id, text, "train"])
            total += 1

            if total % 10000 == 0:
                print(f"  ✅ written {total} rows so far ...")

    print("")
    print(f"✅ Done. Wrote {total} rows to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()






























