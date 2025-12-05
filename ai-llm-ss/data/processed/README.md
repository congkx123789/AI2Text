Data/processed layout (merged + full_merged_dataset)
====================================================

This folder holds ready-to-train datasets. Two variants:

- merged_dataset/: lightly processed (includes audio_sliced in train).
- full_merged_dataset/: full audio copies, plus backup manifests.

Shared split structure
----------------------
- train/
- val/
- test/
Each split can contain:
- audio/: wav files (abs/rel paths referenced in manifest.csv).
- audio_sliced/: (merged_dataset/train only) sliced segments.
- manifest.csv: main table.
- manifest_sorted.csv, manifest_sliced.csv: alternates (train only, merged_dataset).
- manifest.csv.backup: backups (full_merged_dataset).
- timestamps.json: per-utterance timing metadata.

manifest.csv schema
-------------------
Columns:
- id: unique utterance id (matches audio filename without extension).
- transcript: text transcript. Language tagged with <|vi|> or <|en|>.
- audio_path: relative path to wav file (e.g., audio/007_000000980.wav).
- duration: float seconds.
- words_json: JSON string of word-level segments:
  - word: token text
  - start, end: seconds

Example row (CSV):
id,transcript,audio_path,duration,words_json
007_000000980,"<|vi|> nhiều khi khó nén dồn được cảm xúc...",audio/007_000000980.wav,6.145,"[{""word"":""nhiều"",""start"":0.0,""end"":0.201}, ...]"

timestamps.json schema
----------------------
Top-level keys: "{audio_id}.wav".
Value:
- duration: float seconds
- text: full transcript (with language tag)
- segments: list of {word, start, end, score} entries

Example entry (JSON):
{
  "007_000000980.wav": {
    "duration": 6.125,
    "text": "<|vi|> nhiều khi khó nén dồn được cảm xúc...",
    "segments": [
      { "word": "nhiều", "start": 0.0, "end": 0.201, "score": 0.01 },
      { "word": "khi",   "start": 0.221, "end": 0.301, "score": 0.011 },
      ...
    ]
  }
}

Usage hints
-----------
- Training expects a manifest with audio_path paths relative to its directory.
- When using merged_dataset, you can choose manifest.csv (full) or manifest_sliced.csv (if you want sliced audio_sliced/ files); manifest_sorted.csv is just duration-sorted.
- Language mixing is supported via tags in transcript (<|vi|>, <|en|>); tokenizer must include all characters present.
# Processed Merged Dataset

This directory contains the merged dataset created from VietSpeech and LibriSpeech alignments.

## Dataset Structure

```
data/processed/merged_dataset/
├── train/
│   ├── audio/                    # Merged audio files (randomly shuffled)
│   ├── manifest.csv              # Combined manifest with metadata
│   └── timestamps.json           # Combined timestamps
├── val/
│   ├── audio/
│   ├── manifest.csv
│   └── timestamps.json
└── test/
    ├── audio/
    ├── manifest.csv
    └── timestamps.json
```

## Dataset Composition

### Train Split
- **VietSpeech**: ~50 hours
- **LibriSpeech**: ~25 hours
- **Total**: ~75 hours

### Val Split
- **VietSpeech**: ~6.25 hours (1/8 of train VietSpeech)
- **LibriSpeech**: ~3.125 hours (1/8 of train LibriSpeech)
- **Total**: ~9.375 hours (1/8 of train total)

### Test Split
- **VietSpeech**: ~6.25 hours (1/8 of train VietSpeech)
- **LibriSpeech**: ~3.125 hours (1/8 of train LibriSpeech)
- **Total**: ~9.375 hours (1/8 of train total)

## Creating the Dataset

To create this dataset, run:

```bash
python3 scripts/create_merged_dataset.py \
  --train-vs-hours 50 \
  --train-ls-hours 25 \
  --output-dir data/processed/merged_dataset \
  --seed 42
```

The script will:
1. Sample files from VietSpeech and LibriSpeech train splits
2. Randomly merge and shuffle the files
3. Create train/val/test splits with the specified proportions
4. Copy audio files to the output directory
5. Generate manifest.csv and timestamps.json files

## File Formats

### manifest.csv
Columns:
- `id`: Unique utterance identifier
- `transcript`: Text transcription (with language tags `<|vi|>` or `<|en
- `audio_path`: Relative path to audio file (e.g., `audio/filename.wav`)
- `words_json`: JSON string with word-level timestamps
- `sex`: Speaker gender ('U' for VietSpeech, 'M'/'F' for LibriSpeech)
- `subset`: Dataset source ('vietspeech' or 'librispeech')

### timestamps.json
Dictionary format:
```json
{
  "filename.wav": {
    "duration": 5.2,
    "text": "transcript text",
    "segments": [
      {"word": "word1", "start": 0.0, "end": 0.5},
      ...
    ]
  }
}
```

## Usage

The merged dataset can be used for bilingual ASR training:

```python
import pandas as pd
import json

# Load train manifest
train_manifest = pd.read_csv('data/processed/merged_dataset/train/manifest.csv')

# Load timestamps
with open('data/processed/merged_dataset/train/timestamps.json') as f:
    timestamps = json.load(f)
```

## Notes

- Files are randomly shuffled within each split
- No overlap between train/val/test splits
- Audio files are copied (not symlinked) to ensure portability
- The random seed (default: 42) ensures reproducibility

