# Data Directory

This directory contains audio data for ASR training.

## 📁 Structure

- **`raw/`**: Raw audio files (WAV, MP3, FLAC, OGG)
  - `train/`: Training audio files
  - `val/`: Validation audio files
  - `test/`: Test audio files

- **`processed/`**: Preprocessed audio files (mel spectrograms, features)

- **`external/`**: External datasets downloaded from public sources

## 📊 Data Format Requirements

### Audio Files
- **Format**: WAV (preferred), MP3, FLAC, or OGG
- **Sample rate**: 16kHz or higher (16kHz recommended)
- **Channels**: Mono (1 channel) preferred
- **Duration**: 0.5s to 30s (recommended)
- **Quality**: At least 16kHz sample rate for good results

### CSV File Format
Create a CSV file with these columns:
- `file_path`: Path to audio file (relative or absolute)
- `transcript`: Vietnamese transcription text
- `split` (optional): 'train', 'val', or 'test'
- `speaker_id` (optional): Speaker identifier

Example:
```csv
file_path,transcript,split,speaker_id
data/raw/audio1.wav,xin chào việt nam,train,speaker_01
data/raw/audio2.wav,tôi là sinh viên,train,speaker_02
```

## 🚀 Import Data

1. Prepare your CSV file with audio paths and transcripts
2. Run the data import script:
   ```bash
   python scripts/prepare_data.py --csv your_data.csv --audio_base data/raw --auto_split
   ```

See `DATA_PREPARATION_GUIDE.md` in the project root for detailed instructions.

## 📚 Sample Data Sources

- **Common Voice Vietnamese**: https://commonvoice.mozilla.org/en/datasets
- **OpenSLR**: https://www.openslr.org/47 (100 hours Vietnamese)
- **Vivos Dataset**: https://ailab.hcmus.edu.vn/vivos

Use `scripts/download_sample_data.py` to get information about downloading sample datasets.

