
"""
Download thivux/phoaudiobook dataset from HuggingFace
and extract word-level timestamps using Whisper
Saves both raw dataset and aligned version with timestamps
"""

import os
import json
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk
from transformers import pipeline
from tqdm import tqdm
import sys

# Configuration
DATASET_NAME = "thivux/phoaudiobook"
OUTPUT_DIR = Path("/home/alida/datasets/phoaudiobook")
ALIGNED_OUTPUT_FILE = Path("/home/alida/datasets/phoaudiobook_aligned.jsonl")
SPLIT = "train"

# Whisper configuration
MODEL_ID = "openai/whisper-large-v3"
BATCH_SIZE = 24  # Optimized for RTX 5060 Ti
DTYPE = torch.float16

# HuggingFace cache directory
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def get_hf_cache_path():
    """Get the HuggingFace cache path for this dataset."""
    # Convert dataset name to cache directory format: "org/name" -> "datasets--org--name"
    cache_name = f"datasets--{DATASET_NAME.replace('/', '--')}"
    return HF_CACHE_DIR / cache_name


def check_hf_cache():
    """Check if dataset exists in HuggingFace cache."""
    cache_path = get_hf_cache_path()
    
    if not cache_path.exists():
        return None
    
    print(f"🔍 Checking HuggingFace cache: {cache_path}")
    
    # Check if cache directory has content (blobs, snapshots, etc.)
    if not any(cache_path.iterdir()):
        print("   ⚠️  Cache directory exists but is empty")
        return None
    
    # Check if snapshots directory exists (indicates cached dataset)
    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
        print("   ⚠️  Cache directory exists but no snapshots found")
        return None
    
    # Try to get cache size
    try:
        import shutil
        cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        cache_size_gb = cache_size / (1024**3)
        print(f"   📦 Found cached dataset ({cache_size_gb:.2f} GB)")
        return cache_path
    except Exception as e:
        print(f"   ⚠️  Error checking cache: {e}")
        return None


def show_example_format(num_examples=3):
    """Show example format of word-level timestamps."""
    print("=" * 60)
    print("Example Word-Level Timestamp Format")
    print("=" * 60)
    print()
    
    # Check if file exists and show real examples
    if ALIGNED_OUTPUT_FILE.exists():
        print(f"📂 Found existing file: {ALIGNED_OUTPUT_FILE}")
        print("Showing examples from actual data:")
        print()
        
        with open(ALIGNED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                try:
                    entry = json.loads(line.strip())
                    print(f"Example {i+1}:")
                    print("-" * 60)
                    
                    # Show summary
                    print(f"ID: {entry.get('id', i)}")
                    print(f"Original text: {entry.get('original_text', '')[:80]}...")
                    print(f"Transcription: {entry.get('transcription', '')[:80]}...")
                    
                    timestamps = entry.get('timestamps', [])
                    if timestamps:
                        print(f"Word count: {len(timestamps)}")
                        print("\nFirst 5 words with timestamps:")
                        for j, word in enumerate(timestamps[:5]):
                            end_str = f" - {word['end']:.2f}s" if 'end' in word and word['end'] else ""
                            print(f"  {j+1}. '{word.get('text', '')}' at {word.get('start', 0):.2f}s{end_str}")
                        
                        if len(timestamps) > 5:
                            print(f"  ... and {len(timestamps) - 5} more words")
                    else:
                        print("⚠️  No timestamps found")
                    
                    print()
                except json.JSONDecodeError:
                    print(f"⚠️  Line {i+1}: Invalid JSON")
                    print()
        return
    
    # Show example format if file doesn't exist
    print("Each entry in the JSONL file will have this structure:")
    print()
    
    examples = [
        {
            "id": 0,
            "original_text": "Xin chào, tôi là một hệ thống nhận dạng giọng nói.",
            "transcription": "Xin chào, tôi là một hệ thống nhận dạng giọng nói.",
            "timestamps": [
                {"text": "Xin", "start": 0.5, "end": 0.8},
                {"text": "chào", "start": 0.8, "end": 1.2},
                {"text": ",", "start": 1.2, "end": 1.3},
                {"text": "tôi", "start": 1.5, "end": 1.8},
                {"text": "là", "start": 1.8, "end": 2.0},
                {"text": "một", "start": 2.0, "end": 2.3},
                {"text": "hệ", "start": 2.3, "end": 2.5},
                {"text": "thống", "start": 2.5, "end": 3.0},
                {"text": "nhận", "start": 3.0, "end": 3.4},
                {"text": "dạng", "start": 3.4, "end": 3.8},
                {"text": "giọng", "start": 3.8, "end": 4.2},
                {"text": "nói", "start": 4.2, "end": 4.5},
                {"text": ".", "start": 4.5, "end": 4.6}
            ]
        },
        {
            "id": 1,
            "original_text": "Hôm nay trời đẹp quá!",
            "transcription": "Hôm nay trời đẹp quá!",
            "timestamps": [
                {"text": "Hôm", "start": 0.2, "end": 0.5},
                {"text": "nay", "start": 0.5, "end": 0.8},
                {"text": "trời", "start": 0.8, "end": 1.1},
                {"text": "đẹp", "start": 1.1, "end": 1.4},
                {"text": "quá", "start": 1.4, "end": 1.7},
                {"text": "!", "start": 1.7, "end": 1.8}
            ]
        },
        {
            "id": 2,
            "original_text": "Tôi đang học tiếng Việt.",
            "transcription": "Tôi đang học tiếng Việt.",
            "timestamps": [
                {"text": "Tôi", "start": 0.3, "end": 0.6},
                {"text": "đang", "start": 0.6, "end": 0.9},
                {"text": "học", "start": 0.9, "end": 1.2},
                {"text": "tiếng", "start": 1.2, "end": 1.6},
                {"text": "Việt", "start": 1.6, "end": 2.0},
                {"text": ".", "start": 2.0, "end": 2.1}
            ]
        }
    ]
    
    for i, example_entry in enumerate(examples[:num_examples]):
        print(f"Example {i+1}:")
        print("-" * 60)
        print(json.dumps(example_entry, indent=2, ensure_ascii=False))
        print()
    
    print("=" * 60)
    print("Key Fields:")
    print("=" * 60)
    print("• id: Sample index in the dataset")
    print("• original_text: Original transcript from the dataset")
    print("• transcription: Whisper transcription (may differ slightly)")
    print("• timestamps: List of word-level timestamps")
    print("  - text: The word text")
    print("  - start: Start time in seconds")
    print("  - end: End time in seconds (may be None)")
    print()
    print("=" * 60)
    print("Example Usage:")
    print("=" * 60)
    print("To read and use timestamps:")
    print()
    print("```python")
    print("import json")
    print()
    print("with open('phoaudiobook_aligned.jsonl', 'r') as f:")
    print("    for line in f:")
    print("        entry = json.loads(line)")
    print("        print(f\"Text: {entry['original_text']}\")")
    print("        for word in entry['timestamps']:")
    print("            end = word.get('end', 'N/A')")
    print("            print(f\"  {word['text']}: {word['start']:.2f}s - {end}\")")
    print("```")
    print()


def get_processed_count():
    """Count how many samples have been processed (for resume)."""
    if not ALIGNED_OUTPUT_FILE.exists():
        return 0
    with open(ALIGNED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def check_timestamps_in_data(file_path=None, num_samples=5):
    """Check if the aligned data file contains word-level timestamps."""
    if file_path is None:
        file_path = ALIGNED_OUTPUT_FILE
    
    if not Path(file_path).exists():
        print(f"❌ File does not exist: {file_path}")
        return False
    
    print("=" * 60)
    print("Checking Word-Level Timestamps in Data")
    print("=" * 60)
    print(f"File: {file_path}")
    print()
    
    valid_count = 0
    invalid_count = 0
    error_count = 0
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_count += 1
            if i >= num_samples:
                break
            
            try:
                entry = json.loads(line.strip())
                
                # Check if entry has error
                if "error" in entry:
                    error_count += 1
                    print(f"⚠️  Sample {entry.get('id', i)}: Has error - {entry['error']}")
                    continue
                
                # Check for timestamps
                if "timestamps" not in entry:
                    invalid_count += 1
                    print(f"❌ Sample {entry.get('id', i)}: Missing 'timestamps' field")
                    continue
                
                timestamps = entry["timestamps"]
                
                # Check if timestamps is a list
                if not isinstance(timestamps, list):
                    invalid_count += 1
                    print(f"❌ Sample {entry.get('id', i)}: Timestamps is not a list (type: {type(timestamps)})")
                    continue
                
                # Check if timestamps list is empty
                if len(timestamps) == 0:
                    invalid_count += 1
                    print(f"❌ Sample {entry.get('id', i)}: Timestamps list is empty")
                    continue
                
                # Check format of first timestamp
                first_ts = timestamps[0]
                if isinstance(first_ts, dict):
                    # Check for word-level timestamp format
                    has_word = "word" in first_ts or "text" in first_ts
                    has_time = "start" in first_ts or "timestamp" in first_ts or "timestamp_start" in first_ts
                    
                    if has_word and has_time:
                        valid_count += 1
                        print(f"✅ Sample {entry.get('id', i)}: Has word-level timestamps")
                        print(f"   Format: {list(first_ts.keys())}")
                        print(f"   Example: {first_ts}")
                        print(f"   Total words: {len(timestamps)}")
                    else:
                        invalid_count += 1
                        print(f"⚠️  Sample {entry.get('id', i)}: Timestamp format incomplete")
                        print(f"   Keys: {list(first_ts.keys())}")
                elif isinstance(first_ts, (list, tuple)) and len(first_ts) >= 2:
                    # Alternative format: [word, start, end] or similar
                    valid_count += 1
                    print(f"✅ Sample {entry.get('id', i)}: Has timestamps (list format)")
                    print(f"   Format: {first_ts}")
                    print(f"   Total words: {len(timestamps)}")
                else:
                    invalid_count += 1
                    print(f"⚠️  Sample {entry.get('id', i)}: Unknown timestamp format")
                    print(f"   Type: {type(first_ts)}, Value: {first_ts}")
                
                print()
                
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"❌ Line {i+1}: JSON decode error - {e}")
                print()
            except Exception as e:
                error_count += 1
                print(f"❌ Line {i+1}: Error - {e}")
                print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total samples checked: {min(total_count, num_samples)}")
    print(f"✅ Valid with timestamps: {valid_count}")
    print(f"❌ Invalid/missing timestamps: {invalid_count}")
    print(f"⚠️  Errors: {error_count}")
    print()
    
    if total_count > 0:
        # Check total file count
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"📊 Total samples in file: {total_lines:,}")
        print()
    
    return valid_count > 0


def check_disk_space():
    """Check available disk space."""
    import shutil
    
    stat = shutil.disk_usage(OUTPUT_DIR.parent)
    free_gb = stat.free / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 10:
        print("⚠️  Warning: Low disk space! Dataset might be large.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)
    print()


def download_dataset():
    """Download the dataset and save to local directory."""
    
    print("=" * 60)
    print("Downloading phoaudiobook Dataset with Word-Level Timestamps")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Aligned output: {ALIGNED_OUTPUT_FILE}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directory: {OUTPUT_DIR}")
    print()
    
    # Check authentication
    from huggingface_hub import whoami
    try:
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
        print()
    except Exception:
        print("⚠️  Not authenticated with HuggingFace!")
        print()
        print("This dataset requires authentication. Please:")
        print("1. Accept terms at: https://huggingface.co/datasets/thivux/phoaudiobook")
        print("2. Login: huggingface-cli login")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled. Please authenticate first.")
            sys.exit(0)
        print()
    
    # Check if dataset already downloaded in OUTPUT_DIR
    dataset_loaded = False
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        try:
            print("📂 Found existing dataset in output directory, loading from disk...")
            dataset = load_from_disk(str(OUTPUT_DIR))
            if SPLIT in dataset:
                dataset = dataset[SPLIT]
            print(f"✅ Loaded {len(dataset)} samples from output directory")
            dataset_loaded = True
        except Exception as e:
            print(f"⚠️  Error loading from output directory: {e}")
            print("   Will check cache or download fresh...")
            print()
    
    # Check HuggingFace cache if not loaded from OUTPUT_DIR
    if not dataset_loaded:
        cache_path = check_hf_cache()
        if cache_path:
            print("   ✅ Found dataset in HuggingFace cache!")
            print("   📥 Loading from cache (no download needed)...")
            print()
            try:
                # load_dataset will automatically use cache if available
                dataset = load_dataset(DATASET_NAME, split=SPLIT)
                total_samples = len(dataset)
                print(f"   ✅ Loaded {total_samples:,} samples from cache")
                print()
                
                # Optionally save to OUTPUT_DIR for faster future access
                if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
                    print("💾 Saving dataset to output directory for faster access...")
                    print("   (This may take a while, but will speed up future runs)")
                    print()
                    dataset.save_to_disk(str(OUTPUT_DIR))
                    print(f"✅ Dataset saved to: {OUTPUT_DIR}")
                    print()
                else:
                    print("   ℹ️  Output directory already exists, skipping save")
                    print()
                
                dataset_loaded = True
            except Exception as e:
                print(f"⚠️  Error loading from cache: {e}")
                print("   Will try downloading fresh...")
                print()
    
    # Download if not found in cache or OUTPUT_DIR
    if not dataset_loaded:
        try:
            print("📥 Downloading dataset from HuggingFace...")
            print("   (This may take a while for large datasets)")
            print(f"   Dataset will be cached in: {get_hf_cache_path()}")
            print()
            
            dataset = load_dataset(DATASET_NAME, split=SPLIT)
            total_samples = len(dataset)
            print(f"   ✅ Downloaded {total_samples:,} samples")
            print()
            
            # Save dataset to disk
            print("💾 Saving dataset to output directory...")
            print("   (This may take a while...)")
            dataset.save_to_disk(str(OUTPUT_DIR))
            print(f"✅ Dataset saved to: {OUTPUT_DIR}")
            print()
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
                print(f"❌ Authentication required!")
                print()
                print("Please:")
                print("1. Accept dataset terms: https://huggingface.co/datasets/thivux/phoaudiobook")
                print("2. Login: huggingface-cli login")
                print("3. Run this script again")
                sys.exit(1)
            else:
                print(f"❌ Error downloading dataset: {e}")
                sys.exit(1)
    
    # Now extract word-level timestamps
    print("=" * 60)
    print("Extracting Word-Level Timestamps with Whisper")
    print("=" * 60)
    print()
    
    extract_timestamps(dataset)


def extract_timestamps(dataset):
    """Extract word-level timestamps using Whisper."""
    
    # Check previous progress
    start_index = get_processed_count()
    print(f"📂 Checking progress... Found {start_index} completed samples.")
    print()
    
    if start_index > 0:
        print(f"▶️  Resuming from index {start_index}...")
        print()
    
    # Load Whisper model
    print("🚀 Loading Whisper on GPU...")
    print(f"   Model: {MODEL_ID}")
    print(f"   Batch size: {BATCH_SIZE}")
    print()
    
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            torch_dtype=DTYPE,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        print("✅ Whisper model loaded")
        print()
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        print("   Make sure you have CUDA available and transformers installed")
        sys.exit(1)
    
    # Process dataset
    total_samples = len(dataset)
    batch_buffer = []
    
    # Open file in append mode for resume capability
    with open(ALIGNED_OUTPUT_FILE, "a", encoding="utf-8") as f:
        
        print(f"📊 Processing {total_samples:,} samples...")
        print(f"   Starting from index {start_index}")
        print()
        
        for i, sample in tqdm(enumerate(dataset), total=total_samples, desc="Processing"):
            
            # Skip already processed samples
            if i < start_index:
                continue
            
            # Add to batch buffer
            batch_buffer.append({
                "audio": sample["audio"]["array"],
                "sampling_rate": sample["audio"]["sampling_rate"],
                "text": sample["text"],
                "id": i
            })
            
            # Process batch when full
            if len(batch_buffer) >= BATCH_SIZE:
                process_and_save_batch(pipe, batch_buffer, f)
                batch_buffer = []
        
        # Process remaining items
        if len(batch_buffer) > 0:
            process_and_save_batch(pipe, batch_buffer, f)
    
    print()
    print("=" * 60)
    print("✅ Timestamp Extraction Complete!")
    print("=" * 60)
    print()
    print(f"📁 Raw dataset: {OUTPUT_DIR}")
    print(f"📁 Aligned data: {ALIGNED_OUTPUT_FILE}")
    print()
    print(f"📊 Total processed: {get_processed_count():,} samples")
    print()


def process_and_save_batch(pipe, batch, file_handle):
    """Process a batch of audio and save results with timestamps."""
    
    # Extract audio arrays
    audio_inputs = [item["audio"] for item in batch]
    
    # Run Whisper with word-level timestamps
    try:
        results = pipe(audio_inputs, batch_size=len(batch), return_timestamps="word")
    except Exception as e:
        print(f"⚠️  Error processing batch: {e}")
        # Save error entries
        for item in batch:
            entry = {
                "id": item["id"],
                "original_text": item["text"],
                "error": str(e)
            }
            file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_handle.flush()
        return
    
    # Save results
    for original_item, model_output in zip(batch, results):
        # Extract timestamps - Whisper with return_timestamps="word" returns chunks
        # Each chunk is a dict with "text" and "timestamp" (tuple of start, end)
        chunks = model_output.get("chunks", [])
        
        # Process chunks to extract word-level timestamps
        word_timestamps = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")
                
                # Handle timestamp format: can be tuple (start, end) or single value
                if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                    start_time = float(timestamp[0]) if timestamp[0] is not None else None
                    end_time = float(timestamp[1]) if len(timestamp) > 1 and timestamp[1] is not None else None
                elif isinstance(timestamp, (int, float)):
                    start_time = float(timestamp)
                    end_time = None
                else:
                    # Try alternative field names
                    start_time = chunk.get("timestamp_start") or chunk.get("start")
                    end_time = chunk.get("timestamp_end") or chunk.get("end")
                    if start_time is not None:
                        start_time = float(start_time)
                    if end_time is not None:
                        end_time = float(end_time)
                
                # Only add if we have text and at least start time
                if text and start_time is not None:
                    word_info = {
                        "text": text,
                        "start": start_time,
                    }
                    if end_time is not None:
                        word_info["end"] = end_time
                    word_timestamps.append(word_info)
            elif isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
                # Alternative format: [text, start, end] or [start, end, text]
                if len(chunk) == 3:
                    # Try to determine format
                    if isinstance(chunk[0], str):
                        # Format: [text, start, end]
                        word_timestamps.append({
                            "text": str(chunk[0]).strip(),
                            "start": float(chunk[1]) if chunk[1] is not None else None,
                            "end": float(chunk[2]) if len(chunk) > 2 and chunk[2] is not None else None,
                        })
                    else:
                        # Format: [start, end, text]
                        word_timestamps.append({
                            "text": str(chunk[2]).strip() if len(chunk) > 2 else "",
                            "start": float(chunk[0]) if chunk[0] is not None else None,
                            "end": float(chunk[1]) if len(chunk) > 1 and chunk[1] is not None else None,
                        })
        
        entry = {
            "id": original_item["id"],
            "original_text": original_item["text"],
            "transcription": model_output.get("text", ""),
            "timestamps": word_timestamps,
        }
        
        # Write JSONL line
        file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Force write to disk
    file_handle.flush()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download phoaudiobook dataset with word-level timestamps")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check existing timestamp data, don't download/process")
    parser.add_argument("--check-samples", type=int, default=5,
                       help="Number of samples to check (default: 5)")
    parser.add_argument("--show-examples", action="store_true",
                       help="Show example format of word-level timestamps")
    args = parser.parse_args()
    
    if args.show_examples:
        show_example_format()
    elif args.check_only:
        # Only check existing data
        check_timestamps_in_data(num_samples=args.check_samples)
    else:
        check_disk_space()
        download_dataset()


if __name__ == "__main__":
    main()

