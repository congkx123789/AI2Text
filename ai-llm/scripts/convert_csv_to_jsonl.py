"""
Convert CSV manifest sang JSONL format cho training
Chuyển đổi từ format CSV (id, transcript, audio_path, duration, words_json)
sang JSONL format phù hợp cho Whisper và LLM training
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any


def extract_language(transcript: str) -> str:
    """Extract language từ transcript (format: <|vi|> hoặc <|en|>)"""
    match = re.match(r'<\|(vi|en)\|>', transcript)
    if match:
        return match.group(1)
    return "en"  # Default


def clean_transcript(transcript: str) -> str:
    """Remove language tag từ transcript"""
    return re.sub(r'<\|(vi|en)\|>\s*', '', transcript).strip()


def convert_csv_to_whisper_jsonl(csv_path: Path, output_path: Path, base_audio_dir: Path):
    """Convert CSV sang Whisper training format"""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row['transcript']
            language = extract_language(transcript)
            text = clean_transcript(transcript)
            
            # Audio path - có thể là relative hoặc absolute
            audio_path = row['audio_path']
            # Fix path nếu có "audio/audio"
            if audio_path.startswith("audio/"):
                audio_path = audio_path.replace("audio/", "", 1)
            
            if not Path(audio_path).is_absolute():
                # Relative path - cần resolve với base_audio_dir
                audio_path = str(base_audio_dir / audio_path)
            
            data.append({
                "audio": audio_path,
                "text": text,
                "language": language
            })
    
    # Save JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Đã convert {len(data)} samples sang Whisper format")
    print(f"   - File: {output_path}")
    return len(data)


def convert_csv_to_llm_jsonl(csv_path: Path, output_path: Path):
    """Convert CSV sang LLM training format (Instruction/Input/Output)"""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row['transcript']
            language = extract_language(transcript)
            text = clean_transcript(transcript)
            
            # Format cho Qwen instruction template
            if language == "vi":
                instruction = "Chuyển đổi audio thành văn bản tiếng Việt."
            else:
                instruction = "Transcribe audio to English text."
            
            # Input là empty (vì đây là ASR task, không có input text)
            # Output là transcript
            data.append({
                "instruction": instruction,
                "input": "",  # Empty for ASR task
                "output": text
            })
    
    # Save JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Đã convert {len(data)} samples sang LLM format")
    print(f"   - File: {output_path}")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV manifest sang JSONL cho training")
    parser.add_argument(
        "--csv-dir",
        default="data/processed/full_merged_dataset",
        help="Directory chứa CSV files (train/val/test)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_finetune",
        help="Output directory cho JSONL files"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to convert (default: all)"
    )
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    total_whisper = 0
    total_llm = 0
    
    for split in splits:
        csv_path = csv_dir / split / "manifest.csv"
        if not csv_path.exists():
            print(f"⚠️  Không tìm thấy: {csv_path}")
            continue
        
        print(f"\n📂 Đang xử lý split: {split}")
        
        # Base audio directory
        base_audio_dir = csv_dir / split / "audio"
        
        # Convert cho Whisper
        whisper_output = output_dir / f"whisper_{split}.jsonl"
        count = convert_csv_to_whisper_jsonl(csv_path, whisper_output, base_audio_dir)
        total_whisper += count
        
        # Convert cho LLM
        llm_output = output_dir / f"llm_{split}.jsonl"
        count = convert_csv_to_llm_jsonl(csv_path, llm_output)
        total_llm += count
    
    # Tạo file mixed cho training (combine train + val)
    print("\n🔄 Đang tạo file mixed cho training...")
    
    # Whisper mixed
    whisper_train = output_dir / "whisper_train.jsonl"
    whisper_val = output_dir / "whisper_val.jsonl"
    whisper_mixed = output_dir / "whisper_train_mixed.jsonl"
    
    if whisper_train.exists() and whisper_val.exists():
        # Combine train + val
        data = []
        for f in [whisper_train, whisper_val]:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        data.append(json.loads(line))
        
        with open(whisper_mixed, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Đã tạo Whisper mixed: {len(data)} samples")
    
    # LLM mixed
    llm_train = output_dir / "llm_train.jsonl"
    llm_val = output_dir / "llm_val.jsonl"
    llm_mixed = output_dir / "llm_train_mixed.jsonl"
    
    if llm_train.exists() and llm_val.exists():
        # Combine train + val
        data = []
        for f in [llm_train, llm_val]:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        data.append(json.loads(line))
        
        with open(llm_mixed, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Đã tạo LLM mixed: {len(data)} samples")
    
    print("\n✅ Hoàn thành!")
    print(f"\n📝 Files đã được tạo tại: {output_dir}")
    print(f"   - Whisper: whisper_train_mixed.jsonl ({total_whisper} samples)")
    print(f"   - LLM: llm_train_mixed.jsonl ({total_llm} samples)")
    print("\n🚀 Bạn có thể chạy training:")
    print("   - Whisper: python3 scripts/train_whisper.py")
    print("   - LLM: python3 scripts/train_llm.py")


if __name__ == "__main__":
    main()

