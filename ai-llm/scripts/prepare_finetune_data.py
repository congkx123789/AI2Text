"""
BƯỚC 1: Chuẩn bị Dataset cho Fine-tuning
Trộn dataset tiếng Anh và tiếng Việt thành 2 file:
- whisper_train_mixed.jsonl: Cho training Whisper
- llm_train_mixed.jsonl: Cho training LLM
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """Save data to JSONL file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def prepare_whisper_dataset(
    en_data: List[Dict[str, Any]],
    vi_data: List[Dict[str, Any]],
    output_path: Path,
    ratio: float = 0.5
):
    """
    Chuẩn bị dataset cho Whisper training
    
    Format: {"audio": path, "text": text, "language": "en"|"vi"}
    """
    mixed_data = []
    
    # Tính số lượng mỗi loại
    total = len(en_data) + len(vi_data)
    en_count = int(total * ratio)
    vi_count = total - en_count
    
    # Lấy mẫu từ mỗi dataset
    en_samples = random.sample(en_data, min(en_count, len(en_data)))
    vi_samples = random.sample(vi_data, min(vi_count, len(vi_data)))
    
    # Format cho Whisper
    for item in en_samples:
        mixed_data.append({
            "audio": item.get("audio", item.get("file", "")),
            "text": item.get("text", item.get("transcription", "")),
            "language": "en"
        })
    
    for item in vi_samples:
        mixed_data.append({
            "audio": item.get("audio", item.get("file", "")),
            "text": item.get("text", item.get("transcription", "")),
            "language": "vi"
        })
    
    # Shuffle
    random.shuffle(mixed_data)
    
    save_jsonl(mixed_data, output_path)
    print(f"✅ Đã tạo Whisper dataset: {len(mixed_data)} samples")
    print(f"   - Tiếng Anh: {len(en_samples)}")
    print(f"   - Tiếng Việt: {len(vi_samples)}")
    print(f"   - File: {output_path}")


def prepare_llm_dataset(
    en_data: List[Dict[str, Any]],
    vi_data: List[Dict[str, Any]],
    output_path: Path,
    ratio: float = 0.5
):
    """
    Chuẩn bị dataset cho LLM training (QLoRA)
    
    Format: {"instruction": ..., "input": ..., "output": ...}
    """
    mixed_data = []
    
    # Tính số lượng mỗi loại
    total = len(en_data) + len(vi_data)
    en_count = int(total * ratio)
    vi_count = total - en_count
    
    # Lấy mẫu từ mỗi dataset
    en_samples = random.sample(en_data, min(en_count, len(en_data)))
    vi_samples = random.sample(vi_data, min(vi_count, len(vi_data)))
    
    # Format cho Qwen (Instruction template)
    for item in en_samples:
        instruction = item.get("instruction", "Answer the following question.")
        input_text = item.get("input", item.get("question", ""))
        output_text = item.get("output", item.get("answer", item.get("text", "")))
        
        mixed_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    for item in vi_samples:
        instruction = item.get("instruction", "Trả lời câu hỏi sau.")
        input_text = item.get("input", item.get("question", ""))
        output_text = item.get("output", item.get("answer", item.get("text", "")))
        
        mixed_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    # Shuffle
    random.shuffle(mixed_data)
    
    save_jsonl(mixed_data, output_path)
    print(f"✅ Đã tạo LLM dataset: {len(mixed_data)} samples")
    print(f"   - Tiếng Anh: {len(en_samples)}")
    print(f"   - Tiếng Việt: {len(vi_samples)}")
    print(f"   - File: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset cho fine-tuning")
    parser.add_argument("--en-data", required=True, help="Path to English dataset JSONL")
    parser.add_argument("--vi-data", required=True, help="Path to Vietnamese dataset JSONL")
    parser.add_argument("--output-dir", default="data/processed_finetune", help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of English data (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load datasets
    print("📂 Đang load datasets...")
    en_data = load_jsonl(Path(args.en_data))
    vi_data = load_jsonl(Path(args.vi_data))
    
    print(f"   - Tiếng Anh: {len(en_data)} samples")
    print(f"   - Tiếng Việt: {len(vi_data)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare Whisper dataset
    print("\n🎤 Đang chuẩn bị dataset cho Whisper...")
    whisper_output = output_dir / "whisper_train_mixed.jsonl"
    prepare_whisper_dataset(en_data, vi_data, whisper_output, args.ratio)
    
    # Prepare LLM dataset
    print("\n💬 Đang chuẩn bị dataset cho LLM...")
    llm_output = output_dir / "llm_train_mixed.jsonl"
    prepare_llm_dataset(en_data, vi_data, llm_output, args.ratio)
    
    print("\n✅ Hoàn thành! Bạn có thể chạy training:")
    print(f"   - Whisper: python3 scripts/train_whisper.py")
    print(f"   - LLM: python3 scripts/train_llm.py")


if __name__ == "__main__":
    main()


