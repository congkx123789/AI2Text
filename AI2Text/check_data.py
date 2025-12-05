import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import os
import argparse

# Cấu hình mặc định
DEFAULT_DATASET_ROOT = "data/processed/merged_dataset"
DEFAULT_SPLIT = "train"
MIN_DURATION = 0.5  # Giây
MAX_DURATION = 30.0  # Giây
MAX_CPS = 25.0  # Ký tự mỗi giây (Characters Per Second) - Nếu cao hơn là audio quá ngắn so với text
MIN_CPS = 0.5  # Nếu thấp hơn là audio quá dài (hoặc toàn khoảng lặng)


def clean_transcript(transcript: str) -> str:
    """Loại bỏ language tags từ transcript"""
    if pd.isna(transcript):
        return ''
    transcript_str = str(transcript)
    # Remove language tags
    transcript_str = transcript_str.replace('<|vi|>', '').replace('<|en|>', '').strip()
    return transcript_str


def check_data(dataset_root: str = None, split: str = None, output_file: str = None):
    """Kiểm tra dữ liệu trong dataset"""
    dataset_root = dataset_root or DEFAULT_DATASET_ROOT
    split = split or DEFAULT_SPLIT
    output_file = output_file or "bad_files.txt"
    
    print(f"🔍 Đang kiểm tra dữ liệu trong: {dataset_root}/{split}")

    # 1. Load Manifest
    manifest_path = Path(dataset_root) / split / "manifest.csv"
    if not manifest_path.exists():
        print(f"❌ Không tìm thấy file manifest: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"✅ Tổng số file trong manifest: {len(df)}")

    # Xác định base directory cho audio files
    base_audio_dir = manifest_path.parent  # Thư mục chứa manifest.csv

    errors = []
    warnings = []

    # 2. Quét từng file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        # Lấy audio_path từ manifest
        if 'audio_path' in row:
            audio_path = row['audio_path']
        elif 'file_path' in row:
            audio_path = row['file_path']
        else:
            errors.append((f"Row {idx}", "No audio_path or file_path column"))
            continue

        # Xây dựng đường dẫn đầy đủ
        if Path(audio_path).is_absolute():
            full_audio_path = Path(audio_path)
        else:
            # audio_path là relative (ví dụ: "audio/007_000000980.wav")
            full_audio_path = base_audio_dir / audio_path

        # Lấy transcript
        if 'transcript' in row:
            text = clean_transcript(row['transcript'])
        else:
            errors.append((str(full_audio_path), "No transcript column"))
            continue

        # A. Kiểm tra File tồn tại
        if not full_audio_path.exists():
            errors.append((str(full_audio_path), "File Not Found"))
            continue

        # B. Kiểm tra Audio (Corrupted?)
        try:
            # Chỉ lấy metadata để nhanh hơn, không load waveform
            info = sf.info(str(full_audio_path))
            duration = info.frames / info.samplerate

            if duration < MIN_DURATION:
                errors.append((str(full_audio_path), f"Too Short ({duration:.2f}s)"))
                continue
            if duration > MAX_DURATION:
                warnings.append((str(full_audio_path), f"Too Long ({duration:.2f}s)"))

        except Exception as e:
            errors.append((str(full_audio_path), f"Corrupted Audio: {str(e)}"))
            continue

        # C. Kiểm tra Text (Empty?)
        if len(text.strip()) == 0:
            errors.append((str(full_audio_path), "Empty Transcript"))
            continue

        # D. Kiểm tra Tỷ lệ (CTC Crash Risk)
        # Nếu text quá dài mà audio quá ngắn -> CTC Loss sẽ bị NaN/Inf
        char_count = len(text)
        cps = char_count / duration

        if cps > MAX_CPS:
            errors.append((str(full_audio_path), f"CPS Too High ({cps:.1f} char/s) - Risk of NaN"))
        elif cps < MIN_CPS:
            warnings.append((str(full_audio_path), f"CPS Too Low ({cps:.1f} char/s) - Mostly Silence?"))

    # 3. Báo cáo kết quả
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ KIỂM TRA")
    print("=" * 50)
    print(f"Sạch sẽ: {len(df) - len(errors)} files")
    print(f"Cảnh báo: {len(warnings)} files (Vẫn train được)")
    print(f"LỖI NGHIÊM TRỌNG: {len(errors)} files (Cần xóa ngay)")

    if len(errors) > 0:
        print("\nTop 10 lỗi:")
        for path, reason in errors[:10]:
            print(f"  ❌ {reason}: {path}")

        # Lưu danh sách lỗi ra file
        with open(output_file, "w", encoding='utf-8') as f:
            for path, reason in errors:
                f.write(f"{path},{reason}\n")
        print(f"\n👉 Danh sách file lỗi đã lưu vào '{output_file}'. Hãy xóa chúng khỏi manifest!")

    if len(warnings) > 0:
        print(f"\nTop 10 cảnh báo:")
        for path, reason in warnings[:10]:
            print(f"  ⚠️  {reason}: {path}")

        # Lưu danh sách cảnh báo ra file
        with open("warnings.txt", "w", encoding='utf-8') as f:
            for path, reason in warnings:
                f.write(f"{path},{reason}\n")
        print(f"👉 Danh sách cảnh báo đã lưu vào 'warnings.txt'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Kiểm tra chất lượng dữ liệu trong dataset'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=f'Root directory của dataset (mặc định: {DEFAULT_DATASET_ROOT})'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=DEFAULT_SPLIT,
        choices=['train', 'val', 'test'],
        help=f'Split cần kiểm tra (mặc định: {DEFAULT_SPLIT})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bad_files.txt',
        help='Tên file output chứa danh sách file lỗi (mặc định: bad_files.txt)'
    )
    
    args = parser.parse_args()
    check_data(
        dataset_root=args.dataset_root,
        split=args.split,
        output_file=args.output
    )

