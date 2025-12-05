"""
Script to slice audio files into shorter segments (2-4 seconds) based on word timestamps.
This helps small models learn better by breaking long audio into manageable chunks.
"""

import pandas as pd
import json
import librosa
import soundfile as sf
import os
from pathlib import Path
from tqdm import tqdm

# CẤU HÌNH
INPUT_MANIFEST = "data/processed/merged_dataset/train/manifest_sorted.csv"
OUTPUT_DIR = "data/processed/merged_dataset/train/audio_sliced"
OUTPUT_MANIFEST = "data/processed/merged_dataset/train/manifest_sliced.csv"
BASE_AUDIO_DIR = "data/processed/merged_dataset/train"

# Tạo thư mục output
os.makedirs(OUTPUT_DIR, exist_ok=True)

def slice_audio_by_timestamps():
    """Cắt nhỏ file âm thanh dựa trên timestamps để tạo dataset dễ học hơn."""
    
    print(f"📂 Loading manifest from: {INPUT_MANIFEST}")
    df = pd.read_csv(INPUT_MANIFEST)
    
    # Kiểm tra cột cần thiết
    required_cols = ['audio_path', 'words_json', 'transcript', 'duration']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    new_data = []
    total_original = len(df)
    total_sliced = 0
    
    print(f"✂️ Đang cắt nhỏ {total_original} file âm thanh dựa trên Timestamps...")
    print(f"   Target: 2-4 giây mỗi đoạn")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Slicing audio"):
        try:
            # 1. Build full audio path
            audio_path = row['audio_path']
            if not os.path.isabs(audio_path):
                full_audio_path = os.path.join(BASE_AUDIO_DIR, audio_path)
            else:
                full_audio_path = audio_path
            
            if not os.path.exists(full_audio_path):
                print(f"⚠️  File not found: {full_audio_path}, skipping...")
                continue
            
            # 2. Load audio
            audio, sr = librosa.load(full_audio_path, sr=16000)
            
            # 3. Parse JSON timestamps
            words_json_str = row['words_json']
            if pd.isna(words_json_str) or words_json_str == '':
                # Không có timestamps, bỏ qua file này
                continue
            
            try:
                words = json.loads(words_json_str)
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON in row {idx}, skipping...")
                continue
            
            if not words or len(words) == 0:
                continue
            
            # 4. Gom nhóm từ thành các đoạn (Chunks) khoảng 2-4 giây
            current_chunk = []
            current_duration = 0
            chunk_idx = 0
            
            for word_info in words:
                word_start = word_info.get('start', 0.0)
                word_end = word_info.get('end', word_start + 0.5)
                word_text = word_info.get('word', '')
                
                word_dur = word_end - word_start
                current_chunk.append(word_info)
                current_duration += word_dur
                
                # Nếu cụm từ đủ dài (> 2 giây) hoặc là từ cuối cùng -> Cắt!
                if current_duration >= 2.0 or word_info == words[-1]:
                    # Đảm bảo chunk không quá ngắn (< 1 giây) trừ khi là chunk cuối
                    if current_duration < 1.0 and word_info != words[-1]:
                        continue  # Bỏ qua chunk quá ngắn, gộp vào chunk sau
                    
                    # Xác định thời gian cắt
                    start_time = current_chunk[0]['start']
                    end_time = current_chunk[-1]['end']
                    
                    # Đảm bảo end_time không vượt quá audio length
                    audio_duration = len(audio) / sr
                    end_time = min(end_time, audio_duration)
                    
                    if end_time <= start_time:
                        current_chunk = []
                        current_duration = 0
                        continue
                    
                    # Ghép text (loại bỏ language tags nếu có)
                    chunk_text = " ".join([w.get('word', '') for w in current_chunk])
                    chunk_text = chunk_text.replace('<|vi|>', '').replace('<|en|>', '').strip()
                    
                    if not chunk_text:
                        current_chunk = []
                        current_duration = 0
                        continue
                    
                    # Cắt Audio
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if end_sample <= start_sample or end_sample > len(audio):
                        current_chunk = []
                        current_duration = 0
                        continue
                    
                    chunk_audio = audio[start_sample:end_sample]
                    
                    # Lưu file mới
                    original_id = row.get('id', f"file_{idx}")
                    if pd.isna(original_id):
                        original_id = f"file_{idx}"
                    new_filename = f"{original_id}_chunk{chunk_idx}.wav"
                    new_path = os.path.join(OUTPUT_DIR, new_filename)
                    
                    sf.write(new_path, chunk_audio, sr)
                    
                    # Tính duration thực tế
                    chunk_duration = len(chunk_audio) / sr
                    
                    # Tạo words_json cho chunk (chỉ các từ trong chunk này)
                    chunk_words = []
                    for w in current_chunk:
                        # Adjust timestamps relative to chunk start
                        chunk_words.append({
                            'word': w.get('word', ''),
                            'start': w.get('start', 0.0) - start_time,
                            'end': w.get('end', 0.0) - start_time
                        })
                    
                    # Thêm vào list dữ liệu mới
                    new_data.append({
                        'id': f"{original_id}_chunk{chunk_idx}",
                        'transcript': chunk_text,
                        'audio_path': f"audio_sliced/{new_filename}",
                        'duration': chunk_duration,
                        'words_json': json.dumps(chunk_words, ensure_ascii=False)
                    })
                    
                    total_sliced += 1
                    
                    # Reset
                    current_chunk = []
                    current_duration = 0
                    chunk_idx += 1
                    
        except Exception as e:
            print(f"❌ Lỗi file {row.get('audio_path', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. Lưu manifest mới
    if len(new_data) == 0:
        print("❌ Không tạo được file nào! Kiểm tra lại dữ liệu.")
        return
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(OUTPUT_MANIFEST, index=False)
    
    print(f"\n✅ Xong!")
    print(f"   Original files: {total_original}")
    print(f"   Sliced segments: {total_sliced}")
    print(f"   Average segments per file: {total_sliced / total_original:.1f}")
    print(f"   Saved to: {OUTPUT_MANIFEST}")
    print(f"\n📊 Statistics:")
    print(f"   Min duration: {new_df['duration'].min():.2f}s")
    print(f"   Max duration: {new_df['duration'].max():.2f}s")
    print(f"   Avg duration: {new_df['duration'].mean():.2f}s")
    print(f"   Files < 2s: {(new_df['duration'] < 2.0).sum()}")
    print(f"   Files 2-4s: {((new_df['duration'] >= 2.0) & (new_df['duration'] <= 4.0)).sum()}")
    print(f"   Files > 4s: {(new_df['duration'] > 4.0).sum()}")


if __name__ == "__main__":
    slice_audio_by_timestamps()

