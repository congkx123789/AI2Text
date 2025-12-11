"""
Build SentencePiece BPE Tokenizer (kiểu Whisper/GPT).

Đặc điểm:
- Dính dấu cách vào đầu từ (Ví dụ: _xin, _chào)
- Không có token dấu cách riêng lẻ
- Vocab 3500 (Phủ 100% tiếng Việt + Anh)
"""

import sentencepiece as spm
import re
import os
from pathlib import Path
from utils.manifest_loader import load_merged_dataset


# ================= CẤU HÌNH =================
MODEL_NAME = 'tokenizer_vi_en_3500'         # Tên model sẽ tạo ra
VOCAB_SIZE = 3500                           # Số lượng token
DATASET_ROOT = "data/processed/full_merged_dataset"
CLEAN_FILE = 'corpus_clean.txt'             # File sạch tạm thời


# ================= 1. LÀM SẠCH DỮ LIỆU (QUAN TRỌNG) =================
def clean_corpus(texts: list[str]) -> str:
    """
    Làm sạch dữ liệu để phủ 100% ký tự.
    Lọc bỏ rác (icon, emoji, tiếng Tàu...) trước.
    """
    print("🧹 Đang làm sạch dữ liệu...")
    
    # Giữ lại: Chữ (Vi+En), Số, Dấu câu cơ bản, Dấu gạch ngang
    allowed_chars = r'[^\w\s.,!?:;"\'%\-]'
    
    clean_texts = []
    for text in texts:
        if not text or not text.strip():
            continue
        # Lọc bỏ ký tự lạ
        clean_text = re.sub(allowed_chars, '', text.strip())
        if clean_text:
            clean_texts.append(clean_text)
    
    # Ghi vào file tạm
    with open(CLEAN_FILE, 'w', encoding='utf-8') as f_out:
        for text in clean_texts:
            f_out.write(text + '\n')
    
    print(f"✅ Đã tạo file dữ liệu sạch: {len(clean_texts):,} dòng")
    return CLEAN_FILE


# ================= 2. TRAIN TOKENIZER (BPE) =================
def train_bpe(input_file: str, model_prefix: str, vocab_size: int):
    """
    Train SentencePiece BPE tokenizer.
    """
    print(f"🚀 Đang train SentencePiece BPE (Vocab: {vocab_size})...")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        
        # --- CẤU HÌNH CỐT LÕI ---
        model_type='bpe',          # Thuật toán BPE
        character_coverage=1.0,    # Bắt buộc học HẾT 100% ký tự
        
        # --- XỬ LÝ DẤU CÁCH (Cái bạn cần) ---
        # Mặc định SentencePiece sẽ dùng ký tự '_' (U+2581) để đánh dấu đầu từ
        # add_dummy_prefix=True: Thêm '_' vào cả từ đầu tiên của câu 
        # (để 'Xin' và '_xin' được coi là giống nhau -> Tiết kiệm vocab)
        add_dummy_prefix=True,
        
        # Token đặc biệt cho ASR CTC
        pad_id=0,                  # <pad> / <blank>
        unk_id=1,                  # <unk>
        bos_id=2,                  # <s>
        eos_id=3,                  # </s>
        
        normalization_rule_name='nmt_nfkc'  # Chuẩn hóa Unicode
    )
    print(f"🎉 Xong! Đã có file {model_prefix}.model và {model_prefix}.vocab")


# ================= 3. TEST THỬ =================
def test_tokenizer(model_path: str):
    """
    Test tokenizer với text mẫu.
    """
    print("\n🧪 --- KIỂM TRA KẾT QUẢ ---")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # Text mẫu hỗn hợp
    text = "Xin chào Việt Nam hello AI"
    
    # 1. Cắt ra token (Xem dấu gạch dưới)
    tokens = sp.encode_as_pieces(text)
    
    # 2. Chuyển thành số ID
    ids = sp.encode_as_ids(text)
    
    print(f"Input:  '{text}'")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    
    # 3. Decode lại (Xem có mất mát gì ko)
    decoded = sp.decode(ids)
    print(f"Decode: '{decoded}'")
    
    # 4. Kiểm tra vocab size
    print(f"\nVocab size: {sp.get_piece_size()}")
    print(f"Pad ID: {sp.pad_id()}")
    print(f"Unk ID: {sp.unk_id()}")
    print(f"Bos ID: {sp.bos_id()}")
    print(f"Eos ID: {sp.eos_id()}")


# ================= 4. COLLECT TEXTS FROM DATASET =================
def collect_texts_from_dataset(
    dataset_root: str = DATASET_ROOT,
    split: str = "train",
    languages: list[str] | None = None,
) -> list[str]:
    """
    Load transcripts from merged_dataset to train tokenizer.
    """
    df = load_merged_dataset(split=split, dataset_root=dataset_root)
    
    if languages:
        df = df[df["language"].isin(languages)].reset_index(drop=True)
    
    if "transcript" not in df.columns:
        raise ValueError("Manifest must contain a 'transcript' column")
    
    texts = df["transcript"].astype(str).tolist()
    print(f"✅ Collected {len(texts):,} transcripts from merged_dataset ({split})")
    if languages:
        print(f"   Languages: {', '.join(languages)}")
    return texts


if __name__ == "__main__":
    # Tạo thư mục models nếu chưa có
    Path("models").mkdir(exist_ok=True)
    
    # 1. Collect texts from dataset
    print("📚 Đang load transcripts từ dataset...")
    texts = collect_texts_from_dataset(
        dataset_root=DATASET_ROOT,
        split="train",
        languages=["vi", "en"],  # Bilingual: Vietnamese + English
    )
    
    # 2. Clean corpus
    clean_file = clean_corpus(texts)
    
    # 3. Train tokenizer
    model_path = f"models/{MODEL_NAME}"
    train_bpe(clean_file, model_path, VOCAB_SIZE)
    
    # 4. Test tokenizer
    test_tokenizer(f"{model_path}.model")
    
    # 5. Clean up temporary file
    if os.path.exists(CLEAN_FILE):
        os.remove(CLEAN_FILE)
        print(f"\n🧹 Đã xóa file tạm: {CLEAN_FILE}")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT!")
    print("=" * 80)
    print(f"📁 Model file: {model_path}.model")
    print(f"📁 Vocab file: {model_path}.vocab")
    print(f"📊 Vocab size: {VOCAB_SIZE}")
    print("\n💡 Để sử dụng trong code:")
    print("   from preprocessing.sentencepiece_tokenizer import SentencePieceTokenizer")
    print(f"   tokenizer = SentencePieceTokenizer('{model_path}.model')")

