#!/usr/bin/env python3
"""
Script để debug predictions của model ASR.
In ra reference và prediction để xem model đang "nói" gì.

Kịch bản TỐT (Model đang học):
  Ref: "xin chào việt nam"
  Pred: "xxiin cchhaààoo vviiệệtt nnaamm" (Bị lặp ký tự) hoặc "sin chao viet nam" (Sai dấu)
  => Tiếp tục train. CTC sẽ sửa lỗi này sau.

Kịch bản XẤU 1 (Lỗi Blank/Tokenizer):
  Ref: "xin chào việt nam"
  Pred: "" (Rỗng tuếch)
  => Dừng ngay. Kiểm tra lại learning rate hoặc tokenizer.

Kịch bản XẤU 2 (Lỗi Sample Rate):
  Ref: "xin chào việt nam"
  Pred: "xin ch" (Bị cắt cụt) hoặc "x i n c h a o v i e t n a m a b c..." (Dài lê thê vô nghĩa)
  => Dừng ngay. Fix lại code xử lý âm thanh.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import BilingualTextNormalizer, Tokenizer
from training.dataset import ASRDataset


def debug_prediction(model, processor, tokenizer, item, device='cuda'):
    """
    Debug prediction cho 1 sample.
    
    Args:
        model: Model ASR đã load
        processor: AudioProcessor instance
        tokenizer: Tokenizer instance
        item: Dictionary chứa audio_path và transcript
        device: Device để chạy model
    """
    model.eval()
    
    # Load và process audio
    audio_path = item['audio_path']
    reference_text = item.get('transcript', item.get('text', ''))
    
    try:
        audio, sr = processor.load_audio(audio_path)
        audio = processor.trim_silence(audio)
        mel_spec = processor.extract_mel_spectrogram(audio)
        mel_spec = mel_spec.T  # (time, freq)
        
        # Convert to tensor
        audio_features = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(audio_features)
            
            # Lấy logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0]
            
            # Decode prediction
            # CTC decoding: argmax + remove blanks + collapse repeats
            pred_ids = torch.argmax(logits, dim=-1)  # (batch, time)
            pred_ids = pred_ids.squeeze(0).cpu().numpy()
            
            # Decode với tokenizer
            # Remove CTC blank tokens (thường là 0 hoặc tokenizer.blank_id)
            blank_id = getattr(tokenizer, 'blank_id', 0)
            pred_ids = pred_ids[pred_ids != blank_id]
            
            # Collapse repeats (CTC decoding)
            if len(pred_ids) > 0:
                # Remove consecutive duplicates
                unique_pred = [pred_ids[0]]
                for i in range(1, len(pred_ids)):
                    if pred_ids[i] != pred_ids[i-1]:
                        unique_pred.append(pred_ids[i])
                pred_ids = np.array(unique_pred)
            
            # Decode to text
            try:
                pred_text = tokenizer.decode(pred_ids.tolist())
            except:
                # Fallback: decode từng token
                pred_text = tokenizer.decode(list(pred_ids))
        
        # Print results
        print("-" * 80)
        print(f"📁 File: {Path(audio_path).name}")
        print(f"🎯 Reference: {reference_text}")
        print(f"🤖 Prediction: {pred_text}")
        
        # Phân tích kết quả
        if pred_text == "":
            print("🚨 KỊCH BẢN XẤU 1: Prediction rỗng!")
            print("   → Có thể do:")
            print("     - Learning rate quá cao/thấp")
            print("     - Tokenizer không đúng")
            print("     - Model chưa học được gì")
            print("   → HÀNH ĐỘNG: Dừng training, kiểm tra lại config!")
        elif len(pred_text) < len(reference_text) * 0.3:
            print("⚠️  Prediction quá ngắn (bị cắt cụt)")
            print("   → Có thể do sample rate không đúng!")
        elif len(pred_text) > len(reference_text) * 3:
            print("⚠️  Prediction quá dài (lặp lại vô nghĩa)")
            print("   → Có thể do sample rate không đúng!")
        elif any(c in pred_text for c in ['xx', 'aa', 'ii', 'ee', 'oo', 'uu']):
            print("✅ KỊCH BẢN TỐT: Model đang học (bị lặp ký tự)")
            print("   → Đây là bình thường với CTC, sẽ tự sửa khi train tiếp")
        else:
            # So sánh từng từ
            ref_words = reference_text.lower().split()
            pred_words = pred_text.lower().split()
            if len(ref_words) > 0 and len(pred_words) > 0:
                match_count = sum(1 for w in pred_words if w in ref_words)
                match_rate = match_count / len(ref_words)
                if match_rate > 0.5:
                    print(f"✅ Prediction có {match_rate*100:.1f}% từ khớp với reference")
                else:
                    print(f"⚠️  Prediction chỉ có {match_rate*100:.1f}% từ khớp")
        
        print("-" * 80)
        
        return {
            'reference': reference_text,
            'prediction': pred_text,
            'audio_path': audio_path,
            'is_empty': pred_text == "",
            'is_too_short': len(pred_text) < len(reference_text) * 0.3,
            'is_too_long': len(pred_text) > len(reference_text) * 3
        }
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_dataset(model_path: str, 
                  dataset_path: str,
                  manifest_path: str,
                  num_samples: int = 10,
                  device: str = 'cuda'):
    """
    Debug predictions trên dataset.
    
    Args:
        model_path: Đường dẫn đến checkpoint model
        dataset_path: Đường dẫn đến thư mục dataset (chứa audio/)
        manifest_path: Đường dẫn đến manifest.csv
        num_samples: Số samples để debug
        device: Device để chạy model
    """
    print("=" * 80)
    print("🔍 DEBUG PREDICTIONS")
    print("=" * 80)
    
    # Load model
    print(f"📦 Đang load model từ: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            # Cần load model architecture trước
            print("⚠️  Cần model architecture để load state dict")
            print("   → Sử dụng evaluate_checkpoint.py thay vì script này")
            return
        else:
            model = checkpoint
    except Exception as e:
        print(f"❌ Không thể load model: {e}")
        print("   → Hãy sử dụng evaluate_checkpoint.py để load model đúng cách")
        return
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"📂 Đang load dataset từ: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Setup processors
    processor = AudioProcessor(sample_rate=16000, n_mels=80)
    tokenizer = Tokenizer()
    
    # Chọn samples để debug
    if num_samples > len(df):
        num_samples = len(df)
    
    print(f"🎯 Sẽ debug {num_samples} samples...")
    print()
    
    results = []
    for idx in tqdm(range(num_samples), desc="Debugging"):
        row = df.iloc[idx]
        
        # Tạo item
        audio_path = Path(dataset_path) / row['audio_path']
        if not audio_path.exists():
            print(f"⚠️  File không tồn tại: {audio_path}")
            continue
        
        item = {
            'audio_path': str(audio_path),
            'transcript': row['transcript'],
            'text': row.get('transcript', '')
        }
        
        result = debug_prediction(model, processor, tokenizer, item, device)
        if result:
            results.append(result)
    
    # Tổng kết
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT")
    print("=" * 80)
    
    if results:
        empty_count = sum(1 for r in results if r['is_empty'])
        too_short_count = sum(1 for r in results if r['is_too_short'])
        too_long_count = sum(1 for r in results if r['is_too_long'])
        
        print(f"Tổng số samples: {len(results)}")
        print(f"🚨 Prediction rỗng: {empty_count} ({empty_count/len(results)*100:.1f}%)")
        print(f"⚠️  Prediction quá ngắn: {too_short_count} ({too_short_count/len(results)*100:.1f}%)")
        print(f"⚠️  Prediction quá dài: {too_long_count} ({too_long_count/len(results)*100:.1f}%)")
        
        if empty_count > len(results) * 0.5:
            print("\n🚨 CẢNH BÁO: Hơn 50% predictions rỗng!")
            print("   → Model có vấn đề nghiêm trọng!")
            print("   → Kiểm tra:")
            print("     1. Learning rate")
            print("     2. Tokenizer")
            print("     3. Model architecture")
        
        if too_short_count + too_long_count > len(results) * 0.3:
            print("\n⚠️  CẢNH BÁO: Nhiều predictions bất thường về độ dài!")
            print("   → Có thể do sample rate không đúng!")
            print("   → Chạy: python scripts/check_sample_rate.py --dataset <dataset_path>")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Debug predictions của model ASR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Debug với checkpoint và dataset
  python scripts/debug_predictions.py \\
    --model checkpoints/best_model.pt \\
    --dataset data/processed/full_merged_dataset/val \\
    --manifest data/processed/full_merged_dataset/val/manifest.csv \\
    --num_samples 20
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Đường dẫn đến checkpoint model'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Đường dẫn đến thư mục dataset (chứa audio/)'
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Đường dẫn đến manifest.csv'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Số samples để debug (default: 10)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device để chạy model (default: cuda)'
    )
    
    args = parser.parse_args()
    
    debug_dataset(
        model_path=args.model,
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        num_samples=args.num_samples,
        device=args.device
    )

