"""
Test Whisper model chi tiết từng câu và lưu kết quả vào file log
"""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import json
from datetime import datetime
import librosa


def load_audio(example, base_dir):
    """Load audio file"""
    audio_path_str = example["audio"]
    
    # Fix duplicate /audio/audio/ in path
    audio_path_str = audio_path_str.replace("/audio/audio/", "/audio/")
    
    # Try multiple paths
    project_root = Path(__file__).parent.parent
    audio_path = None
    
    # Try 1: Relative to project root
    test_path = project_root / audio_path_str
    if test_path.exists():
        audio_path = test_path
    else:
        # Try 2: Extract filename and find in base_dir
        filename = Path(audio_path_str).name
        # Try to find in test/audio/
        test_audio_path = Path(base_dir) / "test" / "audio" / filename
        if test_audio_path.exists():
            audio_path = test_audio_path
        else:
            # Try 3: Direct path from base_dir
            test_path2 = Path(base_dir) / audio_path_str
            if test_path2.exists():
                audio_path = test_path2
    
    if audio_path and audio_path.exists():
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            example["audio"] = {"array": audio, "sampling_rate": sr}
            example["audio_path"] = str(audio_path)
        except Exception as e:
            print(f"⚠️  Lỗi load audio {audio_path}: {e}")
            example["audio"] = {"array": [], "sampling_rate": 16000}
            example["audio_path"] = None
    else:
        example["audio"] = {"array": [], "sampling_rate": 16000}
        example["audio_path"] = None
    return example


def compute_word_errors(reference, prediction):
    """Compute word-level errors"""
    ref_words = reference.lower().split()
    pred_words = prediction.lower().split()
    
    # Simple word error calculation
    ref_set = set(ref_words)
    pred_set = set(pred_words)
    
    correct = len(ref_set & pred_set)
    total = len(ref_set)
    
    if total == 0:
        return 0, 0, 0
    
    errors = total - correct
    error_rate = errors / total if total > 0 else 0
    
    return errors, total, error_rate


def test_model_detailed(model_path, test_dataset_path, base_audio_dir, output_log):
    """Test model chi tiết từng câu"""
    print(f"\n🔍 Đang test model: {model_path}")
    print(f"📝 Log sẽ được lưu tại: {output_log}")
    
    # Load model và processor
    print("📦 Đang load model...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
        print(f"✅ Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️  Sử dụng CPU (sẽ chậm hơn)")
    
    # Load test dataset
    print(f"\n📂 Đang load test dataset: {test_dataset_path}")
    dataset = load_dataset("json", data_files=test_dataset_path, split="train")
    
    # Load audio files
    print("🎵 Đang load audio files...")
    dataset = dataset.map(
        lambda x: load_audio(x, base_audio_dir),
        desc="Loading audio"
    )
    dataset = dataset.filter(lambda x: len(x["audio"]["array"]) > 0)
    
    print(f"   - Tổng số samples: {len(dataset)}")
    
    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Prepare log file - lưu từng dòng JSON để tránh mất dữ liệu
    log_file_jsonl = output_log.replace(".json", ".jsonl")
    summary_file = output_log.replace(".json", "_summary.json")
    
    # Check if resume from checkpoint
    start_idx = 0
    if Path(log_file_jsonl).exists():
        print(f"📂 Tìm thấy file log cũ: {log_file_jsonl}")
        # Count existing results
        with open(log_file_jsonl, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        print(f"   Resume từ sample {start_idx}")
    
    # Open file for appending
    log_f = open(log_file_jsonl, "a", encoding="utf-8")
    
    print("\n📊 Bắt đầu test từng câu...")
    print("="*80)
    
    all_predictions = []
    all_references = []
    batch_results = []  # Lưu batch để tính metrics
    
    # Process từng sample một
    for idx in tqdm(range(start_idx, len(dataset)), desc="Testing", initial=start_idx, total=len(dataset)):
        sample = dataset[idx]
        
        # Prepare audio
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        reference_text = sample["text"]
        audio_path = sample.get("audio_path", "unknown")
        
        # Extract features
        try:
            input_features = processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features,
                    max_length=448,
                    language=None,  # Auto-detect
                    task="transcribe",
                    num_beams=1,
                    do_sample=False
                )
            
            # Decode
            prediction = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Clear GPU cache
            del input_features, generated_ids
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            prediction = f"[ERROR: {str(e)}]"
            print(f"⚠️  Lỗi ở sample {idx}: {e}")
        
        # Compute metrics for this sample
        wer_sample = wer_metric.compute(
            predictions=[prediction],
            references=[reference_text]
        )
        
        cer_sample = cer_metric.compute(
            predictions=[prediction],
            references=[reference_text]
        )
        
        # Word-level errors
        word_errors, word_total, word_error_rate = compute_word_errors(
            reference_text, prediction
        )
        
        # Store results
        all_predictions.append(prediction)
        all_references.append(reference_text)
        
        result_entry = {
            "sample_id": idx,
            "audio_path": audio_path,
            "reference": reference_text,
            "prediction": prediction,
            "wer": float(wer_sample),
            "cer": float(cer_sample),
            "word_errors": word_errors,
            "word_total": word_total,
            "word_error_rate": float(word_error_rate),
            "match": prediction.strip().lower() == reference_text.strip().lower()
        }
        
        batch_results.append(result_entry)
        
        # Lưu ngay vào file JSONL (từng dòng)
        log_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
        log_f.flush()  # Force write to disk
        
        # Print progress và save summary mỗi 100 samples
        if (idx + 1) % 100 == 0:
            avg_wer = sum(r["wer"] for r in batch_results) / len(batch_results)
            avg_cer = sum(r["cer"] for r in batch_results) / len(batch_results)
            print(f"\n📊 Progress: {idx+1}/{len(dataset)} | Avg WER: {avg_wer:.4f} | Avg CER: {avg_cer:.4f}")
            
            # Save intermediate summary
            intermediate_summary = {
                "last_sample": idx,
                "total_processed": len(batch_results),
                "avg_wer": float(avg_wer),
                "avg_cer": float(avg_cer),
                "timestamp": datetime.now().isoformat()
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(intermediate_summary, f, indent=2, ensure_ascii=False)
        
        # Clear batch để giảm RAM (giữ lại để tính tổng thể)
        if len(batch_results) > 1000:
            batch_results = batch_results[-500:]  # Giữ 500 mẫu gần nhất
    
    log_f.close()
    
    # Compute overall metrics từ file JSONL
    print("\n" + "="*80)
    print("📊 TÍNH TOÁN METRICS TỔNG THỂ...")
    print("="*80)
    
    # Đọc lại tất cả kết quả từ JSONL
    print("📖 Đang đọc kết quả từ file...")
    all_results = []
    all_predictions_final = []
    all_references_final = []
    
    with open(log_file_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line.strip())
            all_results.append(result)
            all_predictions_final.append(result["prediction"])
            all_references_final.append(result["reference"])
    
    print(f"   Đã đọc {len(all_results)} kết quả")
    
    # Compute overall metrics
    overall_wer = wer_metric.compute(
        predictions=all_predictions_final,
        references=all_references_final
    )
    
    overall_cer = cer_metric.compute(
        predictions=all_predictions_final,
        references=all_references_final
    )
    
    # Statistics
    perfect_matches = sum(1 for r in all_results if r["match"])
    perfect_match_rate = perfect_matches / len(all_results) if all_results else 0
    
    avg_wer = sum(r["wer"] for r in all_results) / len(all_results) if all_results else 0
    avg_cer = sum(r["cer"] for r in all_results) / len(all_results) if all_results else 0
    
    # Summary
    summary = {
        "model_path": model_path,
        "test_dataset": test_dataset_path,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(all_results),
        "device": device,
        "overall_wer": float(overall_wer),
        "overall_cer": float(overall_cer),
        "average_wer": float(avg_wer),
        "average_cer": float(avg_cer),
        "perfect_matches": perfect_matches,
        "perfect_match_rate": float(perfect_match_rate),
        "worst_samples": sorted(all_results, key=lambda x: x["wer"], reverse=True)[:10],
        "best_samples": sorted(all_results, key=lambda x: x["wer"])[:10]
    }
    
    # Print summary
    print("\n" + "="*80)
    print("📊 KẾT QUẢ TỔNG THỂ")
    print("="*80)
    print(f"Tổng số samples: {summary['total_samples']}")
    print(f"Overall WER: {summary['overall_wer']:.4f} ({summary['overall_wer']*100:.2f}%)")
    print(f"Overall CER: {summary['overall_cer']:.4f} ({summary['overall_cer']*100:.2f}%)")
    print(f"Average WER: {summary['average_wer']:.4f} ({summary['average_wer']*100:.2f}%)")
    print(f"Average CER: {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
    print(f"Perfect matches: {summary['perfect_matches']}/{summary['total_samples']} ({summary['perfect_match_rate']*100:.2f}%)")
    print("="*80)
    
    # Save summary file
    print(f"\n💾 Đang lưu summary vào: {summary_file}")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save full JSON (optional, có thể rất lớn)
    print(f"💾 Đang lưu full log JSON vào: {output_log}")
    log_data = {
        "model_path": model_path,
        "test_dataset": test_dataset_path,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(all_results),
        "device": device,
        "summary": summary,
        "results": all_results  # Có thể rất lớn
    }
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Hoàn thành!")
    print(f"   JSONL log (từng dòng): {log_file_jsonl}")
    print(f"   Summary file: {summary_file}")
    print(f"   Full JSON log: {output_log}")
    print(f"   Tổng số samples: {len(all_results)}")
    print(f"   Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"   Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test Whisper model chi tiết từng câu")
    parser.add_argument(
        "--model",
        default="models/finetuned/whisper-mixed",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test-dataset",
        default="data/processed_finetune/whisper_test.jsonl",
        help="Path to test dataset JSONL"
    )
    parser.add_argument(
        "--base-audio-dir",
        default="data/processed/full_merged_dataset",
        help="Base directory for audio files"
    )
    parser.add_argument(
        "--output",
        default="whisper_detailed_test_log.json",
        help="Output log file (JSON)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 CHI TIẾT TEST WHISPER MODEL")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Output log: {args.output}")
    print("="*80)
    
    test_model_detailed(
        args.model,
        args.test_dataset,
        args.base_audio_dir,
        args.output
    )


if __name__ == "__main__":
    main()







