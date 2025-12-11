"""
Test Whisper CTranslate2 model trên full test dataset với bf16
Ghi rõ từng prediction và reference cho mỗi sample
"""
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import evaluate
from datetime import datetime
from src.tools.ai2text_bridge import transcribe
from src.config import ASR_MODEL


def load_test_dataset(test_dataset_path, base_audio_dir):
    """Load test dataset"""
    print(f"📂 Loading test dataset: {test_dataset_path}")
    
    data = []
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"   Total samples: {len(data)}")
    
    # Resolve audio paths
    base_dir = Path(base_audio_dir)
    resolved_data = []
    
    for item in tqdm(data, desc="Resolving audio paths"):
        audio_path_str = item.get("audio", "")
        
        # Try multiple paths
        audio_path = None
        
        # Try 1: Direct path
        test_path = Path(audio_path_str)
        if test_path.exists():
            audio_path = test_path
        else:
            # Try 2: Relative to base_dir
            test_path2 = base_dir / audio_path_str
            if test_path2.exists():
                audio_path = test_path2
            else:
                # Try 3: Extract filename
                filename = Path(audio_path_str).name
                # Try in test/audio/
                test_audio_path = base_dir / "test" / "audio" / filename
                if test_audio_path.exists():
                    audio_path = test_audio_path
                else:
                    # Try in train/audio/
                    train_audio_path = base_dir / "train" / "audio" / filename
                    if train_audio_path.exists():
                        audio_path = train_audio_path
        
        if audio_path and audio_path.exists():
            resolved_data.append({
                "audio_path": str(audio_path),
                "reference": item.get("text", ""),
                "id": item.get("id", len(resolved_data))
            })
        else:
            print(f"⚠️  Audio not found: {audio_path_str}")
    
    print(f"   Valid samples: {len(resolved_data)}")
    return resolved_data


def test_model_full(
    model_path,
    test_dataset_path,
    base_audio_dir,
    output_log,
    device="cuda",
    compute="float16",
    max_samples=None
):
    """
    Test model trên full test dataset
    
    Args:
        model_path: Path to Whisper model (CTranslate2 format)
        test_dataset_path: Path to test dataset JSONL
        base_audio_dir: Base directory for audio files
        output_log: Output log file path
        device: Device (cuda, cpu)
        compute: Compute type (float16, bfloat16, int8)
        max_samples: Max samples to test (None = all)
    """
    print("=" * 80)
    print("🧪 TEST WHISPER MODEL - FULL DATASET")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test dataset: {test_dataset_path}")
    print(f"Device: {device}")
    print(f"Compute: {compute}")
    print(f"Max samples: {max_samples or 'All'}")
    print("=" * 80)
    
    # Load test dataset
    test_data = load_test_dataset(test_dataset_path, base_audio_dir)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"📊 Testing {len(test_data)} samples (limited from {len(load_test_dataset(test_dataset_path, base_audio_dir))})")
    else:
        print(f"📊 Testing {len(test_data)} samples")
    
    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Prepare log data
    log_data = {
        "model_path": str(model_path),
        "test_dataset": str(test_dataset_path),
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "compute": compute,
        "total_samples": len(test_data),
        "results": []
    }
    
    all_predictions = []
    all_references = []
    
    # Prepare output log file
    output_path = Path(output_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Starting evaluation...")
    print(f"💾 Results will be saved to: {output_log}")
    print("=" * 80)
    
    # Process each sample
    for idx, sample in enumerate(tqdm(test_data, desc="Testing")):
        audio_path = sample["audio_path"]
        reference = sample["reference"]
        sample_id = sample.get("id", idx)
        
        # Transcribe với fallback nếu GPU fail
        try:
            result = transcribe(
                audio_path,
                size=model_path,
                device=device,
                compute=compute
            )
            prediction = result["text"]
            language = result.get("language", "unknown")
        except Exception as e:
            # Fallback to CPU nếu GPU fail
            if device == "cuda" and "cudnn" in str(e).lower():
                print(f"⚠️  GPU error at sample {idx}, falling back to CPU: {e}")
                try:
                    result = transcribe(
                        audio_path,
                        size=model_path,
                        device="cpu",
                        compute="int8"
                    )
                    prediction = result["text"]
                    language = result.get("language", "unknown")
                except Exception as e2:
                    prediction = f"[ERROR: {str(e2)}]"
                    language = "error"
            else:
                prediction = f"[ERROR: {str(e)}]"
                language = "error"
                print(f"⚠️  Error at sample {idx}: {e}")
        
        # Compute metrics
        wer_sample = wer_metric.compute(
            predictions=[prediction],
            references=[reference]
        )
        
        cer_sample = cer_metric.compute(
            predictions=[prediction],
            references=[reference]
        )
        
        # Store results
        all_predictions.append(prediction)
        all_references.append(reference)
        
        result_entry = {
            "sample_id": sample_id,
            "audio_path": audio_path,
            "reference": reference,
            "prediction": prediction,
            "wer": float(wer_sample),
            "cer": float(cer_sample),
            "language": language,
            "match": prediction.strip().lower() == reference.strip().lower()
        }
        
        log_data["results"].append(result_entry)
        
        # Save to log file after each prediction (real-time)
        try:
            # Update progress info
            log_data["current_progress"] = {
                "completed": idx + 1,
                "total": len(test_data),
                "percentage": (idx + 1) / len(test_data) * 100
            }
            
            # Calculate current metrics
            if log_data["results"]:
                current_avg_wer = sum(r["wer"] for r in log_data["results"]) / len(log_data["results"])
                current_avg_cer = sum(r["cer"] for r in log_data["results"]) / len(log_data["results"])
                current_perfect = sum(1 for r in log_data["results"] if r["match"])
                log_data["current_metrics"] = {
                    "average_wer": float(current_avg_wer),
                    "average_cer": float(current_avg_cer),
                    "perfect_matches": current_perfect,
                    "perfect_match_rate": float(current_perfect / len(log_data["results"]))
                }
            
            # Save to file (append mode for safety, but we'll overwrite each time)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not save log: {e}")
        
        # Print every 100 samples
        if (idx + 1) % 100 == 0:
            avg_wer = sum(r["wer"] for r in log_data["results"]) / len(log_data["results"])
            avg_cer = sum(r["cer"] for r in log_data["results"]) / len(log_data["results"])
            perfect = sum(1 for r in log_data["results"] if r["match"])
            print(f"\n📊 Progress: {idx+1}/{len(test_data)} ({idx+1/len(test_data)*100:.1f}%)")
            print(f"   Avg WER: {avg_wer:.4f} | Avg CER: {avg_cer:.4f} | Perfect: {perfect}/{idx+1}")
            # Print latest example
            if log_data["results"]:
                example = log_data["results"][-1]
                print(f"   Latest Sample {example['sample_id']}:")
                print(f"   Reference: {example['reference'][:60]}...")
                print(f"   Prediction: {example['prediction'][:60]}...")
                print(f"   WER: {example['wer']:.4f} | Match: {'✅' if example['match'] else '❌'}")
            print(f"   💾 Log saved: {output_log}")
    
    # Compute overall metrics
    print("\n" + "=" * 80)
    print("📊 COMPUTING OVERALL METRICS...")
    print("=" * 80)
    
    overall_wer = wer_metric.compute(
        predictions=all_predictions,
        references=all_references
    )
    
    overall_cer = cer_metric.compute(
        predictions=all_predictions,
        references=all_references
    )
    
    # Statistics
    perfect_matches = sum(1 for r in log_data["results"] if r["match"])
    perfect_match_rate = perfect_matches / len(log_data["results"]) if log_data["results"] else 0
    
    avg_wer = sum(r["wer"] for r in log_data["results"]) / len(log_data["results"]) if log_data["results"] else 0
    avg_cer = sum(r["cer"] for r in log_data["results"]) / len(log_data["results"]) if log_data["results"] else 0
    
    # Summary
    summary = {
        "total_samples": len(test_data),
        "overall_wer": float(overall_wer),
        "overall_cer": float(overall_cer),
        "average_wer": float(avg_wer),
        "average_cer": float(avg_cer),
        "perfect_matches": perfect_matches,
        "perfect_match_rate": float(perfect_match_rate),
        "worst_samples": sorted(log_data["results"], key=lambda x: x["wer"], reverse=True)[:10],
        "best_samples": sorted(log_data["results"], key=lambda x: x["wer"])[:10]
    }
    
    log_data["summary"] = summary
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Overall WER: {summary['overall_wer']:.4f} ({summary['overall_wer']*100:.2f}%)")
    print(f"Overall CER: {summary['overall_cer']:.4f} ({summary['overall_cer']*100:.2f}%)")
    print(f"Average WER: {summary['average_wer']:.4f} ({summary['average_wer']*100:.2f}%)")
    print(f"Average CER: {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
    print(f"Perfect matches: {summary['perfect_matches']}/{summary['total_samples']} ({summary['perfect_match_rate']*100:.2f}%)")
    
    # Print worst samples
    print("\n" + "=" * 80)
    print("🔴 TOP 10 WORST SAMPLES (Highest WER)")
    print("=" * 80)
    for i, sample in enumerate(summary["worst_samples"][:5], 1):
        print(f"\n{i}. Sample ID: {sample['sample_id']} | WER: {sample['wer']:.4f}")
        print(f"   Reference: {sample['reference']}")
        print(f"   Prediction: {sample['prediction']}")
    
    # Print best samples
    print("\n" + "=" * 80)
    print("🟢 TOP 10 BEST SAMPLES (Lowest WER)")
    print("=" * 80)
    for i, sample in enumerate(summary["best_samples"][:5], 1):
        print(f"\n{i}. Sample ID: {sample['sample_id']} | WER: {sample['wer']:.4f}")
        print(f"   Reference: {sample['reference']}")
        print(f"   Prediction: {sample['prediction']}")
    
    # Save log file
    print(f"\n💾 Saving detailed log to: {output_log}")
    output_path = Path(output_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Evaluation completed!")
    print(f"   Log file: {output_log}")
    print(f"   Total samples: {len(test_data)}")
    print(f"   Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"   Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    
    return log_data


def main():
    parser = argparse.ArgumentParser(
        description="Test Whisper CTranslate2 model trên full test dataset với bf16"
    )
    parser.add_argument(
        "--model",
        default="./models/final/whisper-vi-en-ct2",
        help="Path to Whisper CTranslate2 model"
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
        default="whisper_full_test_bf16_log.json",
        help="Output log file (JSON)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--compute",
        default="float16",
        choices=["float16", "bfloat16", "int8_float16", "int8"],
        help="Compute type (bf16 = bfloat16)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to test (None = all)"
    )
    
    args = parser.parse_args()
    
    # Map bf16 to bfloat16
    if args.compute == "bf16":
        args.compute = "bfloat16"
    
    print("=" * 80)
    print("🚀 WHISPER FULL DATASET TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Output log: {args.output}")
    print(f"Device: {args.device}")
    print(f"Compute: {args.compute}")
    print("=" * 80)
    
    test_model_full(
        model_path=args.model,
        test_dataset_path=args.test_dataset,
        base_audio_dir=args.base_audio_dir,
        output_log=args.output,
        device=args.device,
        compute=args.compute,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

