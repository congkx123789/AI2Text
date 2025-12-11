"""
Script để merge/average weights giữa base Whisper và fine-tuned Whisper

Có thể merge theo các cách:
1. Weight averaging (interpolation)
2. Copy fine-tuned weights (default - fine-tuned đã là full model)
"""
import argparse
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import json


def merge_whisper_models(
    base_model_path: str,
    finetuned_model_path: str,
    output_path: str,
    merge_alpha: float = 1.0
):
    """
    Merge base Whisper và fine-tuned Whisper models
    
    Args:
        base_model_path: Path to base Whisper model (e.g., "openai/whisper-small")
        finetuned_model_path: Path to fine-tuned Whisper model
        output_path: Path to save merged model
        merge_alpha: Interpolation factor (1.0 = use fine-tuned only, 0.0 = use base only, 0.5 = average)
    """
    base_model_path = Path(base_model_path) if Path(base_model_path).exists() else base_model_path
    finetuned_model_path = Path(finetuned_model_path)
    output_path = Path(output_path)
    
    if not finetuned_model_path.exists():
        raise ValueError(f"Fine-tuned model path not found: {finetuned_model_path}")
    
    print(f"📦 Loading base model: {base_model_path}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float32  # Use float32 for merging
    )
    
    print(f"📦 Loading fine-tuned model: {finetuned_model_path}")
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(
        str(finetuned_model_path),
        torch_dtype=torch.float32
    )
    
    # Get state dicts
    base_state = base_model.state_dict()
    finetuned_state = finetuned_model.state_dict()
    
    print(f"🔀 Merging models with alpha={merge_alpha}...")
    print(f"   (alpha=1.0: use fine-tuned only, alpha=0.5: average, alpha=0.0: use base only)")
    
    # Merge weights
    merged_state = {}
    for key in finetuned_state.keys():
        if key in base_state:
            if merge_alpha == 1.0:
                # Use fine-tuned weights only
                merged_state[key] = finetuned_state[key].clone()
            elif merge_alpha == 0.0:
                # Use base weights only
                merged_state[key] = base_state[key].clone()
            else:
                # Interpolate
                merged_state[key] = (
                    merge_alpha * finetuned_state[key] + 
                    (1 - merge_alpha) * base_state[key]
                )
        else:
            # Key only in fine-tuned model
            merged_state[key] = finetuned_state[key].clone()
    
    # Load merged state into base model
    base_model.load_state_dict(merged_state)
    
    # Save merged model
    print(f"💾 Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_model.save_pretrained(str(output_path))
    
    # Copy processor/tokenizer from fine-tuned model
    print("🔤 Copying processor/tokenizer...")
    processor = WhisperProcessor.from_pretrained(str(finetuned_model_path))
    processor.save_pretrained(str(output_path))
    
    # Update config to indicate merged model
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        config["_merged_from"] = {
            "base_model": str(base_model_path),
            "finetuned_model": str(finetuned_model_path),
            "merge_alpha": merge_alpha
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"✅ Merged model saved successfully!")
    print(f"   - Model: {output_path}")
    print(f"   - Merge alpha: {merge_alpha}")
    print(f"   - Update ASR_MODEL in .env to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge base Whisper and fine-tuned Whisper models"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base Whisper model (path or HuggingFace model ID, e.g., 'openai/whisper-small')"
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="Path to fine-tuned Whisper model"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for merged model"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Merge interpolation factor (1.0=use fine-tuned only, 0.5=average, 0.0=use base only)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔀 MERGE WHISPER MODELS")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned model: {args.finetuned_model}")
    print(f"Output: {args.output}")
    print(f"Alpha: {args.alpha}")
    print("=" * 80)
    
    try:
        merge_whisper_models(
            base_model_path=args.base_model,
            finetuned_model_path=args.finetuned_model,
            output_path=args.output,
            merge_alpha=args.alpha
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

