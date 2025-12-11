"""
Script để merge LoRA adapter vào base model

Sau khi merge, bạn sẽ có một full model (không cần PEFT) để inference nhanh hơn.
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def merge_lora_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str = None
):
    """
    Merge LoRA adapter vào base model
    
    Args:
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        base_model: Base model name (auto-detect from adapter_config.json if None)
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    
    if not adapter_path.exists():
        raise ValueError(f"Adapter path not found: {adapter_path}")
    
    if not (adapter_path / "adapter_config.json").exists():
        raise ValueError(f"Not a LoRA adapter: {adapter_path}")
    
    print(f"📦 Loading LoRA adapter from: {adapter_path}")
    
    # Load adapter config
    config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = base_model or config.base_model_name_or_path
    
    print(f"📦 Base model: {base_model_name}")
    print(f"📦 Merging adapter into base model...")
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model
    print("🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load and merge LoRA adapter
    print("🔗 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    print("🔀 Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"💾 Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print(f"✅ Merged model saved successfully!")
    print(f"   - Model: {output_path}")
    print(f"   - You can now use this model without PEFT")
    print(f"   - Update GEN_MODEL in .env to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for merged model"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name (auto-detect from adapter_config.json if not provided)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔀 MERGE LORA ADAPTER INTO BASE MODEL")
    print("=" * 80)
    
    try:
        merge_lora_adapter(
            adapter_path=args.adapter,
            output_path=args.output,
            base_model=args.base_model
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

