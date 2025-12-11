from __future__ import annotations
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


_DEF = "Qwen2.5-0.5B-Instruct"


def _is_peft_model(model_path: str | Path) -> bool:
    """Check if model path contains LoRA adapter"""
    path = Path(model_path)
    return (path / "adapter_config.json").exists()


def load_llm(name: str | None = None):
    """
    Load LLM model, automatically detect and load LoRA adapter if present.
    
    Args:
        name: Model name or path (default: from config)
    
    Returns:
        Tuple of (tokenizer, model)
    """
    from src.config import GEN_MODEL
    
    name = name or GEN_MODEL
    model_path = Path(name)
    
    # Check if this is a LoRA adapter
    if _is_peft_model(model_path):
        print(f"[LLM] Detected LoRA adapter at: {model_path}")
        print(f"[LLM] Loading base model + LoRA adapter...")
        
        try:
            from peft import PeftModel, PeftConfig
            
            # Load adapter config to get base model
            config = PeftConfig.from_pretrained(str(model_path))
            base_model_name = config.base_model_name_or_path
            print(f"[LLM] Base model: {base_model_name}")
            
            # Load tokenizer
            tok = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, str(model_path))
            model = model.merge_and_unload()  # Merge adapter into base model for faster inference
            
            print(f"[LLM] LoRA adapter loaded and merged successfully")
            return tok, model
            
        except ImportError:
            print(f"[LLM] Warning: PEFT not installed. Install with: pip install peft")
            print(f"[LLM] Falling back to base model only")
            # Fallback: try to load base model from adapter config
            try:
                import json
                with open(model_path / "adapter_config.json") as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path", _DEF)
                    print(f"[LLM] Loading base model: {base_model_name}")
                    tok = AutoTokenizer.from_pretrained(base_model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto"
                    )
                    return tok, model
            except Exception as e:
                print(f"[LLM] Error loading base model: {e}")
                raise
        except Exception as e:
            print(f"[LLM] Error loading LoRA adapter: {e}")
            raise
    else:
        # Regular model (not LoRA adapter)
        print(f"[LLM] Loading model: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tok, model