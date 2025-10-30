from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


_DEF = "Qwen2.5-0.5B-Instruct"


def load_llm(name: str | None = None):
    name = name or _DEF
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    return tok, model