from __future__ import annotations
from typing import List, Tuple
from transformers import TextStreamer
from .load import load_llm


_tok, _model = None, None


def _ensure():
    global _tok, _model
    if _tok is None:
        _tok, _model = load_llm()




def generate_with_citations(prompt: str, hits: List[Tuple[str, str, float]]):
    _ensure()
    ctx = "\n\n".join([f"[{i+1}] {h[1]}" for i, h in enumerate(hits)])
    full = f"Answer the question using the sources and cite like [1], [2].\n\nQuestion: {prompt}\n\nSources:\n{ctx}\n\nAnswer:"
    ids = _tok([full], return_tensors="pt").to(_model.device)
    out = _model.generate(**ids, max_new_tokens=400)
    text = _tok.decode(out[0], skip_special_tokens=True)
    answer = text.split("Answer:")[-1].strip()
    cites = [{"id": hits[i][0], "text": hits[i][1]} for i in range(len(hits))]
    return answer, cites


def generate_text(
    text: str,
    task: str = "summarize",
    max_new_tokens: int = 400
) -> str:
    """
    Generate text using Qwen LLM based on input text.
    
    Args:
        text: Input text to process
        task: Task type - "summarize", "answer", "translate", "analyze", etc.
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text response
    """
    _ensure()
    
    # Create prompt based on task
    task_prompts = {
        "summarize": "Summarize the following text:\n\n",
        "answer": "Based on the following text, provide a detailed answer:\n\n",
        "translate": "Translate the following text to English:\n\n",
        "analyze": "Analyze the following text and provide insights:\n\n",
        "extract": "Extract key information from the following text:\n\n",
    }
    
    prompt_prefix = task_prompts.get(task, f"Process the following text ({task}):\n\n")
    full_prompt = f"{prompt_prefix}{text}\n\nResponse:"
    
    ids = _tok([full_prompt], return_tensors="pt").to(_model.device)
    out = _model.generate(**ids, max_new_tokens=max_new_tokens)
    response = _tok.decode(out[0], skip_special_tokens=True)
    
    # Extract response part (after "Response:")
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    
    return response