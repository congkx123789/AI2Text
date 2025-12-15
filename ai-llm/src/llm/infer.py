from __future__ import annotations
from typing import List, Tuple, Optional
from transformers import TextStreamer
from .load import load_llm
from .gemini import generate_with_gemini, generate_with_citations_gemini
from src.config import LLM_PROVIDER


_tok, _model = None, None


def _ensure():
    global _tok, _model
    if _tok is None:
        _tok, _model = load_llm()




def generate_with_citations(
    prompt: str,
    hits: List[Tuple[str, str, float]],
    provider: Optional[str] = None
):
    """
    Generate answer with citations using LLM (Qwen or Gemini)
    
    Args:
        prompt: Question to answer
        hits: List of (id, text, score) tuples
        provider: LLM provider - "qwen" or "gemini". Default: from config
    
    Returns:
        Tuple of (answer_text, citations_list)
    """
    use_provider = provider or LLM_PROVIDER
    
    # Use Gemini API if requested
    if use_provider == "gemini":
        return generate_with_citations_gemini(prompt, hits)
    
    # Default: Use Qwen (local)
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
    question: Optional[str] = None,
    max_new_tokens: int = 400,
    provider: Optional[str] = None
) -> str:
    """
    Generate text using LLM (Qwen local or Gemini API) based on input text.
    
    Args:
        text: Input text to process
        task: Task type - "summarize", "answer", "translate", "analyze", etc.
        question: Optional question if task is "answer"
        max_new_tokens: Maximum tokens to generate
        provider: LLM provider - "qwen" (local) or "gemini" (API). Default: from config
    
    Returns:
        Generated text response
    """
    # Determine provider
    use_provider = provider or LLM_PROVIDER
    
    # Use Gemini API if requested
    if use_provider == "gemini":
        return generate_with_gemini(
            text=text,
            task=task,
            question=question,
            max_tokens=max_new_tokens
        )
    
    # Default: Use Qwen (local)
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
    
    if task == "answer" and question:
        full_prompt = f"{prompt_prefix}Question: {question}\n\nText: {text}\n\nResponse:"
    else:
        full_prompt = f"{prompt_prefix}{text}\n\nResponse:"
    
    ids = _tok([full_prompt], return_tensors="pt").to(_model.device)
    out = _model.generate(**ids, max_new_tokens=max_new_tokens)
    response = _tok.decode(out[0], skip_special_tokens=True)
    
    # Extract response part (after "Response:")
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    
    return response