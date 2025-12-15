"""
Google Gemini API Integration
"""
from __future__ import annotations
from typing import Optional
import os
import google.generativeai as genai


# Initialize Gemini client
_gemini_client = None
_gemini_model = None


def _init_gemini():
    """Initialize Gemini API client"""
    global _gemini_client, _gemini_model
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    
    # Get model name from env (default: gemini-2.5-flash)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Try model name as-is first (without models/ prefix)
    # Gemini API usually works with just the model name
    try:
        _gemini_model = genai.GenerativeModel(model_name)
        print(f"[Gemini] Initialized with model: {model_name}")
    except Exception as e:
        # Fallback: try with models/ prefix
        model_name_with_prefix = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        try:
            print(f"[Gemini] Trying with models/ prefix: {model_name_with_prefix}")
            _gemini_model = genai.GenerativeModel(model_name_with_prefix)
            print(f"[Gemini] Initialized with model: {model_name_with_prefix}")
        except Exception as e2:
            raise RuntimeError(f"Failed to initialize Gemini model '{model_name}': {e2}. Try checking available models.")


def generate_with_gemini(
    text: str,
    task: str = "summarize",
    question: Optional[str] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate text using Google Gemini API
    
    Args:
        text: Input text to process
        task: Task type - "summarize", "answer", "translate", "analyze", "extract"
        question: Optional question if task is "answer"
        max_tokens: Maximum tokens to generate (Gemini uses max_output_tokens)
    
    Returns:
        Generated text response
    """
    global _gemini_model
    
    if _gemini_model is None:
        _init_gemini()
    
    # Create prompt based on task
    task_prompts = {
        "summarize": "Summarize the following text concisely:\n\n",
        "answer": "Based on the following text, provide a detailed answer to the question:\n\n",
        "translate": "Translate the following text to English:\n\n",
        "analyze": "Analyze the following text and provide key insights:\n\n",
        "extract": "Extract key information from the following text:\n\n",
    }
    
    prompt_prefix = task_prompts.get(task, f"Process the following text ({task}):\n\n")
    
    if task == "answer" and question:
        full_prompt = f"{prompt_prefix}Question: {question}\n\nText: {text}\n\nResponse:"
    else:
        full_prompt = f"{prompt_prefix}{text}\n\nResponse:"
    
    # Configure generation parameters
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    if max_tokens:
        generation_config["max_output_tokens"] = max_tokens
    
    try:
        response = _gemini_model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Extract text from response
        if response.text:
            return response.text.strip()
        else:
            raise ValueError("Empty response from Gemini API")
            
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")


def generate_with_citations_gemini(
    prompt: str,
    contexts: list[tuple[str, str, float]]
) -> tuple[str, list[dict]]:
    """
    Generate answer with citations using Gemini API (for RAG)
    
    Args:
        prompt: Question to answer
        contexts: List of (id, text, score) tuples
    
    Returns:
        Tuple of (answer_text, citations_list)
    """
    global _gemini_model
    
    if _gemini_model is None:
        _init_gemini()
    
    # Format contexts with citations
    ctx_text = "\n\n".join([f"[{i+1}] {ctx[1]}" for i, ctx in enumerate(contexts)])
    
    full_prompt = f"""Answer the question using the sources below and cite like [1], [2], etc.

Question: {prompt}

Sources:
{ctx_text}

Answer:"""
    
    try:
        response = _gemini_model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 512,
            }
        )
        
        answer = response.text.strip() if response.text else ""
        citations = [{"id": ctx[0], "text": ctx[1]} for ctx in contexts]
        
        return answer, citations
        
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

