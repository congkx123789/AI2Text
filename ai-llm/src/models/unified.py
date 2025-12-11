"""
Unified Model Manager - Quản lý và sử dụng Whisper + Qwen cùng lúc
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
from src.config import ASR_MODEL, GEN_MODEL, GEN_MAX_TOKENS
from src.tools.ai2text_bridge import transcribe
from src.llm.infer import generate_text


class UnifiedModelManager:
    """
    Unified manager cho Whisper ASR và Qwen LLM models.
    Quản lý việc load và sử dụng cả hai models cùng lúc.
    """
    
    def __init__(
        self,
        asr_model: Optional[str] = None,
        gen_model: Optional[str] = None,
        asr_device: Optional[str] = None,
        asr_compute: Optional[str] = None
    ):
        """
        Initialize unified model manager
        
        Args:
            asr_model: Whisper model path or size (default: from config)
            gen_model: Qwen model path (default: from config)
            asr_device: Device for ASR ('cuda', 'cpu', 'auto')
            asr_compute: Compute type for ASR ('float16', 'int8')
        """
        self.asr_model = asr_model or ASR_MODEL
        self.gen_model = gen_model or GEN_MODEL
        self.asr_device = asr_device
        self.asr_compute = asr_compute
        
        # Models will be loaded lazily
        self._asr_loaded = False
        self._gen_loaded = False
        
        print(f"[UnifiedModel] Initialized with:")
        print(f"  - ASR Model: {self.asr_model}")
        print(f"  - GEN Model: {self.gen_model}")
    
    def ensure_models_loaded(self):
        """Ensure both models are loaded (lazy loading)"""
        # ASR model is loaded on-demand via transcribe()
        # GEN model is loaded on-demand via generate_text()
        # This is just a placeholder for future explicit loading if needed
        pass
    
    def process_audio(
        self,
        audio_path: str | Path,
        task: str = "summarize",
        question: Optional[str] = None,
        language: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process audio through Whisper → Qwen pipeline
        
        Args:
            audio_path: Path to audio file
            task: Task type - "summarize", "answer", "translate", "analyze", "extract"
            question: Optional question if task is "answer"
            language: Language code for transcription (None = auto-detect)
            max_tokens: Max tokens for generation (default: from config)
        
        Returns:
            Dict with:
                - transcription: Transcribed text
                - response: Generated response from Qwen
                - language: Detected language
                - task: Task performed
        """
        # Step 1: Transcribe với Whisper
        print(f"[UnifiedModel] Transcribing audio: {audio_path}")
        transcribe_result = transcribe(
            str(audio_path),
            size=self.asr_model,
            lang=language,
            device=self.asr_device,
            compute=self.asr_compute
        )
        
        transcription = transcribe_result["text"]
        detected_language = transcribe_result.get("language", "unknown")
        
        if not transcription or not transcription.strip():
            raise ValueError("Transcription is empty")
        
        print(f"[UnifiedModel] Transcription: {transcription[:100]}...")
        
        # Step 2: Process với Qwen
        if task == "answer" and question:
            text_to_process = f"Question: {question}\n\nText: {transcription}"
        else:
            text_to_process = transcription
        
        print(f"[UnifiedModel] Processing with Qwen (task: {task})...")
        response = generate_text(
            text_to_process,
            task=task,
            max_new_tokens=max_tokens or GEN_MAX_TOKENS
        )
        
        print(f"[UnifiedModel] Generated response: {response[:100]}...")
        
        return {
            "transcription": transcription,
            "response": response,
            "language": detected_language,
            "task": task,
            "asr_model": str(self.asr_model),
            "gen_model": str(self.gen_model)
        }
    
    def transcribe_only(
        self,
        audio_path: str | Path,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chỉ transcribe audio (không qua Qwen)
        
        Args:
            audio_path: Path to audio file
            language: Language code (None = auto-detect)
        
        Returns:
            Dict with transcription results
        """
        return transcribe(
            str(audio_path),
            size=self.asr_model,
            lang=language,
            device=self.asr_device,
            compute=self.asr_compute
        )
    
    def generate_only(
        self,
        text: str,
        task: str = "summarize",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chỉ generate text với Qwen (không cần audio)
        
        Args:
            text: Input text
            task: Task type
            max_tokens: Max tokens for generation
        
        Returns:
            Generated text
        """
        return generate_text(
            text,
            task=task,
            max_new_tokens=max_tokens or GEN_MAX_TOKENS
        )


# Global singleton instance
_unified_manager: Optional[UnifiedModelManager] = None


def get_unified_manager(
    asr_model: Optional[str] = None,
    gen_model: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute: Optional[str] = None
) -> UnifiedModelManager:
    """
    Get or create global unified model manager instance
    
    Args:
        asr_model: Whisper model path or size
        gen_model: Qwen model path
        asr_device: Device for ASR
        asr_compute: Compute type for ASR
    
    Returns:
        UnifiedModelManager instance
    """
    global _unified_manager
    
    if _unified_manager is None:
        _unified_manager = UnifiedModelManager(
            asr_model=asr_model,
            gen_model=gen_model,
            asr_device=asr_device,
            asr_compute=asr_compute
        )
    elif asr_model or gen_model or asr_device or asr_compute:
        # Recreate if different config requested
        _unified_manager = UnifiedModelManager(
            asr_model=asr_model or _unified_manager.asr_model,
            gen_model=gen_model or _unified_manager.gen_model,
            asr_device=asr_device or _unified_manager.asr_device,
            asr_compute=asr_compute or _unified_manager.asr_compute
        )
    
    return _unified_manager


def reset_unified_manager():
    """Reset global unified manager (useful for testing)"""
    global _unified_manager
    _unified_manager = None

