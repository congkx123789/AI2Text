"""
Python client library for AI-LLM API
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AILLMClient:
    """Client for interacting with AI-LLM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize client
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds (default: 300 for transcription)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file from server path
        
        Args:
            audio_path: Path to audio file on the server
            
        Returns:
            Dict with 'text', 'segments', and 'language' keys
        """
        response = self.session.post(
            f"{self.base_url}/transcribe",
            json={"audio_path": audio_path},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def transcribe_upload(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and transcribe audio file
        
        Args:
            file_path: Local path to audio file
            
        Returns:
            Dict with 'text', 'segments', and 'language' keys
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'audio/wav')}
            response = self.session.post(
                f"{self.base_url}/transcribe/upload",
                files=files,
                timeout=self.timeout
            )
        response.raise_for_status()
        return response.json()
    
    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question using RAG pipeline
        
        Args:
            query: Your question
            top_k: Number of top results to retrieve (default: 5)
            
        Returns:
            Dict with 'answer' and 'contexts' keys
        """
        response = self.session.post(
            f"{self.base_url}/ask",
            json={"query": query, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()


# Convenience function for quick usage
def create_client(base_url: str = "http://localhost:8000") -> AILLMClient:
    """Create and return an AILLMClient instance"""
    return AILLMClient(base_url=base_url)

