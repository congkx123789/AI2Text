"""
Example client for ASR API.
Demonstrates how to use the API from Python.
"""

import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any


class ASRAPIClient:
    """Client for ASR API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
    
    def load_model(self, checkpoint_path: str, model_name: str = "default", device: str = "cuda") -> Dict[str, Any]:
        """Load a model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Name to assign to the model
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            Response from API
        """
        response = requests.post(
            f"{self.base_url}/load_model",
            json={
                "checkpoint_path": checkpoint_path,
                "model_name": model_name,
                "device": device
            }
        )
        response.raise_for_status()
        return response.json()
    
    def transcribe(self, 
                   audio_path: str,
                   model_name: str = "default",
                   return_timestamps: bool = False,
                   language_id: Optional[int] = None) -> Dict[str, Any]:
        """Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            model_name: Name of loaded model
            return_timestamps: If True, return word-level timestamps
            language_id: Language ID (0=Vietnamese, 1=English)
            
        Returns:
            Transcription result
        """
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            params = {
                "model_name": model_name,
                "return_timestamps": return_timestamps
            }
            if language_id is not None:
                params["language_id"] = language_id
            
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                params=params
            )
            response.raise_for_status()
            return response.json()
    
    def transcribe_batch(self,
                        audio_paths: list,
                        model_name: str = "default",
                        return_timestamps: bool = False,
                        language_id: Optional[int] = None) -> Dict[str, Any]:
        """Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            model_name: Name of loaded model
            return_timestamps: If True, return word-level timestamps
            language_id: Language ID (0=Vietnamese, 1=English)
            
        Returns:
            Batch transcription results
        """
        response = requests.post(
            f"{self.base_url}/transcribe_batch",
            json={
                "audio_paths": audio_paths,
                "return_timestamps": return_timestamps,
                "language_id": language_id
            },
            params={"model_name": model_name}
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models.
        
        Returns:
            List of available models
        """
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health.
        
        Returns:
            Health status
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model.
        
        Args:
            model_name: Name of model to unload
            
        Returns:
            Unload status
        """
        response = requests.delete(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR API Client Example")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model-name", default="default", help="Model name")
    parser.add_argument("--timestamps", action="store_true", help="Return timestamps")
    args = parser.parse_args()
    
    # Create client
    client = ASRAPIClient(base_url=args.url)
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Health: {health}")
    print()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    load_result = client.load_model(args.checkpoint, model_name=args.model_name)
    print(f"Load result: {load_result}")
    print()
    
    # Transcribe
    print(f"Transcribing {args.audio}...")
    result = client.transcribe(
        args.audio,
        model_name=args.model_name,
        return_timestamps=args.timestamps
    )
    
    print("Transcription result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

