#!/usr/bin/env python3
"""
Test script for the ASR API.
Usage: python scripts/test_api.py [audio_file.wav]
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8001"

def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  Device: {data['device']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Vocab size: {data['vocab_size']}")
        return data['model_loaded']
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        print(f"  Start it with: python scripts/serve_asr.py")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting /model/info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Model info retrieved")
        print(f"  Model path: {data['model_path']}")
        print(f"  Model exists: {data['model_exists']}")
        print(f"  Device: {data['device']}")
        print(f"  Total parameters: {data['total_parameters']:,}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False

def test_transcribe(audio_file):
    """Test the transcription endpoint"""
    print(f"\nTesting /transcribe endpoint with {audio_file}...")
    if not Path(audio_file).exists():
        print(f"✗ Audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            response = requests.post(f"{API_URL}/transcribe", files=files)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Transcription successful")
            print(f"  Text: {data['text']}")
            if 'duration' in data:
                print(f"  Duration: {data['duration']:.2f}s")
            return True
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"  Error detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"  Error: {e.response.text}")
        return False

def main():
    print("=" * 60)
    print("ASR API Test Script")
    print("=" * 60)
    
    # Test health
    if not test_health():
        sys.exit(1)
    
    # Test model info
    test_model_info()
    
    # Test transcription if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_transcribe(audio_file)
    else:
        print("\n" + "=" * 60)
        print("No audio file provided. To test transcription:")
        print(f"  python {sys.argv[0]} <audio_file.wav>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} data/processed/merged_dataset/train/audio/*.wav")
    
    print("\n" + "=" * 60)
    print("API Documentation: http://127.0.0.1:8001/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()

