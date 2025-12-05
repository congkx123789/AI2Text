"""
Example: How to use AI-LLM API with plain requests library (no client)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    response.raise_for_status()
    return response.json()

# Transcribe from server path
def transcribe_file(audio_path):
    response = requests.post(
        f"{BASE_URL}/transcribe",
        json={"audio_path": audio_path}
    )
    response.raise_for_status()
    return response.json()

# Upload and transcribe
def transcribe_upload(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'audio/wav')}
        response = requests.post(
            f"{BASE_URL}/transcribe/upload",
            files=files
        )
    response.raise_for_status()
    return response.json()

# Ask question
def ask(query, top_k=5):
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"query": query, "top_k": top_k}
    )
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # Example usage
    print("Health check:", check_health())
    
    # Transcribe
    result = transcribe_file("data/raw/audio/why-hello-there-103596.wav")
    print(f"\nTranscribed: {result['text']}")
    
    # Ask question
    answer = ask("What is discussed in the transcripts?")
    print(f"\nAnswer: {answer['answer']}")
    print(f"Found {len(answer['contexts'])} citations")

