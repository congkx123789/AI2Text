"""
Example: How to use AI-LLM API from Python
"""
from src.api.client import AILLMClient

def main():
    # Initialize client
    client = AILLMClient(base_url="http://localhost:8000")
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    print(f"Index available: {health['index_available']}\n")
    
    # Example 1: Transcribe audio file (file must be on server)
    print("Example 1: Transcribe audio from server path")
    try:
        result = client.transcribe_file("data/raw/audio/why-hello-there-103596.wav")
        print(f"Transcribed text: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Segments: {len(result['segments'])} segments\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Upload and transcribe local file
    print("Example 2: Upload and transcribe local file")
    try:
        result = client.transcribe_upload("data/raw/audio/i-dont-like-you-87027.wav")
        print(f"Transcribed text: {result['text']}")
        print(f"Language: {result['language']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Ask a question using RAG
    print("Example 3: Ask a question")
    try:
        result = client.ask("What is the main topic discussed in the transcripts?")
        print(f"Answer: {result['answer']}")
        print(f"\nCitations ({len(result['contexts'])} sources):")
        for i, cite in enumerate(result['contexts'], 1):
            print(f"  [{i}] {cite['text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

