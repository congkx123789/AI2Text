#!/usr/bin/env python3
"""
Test script for Gemini API integration
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.gemini import generate_with_gemini
from src.models.unified import get_unified_manager

def test_gemini_direct():
    """Test Gemini API directly"""
    print("=" * 60)
    print("Testing Gemini API Direct Integration")
    print("=" * 60)
    
    test_text = """
    Artificial Intelligence (AI) has revolutionized many industries in recent years.
    From healthcare to finance, AI technologies are being used to solve complex problems
    and improve efficiency. Machine learning, a subset of AI, enables computers to learn
    from data without being explicitly programmed.
    """
    
    print(f"\nInput text: {test_text[:100]}...")
    
    try:
        # Test summarize
        print("\n[1] Testing summarize task...")
        result = generate_with_gemini(test_text, task="summarize")
        print(f"✓ Summarize result: {result[:200]}...")
        
        # Test analyze
        print("\n[2] Testing analyze task...")
        result = generate_with_gemini(test_text, task="analyze")
        print(f"✓ Analyze result: {result[:200]}...")
        
        # Test answer
        print("\n[3] Testing answer task...")
        result = generate_with_gemini(
            test_text,
            task="answer",
            question="What is machine learning?"
        )
        print(f"✓ Answer result: {result[:200]}...")
        
        print("\n✅ All Gemini direct tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_unified_manager_gemini():
    """Test UnifiedModelManager with Gemini"""
    print("\n" + "=" * 60)
    print("Testing UnifiedModelManager with Gemini")
    print("=" * 60)
    
    # Check if we have a test audio file
    test_audio = Path("data/processed/full_merged_dataset/train/audio")
    if not test_audio.exists():
        print("⚠️  No test audio directory found. Skipping audio test.")
        return True
    
    # Find first audio file
    audio_files = list(test_audio.glob("*.wav"))
    if not audio_files:
        print("⚠️  No audio files found. Skipping audio test.")
        return True
    
    test_file = audio_files[0]
    print(f"\nUsing test audio: {test_file}")
    
    try:
        manager = get_unified_manager(llm_provider="gemini")
        
        print("\n[1] Testing transcribe_only...")
        transcription = manager.transcribe_only(test_file)
        print(f"✓ Transcription: {transcription['text'][:100]}...")
        
        print("\n[2] Testing process_audio with summarize...")
        result = manager.process_audio(
            audio_path=test_file,
            task="summarize"
        )
        print(f"✓ Transcription: {result['transcription'][:100]}...")
        print(f"✓ Summary: {result['response'][:200]}...")
        
        print("\n[3] Testing generate_only...")
        text = transcription['text']
        summary = manager.generate_only(
            text=text[:500],  # Limit text length
            task="summarize"
        )
        print(f"✓ Generated summary: {summary[:200]}...")
        
        print("\n✅ All UnifiedModelManager tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Gemini Integration Tests\n")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment!")
        print("   Please set it in .env file or export it:")
        print("   export GEMINI_API_KEY=your_api_key")
        sys.exit(1)
    
    print(f"✓ Found GEMINI_API_KEY: {api_key[:20]}...")
    
    # Run tests
    success1 = test_gemini_direct()
    success2 = test_unified_manager_gemini()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("=" * 60)
        sys.exit(1)

