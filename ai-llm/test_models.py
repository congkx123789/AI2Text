"""
Test script cho fine-tuned models (Whisper CTranslate2 + Qwen Merged)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_whisper_model():
    """Test Whisper CTranslate2 model"""
    print("=" * 80)
    print("🧪 TEST 1: Whisper CTranslate2 Model")
    print("=" * 80)
    
    try:
        import os
        # Force CPU if CUDA has issues
        os.environ.setdefault("ASR_DEVICE", "cpu")
        os.environ.setdefault("ASR_COMPUTE", "int8")
        
        from src.tools.ai2text_bridge import transcribe
        from src.config import ASR_MODEL
        
        print(f"📦 Loading Whisper model: {ASR_MODEL}")
        
        # Find test audio file (look in multiple locations)
        test_audio = None
        import glob
        
        search_paths = [
            "data/raw/audio/*.wav",
            "data/processed/full_merged_dataset/train/audio/*.wav",
            "data/processed/full_merged_dataset/val/audio/*.wav",
            "data/processed/full_merged_dataset/test/audio/*.wav",
        ]
        
        for pattern in search_paths:
            files = glob.glob(pattern)
            # Filter out very small files (likely invalid)
            valid_files = [f for f in files if Path(f).stat().st_size > 1000]
            if valid_files:
                test_audio = valid_files[0]
                break
        
        if not test_audio:
            print("⚠️  No valid test audio file found, skipping Whisper test")
            print("   (Audio files must be > 1KB)")
            return False
        
        print(f"🎵 Testing with audio: {test_audio}")
        print(f"   File size: {Path(test_audio).stat().st_size / 1024:.1f} KB")
        
        # Transcribe with CPU fallback
        try:
            result = transcribe(test_audio, device="cpu", compute="int8")
        except Exception as e:
            print(f"⚠️  GPU failed, trying CPU: {e}")
            result = transcribe(test_audio, device="cpu", compute="int8")
        
        print(f"✅ Transcription successful!")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Language: {result.get('language', 'unknown')}")
        print(f"   Segments: {len(result.get('segments', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_model():
    """Test Qwen merged model"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Qwen Merged Model")
    print("=" * 80)
    
    try:
        from src.llm.infer import generate_text
        from src.config import GEN_MODEL
        
        print(f"📦 Loading Qwen model: {GEN_MODEL}")
        
        # Test text
        test_text = "This is a sample text that needs to be summarized. " * 5
        
        print(f"📝 Testing with text: {test_text[:50]}...")
        
        # Generate summary
        summary = generate_text(
            text=test_text,
            task="summarize",
            max_new_tokens=100
        )
        
        print(f"✅ Generation successful!")
        print(f"   Summary: {summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Qwen test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_manager():
    """Test Unified Model Manager"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Unified Model Manager")
    print("=" * 80)
    
    try:
        from src.models.unified import get_unified_manager
        
        print("📦 Initializing Unified Model Manager...")
        manager = get_unified_manager()
        
        print(f"   ASR Model: {manager.asr_model}")
        print(f"   GEN Model: {manager.gen_model}")
        
        # Find test audio (look in multiple locations)
        test_audio = None
        import glob
        
        search_paths = [
            "data/raw/audio/*.wav",
            "data/processed/full_merged_dataset/train/audio/*.wav",
            "data/processed/full_merged_dataset/val/audio/*.wav",
            "data/processed/full_merged_dataset/test/audio/*.wav",
        ]
        
        for pattern in search_paths:
            files = glob.glob(pattern)
            # Filter out very small files (likely invalid)
            valid_files = [f for f in files if Path(f).stat().st_size > 1000]
            if valid_files:
                test_audio = valid_files[0]
                break
        
        if not test_audio:
            print("⚠️  No test audio file found, testing generate_only only")
            # Test generate only
            test_text = "This is a test text for unified manager."
            result = manager.generate_only(test_text, task="summarize")
            print(f"✅ Generate only successful!")
            print(f"   Result: {result[:100]}...")
            return True
        
        print(f"🎵 Testing with audio: {test_audio}")
        
        # Process audio
        result = manager.process_audio(
            audio_path=test_audio,
            task="summarize"
        )
        
        print(f"✅ Unified pipeline successful!")
        print(f"   Transcription: {result['transcription'][:100]}...")
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Language: {result['language']}")
        print(f"   Task: {result['task']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_health():
    """Test API health (if running)"""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: API Health Check")
    print("=" * 80)
    
    try:
        import requests
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running!")
            print(f"   Status: {data.get('status')}")
            print(f"   Models loaded: {data.get('models_loaded')}")
            print(f"   Index available: {data.get('index_available')}")
            return True
        else:
            print(f"⚠️  API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server is not running (this is OK if you're only testing models)")
        return None
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    print("🚀 TESTING FINE-TUNED MODELS")
    print("=" * 80)
    print("Testing:")
    print("  - Whisper CTranslate2 model")
    print("  - Qwen merged model")
    print("  - Unified Model Manager")
    print("  - API health (if running)")
    print("=" * 80)
    print()
    
    results = []
    
    # Test Whisper
    results.append(("Whisper Model", test_whisper_model()))
    
    # Test Qwen
    results.append(("Qwen Model", test_qwen_model()))
    
    # Test Unified Manager
    results.append(("Unified Manager", test_unified_manager()))
    
    # Test API
    api_result = test_api_health()
    if api_result is not None:
        results.append(("API Health", api_result))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        if passed:
            status = "✅ PASS"
        elif passed is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p is True)
    failed = sum(1 for _, p in results if p is False)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {total - passed - failed} skipped")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())

