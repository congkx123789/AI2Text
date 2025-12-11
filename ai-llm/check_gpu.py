"""
Check GPU và cuDNN cho CTranslate2
"""
from faster_whisper import WhisperModel

try:
    # Thử load model lên GPU
    print("🔍 Testing GPU với faster-whisper...")
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("✅ GPU WORKS! CTranslate2 đã nhận cuDNN.")
    
    # Test transcribe
    print("🔍 Testing transcribe...")
    import numpy as np
    audio = np.random.randn(16000).astype(np.float32)
    segments, info = model.transcribe(audio, beam_size=1)
    print(f"✅ Transcribe works! Language: {info.language}")
    
except Exception as e:
    print(f"❌ Vẫn lỗi: {e}")
    import traceback
    traceback.print_exc()

