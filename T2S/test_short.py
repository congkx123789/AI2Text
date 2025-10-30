from pathlib import Path
from TTS.api import TTS
import soundfile as sf
import numpy as np

# 1️⃣ Đoạn test ngắn khoảng 10 từ
text = "Hello everyone, this is a short speech test."

# 2️⃣ Khởi tạo XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")  # nếu có GPU, nếu không có thì bỏ dòng này

# 3️⃣ Đường dẫn file mẫu (output)
out_wav = Path("short_test.wav")

# 4️⃣ Gọi model synthesize (giọng tiếng Anh, dùng speaker.wav nếu có)
wav = tts.tts(text=text, speaker_wav="speaker_24k_mono.wav", language="en")

# 5️⃣ Ghi ra file WAV
wav = np.asarray(wav, dtype=np.float32).reshape(-1)
sf.write(out_wav, wav, 24000)

print(f"✅ Done! File saved: {out_wav.resolve()}")
