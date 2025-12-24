# AI2Text demo_package (Standalone)

Folder này là **đóng kín** và có thể chạy độc lập trên máy khác.

## Chạy nhanh

```bash
cd demo_package
pip install -r requirements.txt

# Vietnamese
python run_demo.py audio.wav vi

# English
python run_demo.py audio.wav en

# Auto-detect
python run_demo.py audio.wav
```

## Files cần thiết (tối thiểu)

- `best_model.pt`: checkpoint
- `run_demo.py`: script chạy inference
- `requirements.txt`: dependencies
- `configs/default.yaml`: config audio/model
- `models/tokenizer_vi_en_3500.model` + `models/tokenizer_vi_en_3500.vocab`: tokenizer
- `ai2text_src/`: source code đã bundle để chạy độc lập

