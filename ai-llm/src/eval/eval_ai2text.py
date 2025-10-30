# from __future__ import annotations
# import argparse
# from pathlib import Path
# from src.tools.ai2text_bridge import AI2TextBridge


# parser = argparse.ArgumentParser()
# parser.add_argument("path", help="Audio or image path")
# parser.add_argument("--lang", default=None)
# args = parser.parse_args()


# bridge = AI2TextBridge()


# p = Path(args.path)
# if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
#     print(bridge.ocr_image(str(p), lang=args.lang or "eng"))
# else:
#     text, segments = bridge.transcribe_audio(str(p), language=args.lang)
#     print(text)