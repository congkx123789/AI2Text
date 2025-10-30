import subprocess, sys
raise SystemExit(subprocess.call([sys.executable, "-m", "uvicorn", "src.asr.api:app", "--host", "127.0.0.1", "--port", "8001", "--reload"]))
