#!/usr/bin/env python3
"""
Simple wrapper script for running ASR inference.
Usage: python3 run.py <audio_file> [language]

Examples:
    python3 run.py audio.wav vi
    python3 run.py audio.wav en
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = Path(__file__).parent
    demo_script = script_dir / "demo_inference.py"
    
    # Pass all arguments to demo_inference.py
    cmd = [sys.executable, str(demo_script)] + sys.argv[1:]
    subprocess.run(cmd)

