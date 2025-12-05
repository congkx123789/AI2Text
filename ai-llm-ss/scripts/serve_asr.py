#!/usr/bin/env python3
"""
Start the ASR API server.
Usage: python scripts/serve_asr.py [--host HOST] [--port PORT] [--reload]
"""
import subprocess
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Start ASR API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to (default: 8001)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    args = parser.parse_args()
    
    # Set PYTHONPATH to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", project_root)
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.asr.api:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    if args.reload:
        cmd.append("--reload")
    else:
        cmd.extend(["--workers", str(args.workers)])
    
    print(f"Starting ASR API server on http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    
    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()
