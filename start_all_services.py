#!/usr/bin/env python3
"""
Multi-Model ASR Services Manager
Start/Stop/Status cho tất cả services: ai-llm-ss, ai-llm, AI2Text, và frontend
"""

import os
import sys
import subprocess
import time
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional
import webbrowser
from threading import Thread

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Service configurations
SERVICES = {
    'ai-llm-ss': {
        'name': 'ai-llm-ss',
        'port': 8001,
        'path': PROJECT_ROOT / 'ai-llm-ss',
        'start_cmd': [sys.executable, '-m', 'uvicorn', 'src.asr.api:app', '--host', '0.0.0.0', '--port', '8001'],
        'health_url': 'http://localhost:8001/health',
        'description': 'CTC ASR Model (Port 8001)'
    },
    'ai-llm': {
        'name': 'ai-llm',
        'port': 8000,
        'path': PROJECT_ROOT / 'ai-llm',
        'start_cmd': [sys.executable, '-m', 'uvicorn', 'src.api.server:app', '--host', '0.0.0.0', '--port', '8000'],
        'health_url': 'http://localhost:8000/health',
        'description': 'Whisper + RAG Model (Port 8000)'
    },
    'ai2text': {
        'name': 'ai2text',
        'port': 8002,
        'path': PROJECT_ROOT / 'AI2Text',
        'start_cmd': [sys.executable, '-m', 'uvicorn', 'api.app:app', '--host', '0.0.0.0', '--port', '8002'],
        'health_url': 'http://localhost:8002/health',
        'description': 'Bilingual ASR Model (Port 8002)'
    },
    'frontend': {
        'name': 'frontend',
        'port': 8080,
        'path': PROJECT_ROOT,
        'start_cmd': [sys.executable, '-m', 'http.server', '8080'],
        'health_url': 'http://localhost:8080/frontend/index.html',
        'description': 'Frontend Web Server (Port 8080)'
    }
}

# Store running processes
running_processes: Dict[str, subprocess.Popen] = {}
process_logs: Dict[str, List[str]] = {}


def print_colored(message: str, color: str = Colors.RESET):
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")


def check_port(port: int) -> bool:
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def check_service_health(health_url: str) -> bool:
    """Check if service is healthy"""
    try:
        import urllib.request
        response = urllib.request.urlopen(health_url, timeout=2)
        return response.getcode() == 200
    except:
        return False


def find_python_executable(service_path: Path) -> str:
    """Find Python executable in virtual environment or use system Python"""
    # Check for .venv first
    venv_python = service_path / '.venv' / 'bin' / 'python'
    if venv_python.exists():
        return str(venv_python)
    
    # Check for venv
    venv_python = service_path / 'venv' / 'bin' / 'python'
    if venv_python.exists():
        return str(venv_python)
    
    # Use system Python
    return sys.executable


def start_service(service_name: str, service_config: Dict) -> Optional[subprocess.Popen]:
    """Start a service"""
    name = service_config['name']
    port = service_config['port']
    path = service_config['path']
    cmd = service_config['start_cmd'].copy()
    
    # Check if port is already in use
    if check_port(port):
        print_colored(f"⚠️  Port {port} is already in use for {name}", Colors.YELLOW)
        # Try to check if it's our service
        if check_service_health(service_config['health_url']):
            print_colored(f"✓ Service {name} is already running on port {port}", Colors.GREEN)
            return None
        else:
            print_colored(f"✗ Port {port} is occupied by another service", Colors.RED)
            return None
    
    # Change to service directory
    if not path.exists():
        print_colored(f"✗ Path not found: {path}", Colors.RED)
        return None
    
    # Find Python executable (use venv if available)
    python_exe = find_python_executable(path)
    if cmd[0] == sys.executable or cmd[0] == 'python':
        cmd[0] = python_exe
    
    print_colored(f"🚀 Starting {name} on port {port}...", Colors.CYAN)
    print_colored(f"   Using Python: {python_exe}", Colors.CYAN)
    
    # Create log file
    log_file = PROJECT_ROOT / f"{name}.log"
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            cwd=str(path),
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        # Wait a bit to check if process started successfully
        time.sleep(2)
        
        if process.poll() is not None:
            print_colored(f"✗ {name} failed to start (exit code: {process.returncode})", Colors.RED)
            print_colored(f"   Check log: {log_file}", Colors.YELLOW)
            return None
        
        running_processes[name] = process
        print_colored(f"✓ {name} started (PID: {process.pid})", Colors.GREEN)
        
        # Wait a bit more and check health
        time.sleep(3)
        if check_service_health(service_config['health_url']):
            print_colored(f"✓ {name} is healthy", Colors.GREEN)
        else:
            print_colored(f"⚠️  {name} started but health check failed", Colors.YELLOW)
        
        return process
        
    except Exception as e:
        print_colored(f"✗ Error starting {name}: {e}", Colors.RED)
        return None


def stop_service(service_name: str):
    """Stop a service"""
    if service_name not in running_processes:
        print_colored(f"⚠️  Service {service_name} is not running", Colors.YELLOW)
        return
    
    process = running_processes[service_name]
    
    try:
        print_colored(f"🛑 Stopping {service_name}...", Colors.YELLOW)
        
        if os.name == 'nt':
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        # Wait for process to terminate
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            if os.name == 'nt':
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        
        del running_processes[service_name]
        print_colored(f"✓ {service_name} stopped", Colors.GREEN)
        
    except Exception as e:
        print_colored(f"✗ Error stopping {service_name}: {e}", Colors.RED)


def stop_all_services():
    """Stop all running services"""
    print_colored("\n🛑 Stopping all services...", Colors.YELLOW)
    for service_name in list(running_processes.keys()):
        stop_service(service_name)
    print_colored("✓ All services stopped", Colors.GREEN)


def show_status():
    """Show status of all services"""
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("📊 Service Status", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    
    for name, config in SERVICES.items():
        port = config['port']
        health_url = config['health_url']
        is_running = name in running_processes
        is_healthy = check_service_health(health_url) if is_running else False
        
        status_icon = "✓" if (is_running and is_healthy) else "✗"
        status_color = Colors.GREEN if (is_running and is_healthy) else Colors.RED
        status_text = "Running & Healthy" if (is_running and is_healthy) else "Not Running"
        
        print_colored(f"\n{status_icon} {config['description']}", status_color)
        print(f"   Port: {port}")
        print(f"   Status: {status_text}")
        if is_running:
            print(f"   PID: {running_processes[name].pid}")
        print(f"   URL: {health_url}")
    
    print_colored("\n" + "="*60 + "\n", Colors.CYAN)


def start_all_services():
    """Start all services"""
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("🚀 Starting All Services", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    
    # Start services in order
    for name, config in SERVICES.items():
        start_service(name, config)
        time.sleep(2)  # Small delay between services
    
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("✓ All services started!", Colors.GREEN)
    print_colored("="*60 + "\n", Colors.CYAN)
    
    # Show status
    show_status()
    
    # Open browser after a delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:8080/frontend/index.html')
    
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print_colored("🌐 Frontend will open in browser automatically", Colors.CYAN)
    print_colored("   Or visit: http://localhost:8080/frontend/index.html\n", Colors.CYAN)


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print_colored("\n\n⚠️  Interrupted! Stopping all services...", Colors.YELLOW)
    stop_all_services()
    sys.exit(0)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Model ASR Services Manager')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'], 
                       help='Action to perform')
    parser.add_argument('--service', choices=list(SERVICES.keys()),
                       help='Specific service to start/stop (optional)')
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.action == 'start':
        if args.service:
            start_service(args.service, SERVICES[args.service])
        else:
            start_all_services()
            # Keep running
            try:
                print_colored("Press Ctrl+C to stop all services\n", Colors.YELLOW)
                while True:
                    time.sleep(1)
                    # Check if any process died
                    for name, process in list(running_processes.items()):
                        if process.poll() is not None:
                            print_colored(f"⚠️  {name} process died (exit code: {process.returncode})", Colors.YELLOW)
                            del running_processes[name]
            except KeyboardInterrupt:
                signal_handler(None, None)
    
    elif args.action == 'stop':
        if args.service:
            stop_service(args.service)
        else:
            stop_all_services()
    
    elif args.action == 'status':
        show_status()
    
    elif args.action == 'restart':
        if args.service:
            stop_service(args.service)
            time.sleep(2)
            start_service(args.service, SERVICES[args.service])
        else:
            stop_all_services()
            time.sleep(2)
            start_all_services()
            try:
                print_colored("Press Ctrl+C to stop all services\n", Colors.YELLOW)
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                signal_handler(None, None)


if __name__ == '__main__':
    main()

