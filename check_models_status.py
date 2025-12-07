#!/usr/bin/env python3
"""
Kiểm tra trạng thái của các ASR models: AI2Text và ai-llm-ss
"""

import requests
import json
import sys
from typing import Dict, Optional
from datetime import datetime

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message: str, color: str = Colors.RESET):
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")

def check_service(url: str, timeout: int = 5) -> Optional[Dict]:
    """Check service health"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException as e:
        return None

def check_ai_llm_ss(base_url: str = "http://localhost:8001") -> Dict:
    """Check ai-llm-ss model status"""
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("📊 ai-llm-ss Model Status", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    
    status = {
        'service': 'ai-llm-ss',
        'base_url': base_url,
        'online': False,
        'health': None,
        'model_info': None,
        'error': None
    }
    
    # Check health
    health_url = f"{base_url}/health"
    print_colored(f"\n🔍 Checking health: {health_url}", Colors.YELLOW)
    health = check_service(health_url)
    
    if health:
        status['online'] = True
        status['health'] = health
        print_colored("✓ Service is online", Colors.GREEN)
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Device: {health.get('device', 'unknown')}")
        print(f"   Model loaded: {health.get('model_loaded', False)}")
        print(f"   Vocab size: {health.get('vocab_size', 'unknown')}")
    else:
        status['online'] = False
        status['error'] = "Service not responding"
        print_colored("✗ Service is offline or not responding", Colors.RED)
        return status
    
    # Check model info
    model_info_url = f"{base_url}/model/info"
    print_colored(f"\n🔍 Checking model info: {model_info_url}", Colors.YELLOW)
    model_info = check_service(model_info_url)
    
    if model_info:
        status['model_info'] = model_info
        print_colored("✓ Model info retrieved", Colors.GREEN)
        print(f"   Model path: {model_info.get('model_path', 'unknown')}")
        print(f"   Model exists: {model_info.get('model_exists', False)}")
        print(f"   Total parameters: {model_info.get('total_parameters', 'unknown'):,}" if model_info.get('total_parameters') else "   Total parameters: unknown")
        print(f"   Model type: {model_info.get('model_type', 'unknown')}")
    else:
        print_colored("⚠️  Could not retrieve model info", Colors.YELLOW)
    
    return status

def check_ai2text(base_url: str = "http://localhost:8002") -> Dict:
    """Check AI2Text model status"""
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("📊 AI2Text Model Status", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    
    status = {
        'service': 'ai2text',
        'base_url': base_url,
        'online': False,
        'health': None,
        'models': None,
        'error': None
    }
    
    # Check health
    health_url = f"{base_url}/health"
    print_colored(f"\n🔍 Checking health: {health_url}", Colors.YELLOW)
    health = check_service(health_url)
    
    if health:
        status['online'] = True
        status['health'] = health
        print_colored("✓ Service is online", Colors.GREEN)
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Device: {health.get('device', 'unknown')}")
        models_loaded = health.get('models_loaded', [])
        print(f"   Models loaded: {len(models_loaded)} model(s)")
        if models_loaded:
            for model in models_loaded:
                print(f"     - {model}")
    else:
        status['online'] = False
        status['error'] = "Service not responding"
        print_colored("✗ Service is offline or not responding", Colors.RED)
        return status
    
    # Check available models
    models_url = f"{base_url}/models"
    print_colored(f"\n🔍 Checking available models: {models_url}", Colors.YELLOW)
    models = check_service(models_url)
    
    if models:
        status['models'] = models
        print_colored("✓ Models list retrieved", Colors.GREEN)
        if isinstance(models, list) and len(models) > 0:
            print(f"   Available checkpoints: {len(models)}")
            for i, model in enumerate(models[:5], 1):  # Show first 5
                print(f"     {i}. {model}")
            if len(models) > 5:
                print(f"     ... and {len(models) - 5} more")
        else:
            print_colored("   ⚠️  No models available", Colors.YELLOW)
    else:
        print_colored("⚠️  Could not retrieve models list", Colors.YELLOW)
    
    return status

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check status of ASR models')
    parser.add_argument('--ai-llm-ss-url', default='http://localhost:8001',
                       help='ai-llm-ss API URL')
    parser.add_argument('--ai2text-url', default='http://localhost:8002',
                       help='AI2Text API URL')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    print_colored("\n" + "="*60, Colors.BOLD + Colors.CYAN)
    print_colored("🔍 Checking ASR Models Status", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check both models
    ai_llm_ss_status = check_ai_llm_ss(args.ai_llm_ss_url)
    ai2text_status = check_ai2text(args.ai2text_url)
    
    # Summary
    print_colored("\n" + "="*60, Colors.CYAN)
    print_colored("📋 Summary", Colors.BOLD + Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    
    print(f"\n{'Service':<20} {'Status':<15} {'Details'}")
    print("-" * 60)
    
    # ai-llm-ss
    if ai_llm_ss_status['online']:
        model_loaded = ai_llm_ss_status['health'].get('model_loaded', False) if ai_llm_ss_status['health'] else False
        status_text = "✓ Online" if model_loaded else "⚠️  Online (no model)"
        print_colored(f"{'ai-llm-ss':<20} {status_text:<15}", Colors.GREEN if model_loaded else Colors.YELLOW)
        if ai_llm_ss_status['model_info']:
            params = ai_llm_ss_status['model_info'].get('total_parameters', 0)
            print(f"   Parameters: {params:,}" if params else "   Parameters: unknown")
    else:
        print_colored(f"{'ai-llm-ss':<20} {'✗ Offline':<15}", Colors.RED)
    
    # AI2Text
    if ai2text_status['online']:
        models_count = len(ai2text_status['health'].get('models_loaded', [])) if ai2text_status['health'] else 0
        status_text = f"✓ Online ({models_count} models)" if models_count > 0 else "⚠️  Online (no models)"
        print_colored(f"{'AI2Text':<20} {status_text:<15}", Colors.GREEN if models_count > 0 else Colors.YELLOW)
        if ai2text_status['models']:
            print(f"   Available checkpoints: {len(ai2text_status['models'])}")
    else:
        print_colored(f"{'AI2Text':<20} {'✗ Offline':<15}", Colors.RED)
    
    print_colored("\n" + "="*60 + "\n", Colors.CYAN)
    
    # JSON output
    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'ai_llm_ss': ai_llm_ss_status,
            'ai2text': ai2text_status
        }
        print(json.dumps(output, indent=2))
    
    # Exit code
    all_online = ai_llm_ss_status['online'] and ai2text_status['online']
    sys.exit(0 if all_online else 1)

if __name__ == '__main__':
    main()

