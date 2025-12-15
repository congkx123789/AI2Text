#!/usr/bin/env python3
"""
Simple test to verify Gemini API key and find available models
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    sys.exit(1)

print(f"✓ Found API key: {api_key[:20]}...")
print("\nConfiguring Gemini API...")

try:
    genai.configure(api_key=api_key)
    print("✓ Configuration successful")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Try to list available models
print("\nTrying to list available models...")
try:
    models = genai.list_models()
    print("✓ Successfully listed models:")
    available_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            model_name = model.name.replace('models/', '')
            available_models.append(model_name)
            print(f"  - {model_name}")
    
    if available_models:
        # Try the first available model
        test_model = available_models[0]
        print(f"\nTesting with model: {test_model}")
        model = genai.GenerativeModel(test_model)
        response = model.generate_content("Say hello in one word")
        print(f"✓ Test successful! Response: {response.text}")
        print(f"\n✅ Recommended GEMINI_MODEL: {test_model}")
    else:
        print("⚠️  No models with generateContent support found")
        
except Exception as e:
    print(f"⚠️  Could not list models: {e}")
    print("\nTrying common model names directly...")
    
    # Try common model names
    common_models = [
        "gemini-pro",
        "models/gemini-pro", 
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash"
    ]
    
    for model_name in common_models:
        try:
            print(f"  Trying {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hi")
            print(f"  ✓ {model_name} works! Response: {response.text[:50]}")
            print(f"\n✅ Recommended GEMINI_MODEL: {model_name}")
            break
        except Exception as e:
            print(f"  ✗ {model_name} failed: {str(e)[:100]}")
    else:
        print("\n❌ None of the common model names worked.")
        print("   Please check:")
        print("   1. API key is valid and has proper permissions")
        print("   2. You have access to Gemini API")
        print("   3. Check https://ai.google.dev/gemini-api/docs for latest model names")
        sys.exit(1)












