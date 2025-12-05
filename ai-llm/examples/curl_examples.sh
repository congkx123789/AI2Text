#!/bin/bash
# Example: How to use AI-LLM API from command line with curl

BASE_URL="http://localhost:8000"

echo "=== Health Check ==="
curl -X GET "${BASE_URL}/health" | jq '.'

echo -e "\n=== Transcribe Audio (from server path) ==="
curl -X POST "${BASE_URL}/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "data/raw/audio/why-hello-there-103596.wav"}' | jq '.'

echo -e "\n=== Transcribe Audio (upload file) ==="
curl -X POST "${BASE_URL}/transcribe/upload" \
  -F "file=@data/raw/audio/i-dont-like-you-87027.wav" | jq '.'

echo -e "\n=== Ask Question ==="
curl -X POST "${BASE_URL}/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic discussed?", "top_k": 5}' | jq '.'

