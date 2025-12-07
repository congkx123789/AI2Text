#!/bin/bash

# Quick status check for AI2Text and ai-llm-ss models

echo "🔍 Checking ASR Models Status..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check ai-llm-ss
echo "📊 ai-llm-ss (Port 8001):"
if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is online${NC}"
    echo "Health:"
    curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8001/health
    echo ""
    echo "Model Info:"
    curl -s http://localhost:8001/model/info | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8001/model/info
else
    echo -e "${RED}✗ Service is offline${NC}"
fi

echo ""
echo "---"
echo ""

# Check AI2Text
echo "📊 AI2Text (Port 8002):"
if curl -s -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is online${NC}"
    echo "Health:"
    curl -s http://localhost:8002/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8002/health
    echo ""
    echo "Available Models:"
    curl -s http://localhost:8002/models | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8002/models
else
    echo -e "${RED}✗ Service is offline${NC}"
fi

echo ""

