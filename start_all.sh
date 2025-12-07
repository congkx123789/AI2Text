#!/bin/bash

# Multi-Model ASR Services Manager - Simple Bash Script
# Start tất cả services: ai-llm-ss, ai-llm, AI2Text, và frontend

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}🚀 Starting All ASR Services${NC}"
echo -e "${CYAN}========================================${NC}\n"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for service at $url...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Service is ready!${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    echo -e "${RED}✗ Service failed to start${NC}"
    return 1
}

# Create logs directory
mkdir -p logs

# Start ai-llm-ss (Port 8001)
echo -e "${CYAN}Starting ai-llm-ss (Port 8001)...${NC}"
if check_port 8001; then
    echo -e "${YELLOW}⚠️  Port 8001 is already in use${NC}"
else
    cd "$SCRIPT_DIR/ai-llm-ss"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    nohup python -m uvicorn src.asr.api:app --host 0.0.0.0 --port 8001 > "$SCRIPT_DIR/logs/ai-llm-ss.log" 2>&1 &
    AI_LLM_SS_PID=$!
    echo $AI_LLM_SS_PID > "$SCRIPT_DIR/logs/ai-llm-ss.pid"
    echo -e "${GREEN}✓ ai-llm-ss started (PID: $AI_LLM_SS_PID)${NC}"
    sleep 3
fi

# Start ai-llm (Port 8000)
echo -e "${CYAN}Starting ai-llm (Port 8000)...${NC}"
if check_port 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
else
    cd "$SCRIPT_DIR/ai-llm"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    nohup python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 > "$SCRIPT_DIR/logs/ai-llm.log" 2>&1 &
    AI_LLM_PID=$!
    echo $AI_LLM_PID > "$SCRIPT_DIR/logs/ai-llm.pid"
    echo -e "${GREEN}✓ ai-llm started (PID: $AI_LLM_PID)${NC}"
    sleep 3
fi

# Start AI2Text (Port 8002)
echo -e "${CYAN}Starting AI2Text (Port 8002)...${NC}"
if check_port 8002; then
    echo -e "${YELLOW}⚠️  Port 8002 is already in use${NC}"
else
    cd "$SCRIPT_DIR/AI2Text"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    nohup python -m uvicorn api.app_v2:app --host 0.0.0.0 --port 8002 > "$SCRIPT_DIR/logs/ai2text.log" 2>&1 &
    AI2TEXT_PID=$!
    echo $AI2TEXT_PID > "$SCRIPT_DIR/logs/ai2text.pid"
    echo -e "${GREEN}✓ AI2Text started (PID: $AI2TEXT_PID)${NC}"
    sleep 3
fi

# Start Frontend (Port 8080)
echo -e "${CYAN}Starting Frontend Server (Port 8080)...${NC}"
if check_port 8080; then
    echo -e "${YELLOW}⚠️  Port 8080 is already in use${NC}"
else
    cd "$SCRIPT_DIR"
    nohup python3 -m http.server 8080 > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$SCRIPT_DIR/logs/frontend.pid"
    echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"
    sleep 2
fi

# Wait for services
echo -e "\n${CYAN}Waiting for services to be ready...${NC}\n"
sleep 5

# Check services
echo -e "${CYAN}Checking service health...${NC}\n"

if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ai-llm-ss is healthy${NC}"
else
    echo -e "${RED}✗ ai-llm-ss health check failed${NC}"
fi

if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ai-llm is healthy${NC}"
else
    echo -e "${RED}✗ ai-llm health check failed${NC}"
fi

if curl -s -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ AI2Text is healthy${NC}"
else
    echo -e "${RED}✗ AI2Text health check failed${NC}"
fi

# Summary
echo -e "\n${CYAN}========================================${NC}"
echo -e "${GREEN}✓ All services started!${NC}"
echo -e "${CYAN}========================================${NC}\n"

echo -e "Services:"
echo -e "  ${CYAN}ai-llm-ss:${NC}    http://localhost:8001"
echo -e "  ${CYAN}ai-llm:${NC}       http://localhost:8000"
echo -e "  ${CYAN}AI2Text:${NC}      http://localhost:8002"
echo -e "  ${CYAN}Frontend:${NC}     http://localhost:8080/frontend.html"
echo -e "\n"

# Open browser
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:8080/frontend.html 2>/dev/null &
elif command -v open > /dev/null; then
    open http://localhost:8080/frontend.html 2>/dev/null &
fi

echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${YELLOW}Or run: ./stop_all.sh${NC}\n"

# Keep script running
trap 'echo -e "\n${YELLOW}Stopping all services...${NC}"; ./stop_all.sh; exit' INT TERM

# Wait
while true; do
    sleep 1
done

