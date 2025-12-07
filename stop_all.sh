#!/bin/bash

# Stop all ASR services

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}Stopping all services...${NC}\n"

# Function to stop service by PID file
stop_service() {
    local name=$1
    local pid_file="$SCRIPT_DIR/logs/$name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $name (PID: $pid)...${NC}"
            kill $pid 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid 2>/dev/null || true
            fi
            echo -e "${GREEN}✓ $name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $name process not found${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  PID file not found for $name${NC}"
    fi
}

# Stop services
stop_service "ai-llm-ss"
stop_service "ai-llm"
stop_service "ai2text"
stop_service "frontend"

# Also kill by port (fallback)
echo -e "\n${CYAN}Cleaning up any remaining processes on ports...${NC}"

for port in 8000 8001 8002 8080; do
    pid=$(lsof -ti :$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Killing process on port $port (PID: $pid)...${NC}"
        kill $pid 2>/dev/null || true
        sleep 1
        kill -9 $pid 2>/dev/null || true
    fi
done

echo -e "\n${GREEN}✓ All services stopped${NC}"

