#!/bin/bash

# Quick Start Script - One-click start all services
# Giống như Docker, chỉ cần chạy một lệnh để start tất cả

echo "🚀 Quick Start - Starting All Services..."
echo ""

# Check if Python script exists
if [ -f "start_all_services.py" ]; then
    echo "Using Python script (recommended)..."
    python3 start_all_services.py start
else
    echo "Using Bash script..."
    ./start_all.sh
fi

