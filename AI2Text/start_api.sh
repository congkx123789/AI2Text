#!/bin/bash
# Start ASR API server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set default port
PORT=${1:-8000}

echo "Starting ASR API server on port $PORT..."
echo "API will be available at http://localhost:$PORT"
echo ""
echo "API Documentation: http://localhost:$PORT/docs"
echo ""

# Run the API
python -m api.app_v2

# Or using uvicorn directly:
# uvicorn api.app_v2:app --host 0.0.0.0 --port $PORT --reload

