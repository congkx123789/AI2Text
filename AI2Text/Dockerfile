# Dockerfile for ASR system

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for API and embeddings
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pyctcdecode \
    gensim \
    faiss-cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed checkpoints logs results models/embeddings

# Expose API port
EXPOSE 8000

# Default command (API server)
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

