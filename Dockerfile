# Multi-stage Dockerfile for Training Job Manager
#
# Targets:
#   - api: FastAPI server
#   - worker: Training worker with GPU support
#
# Build:
#   docker build --target api -t training-api .
#   docker build --target worker -t training-worker .

# Base image with Python and system dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install base Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for job manager
RUN pip install --no-cache-dir \
    redis>=4.5.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    pydantic>=2.0.0 \
    websockets>=11.0

# Copy source code
COPY . .

# =====================================================================
# API Server target
# =====================================================================
FROM base as api

# Expose API port
EXPOSE 8080

# Default command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]

# =====================================================================
# Worker target (with GPU support)
# =====================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as worker-base

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (including PyTorch with CUDA)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Install job manager dependencies
RUN pip install --no-cache-dir \
    redis>=4.5.0 \
    fastapi>=0.100.0 \
    pydantic>=2.0.0

# Copy source code
COPY . .

FROM worker-base as worker

# Default command
CMD ["python", "-m", "ml_engine.jobs", "--gpu", "0"]
