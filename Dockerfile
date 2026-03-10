FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies and Python 3.10 for uv project sync.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    redis-tools \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libxcb1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"

# Layer 1: dependency files only (cache-friendly).
COPY pyproject.toml uv.lock ./
COPY deps/ deps/
COPY GroundingDINO/ GroundingDINO/

# Install all locked dependencies and compile extension packages.
RUN uv sync --frozen --no-editable --no-install-project

# Layer 2: source code (changes frequently).
COPY . .
RUN uv sync --frozen --no-editable

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8080

CMD ["bash", "startup.bash"]
