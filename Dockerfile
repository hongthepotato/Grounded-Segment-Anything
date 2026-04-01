# ============================================================
# Stage 1: builder — compile CUDA extensions, install all deps
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9"

COPY pyproject.toml uv.lock ./
COPY deps/ deps/
COPY GroundingDINO/ GroundingDINO/

RUN uv sync --frozen --no-editable --no-install-project

COPY . .
RUN uv sync --frozen --no-editable

# ============================================================
# Stage 2: runtime — lean image with only what's needed
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    redis-tools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libxcb1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8080

CMD ["bash", "startup.bash"]
