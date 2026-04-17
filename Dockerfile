# ============================================================
# Stage 1: builder — compile CUDA extensions, install all deps
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

WORKDIR /app

# Switch to Aliyun mirror — archive.ubuntu.com and security.ubuntu.com are
# unreliable on restricted networks (e.g., mainland China).
RUN sed -i \
    's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; \
     s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    /etc/apt/sources.list

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
ENV TORCH_CUDA_ARCH_LIST="7.5"

COPY pyproject.toml uv.lock ./
COPY deps/ deps/
COPY GroundingDINO/ GroundingDINO/

RUN uv sync --frozen --no-editable --no-install-project

COPY . .
RUN uv sync --frozen --no-editable && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# Stage 2: runtime — lean image with only what's needed
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN sed -i \
    's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; \
     s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    /etc/apt/sources.list

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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder /opt/venv /opt/venv

# Copy only the runtime source — skip deps/ and GroundingDINO/ source trees
# (both are installed --no-editable into the venv and don't need to ship as source)
COPY --from=builder /app/api /app/api
COPY --from=builder /app/core /app/core
COPY --from=builder /app/ml_engine /app/ml_engine
COPY --from=builder /app/augmentation /app/augmentation
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/startup.bash /app/startup.bash

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8080

CMD ["bash", "startup.bash"]
