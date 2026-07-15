# ros2-block-base:humble-cuda
# CUDA variant of the shared base — used ONLY by blocks that need GPU (yolo-inference).
# Built ONCE. ~750MB + CUDA runtime delta on top of base.
#
# Usage in block Dockerfiles:
#   FROM host-gateway:5000/ros2-block-base:humble-cuda
#
# Requires: NVIDIA Container Toolkit on the Docker host.
# Without NVIDIA GPU: build succeeds, but torch will fall back to CPU at runtime.

FROM host-gateway:5000/ros2-block-base:humble

# NVIDIA CUDA runtime — minimal (no cuDNN, no full toolkit)
# Pin to CUDA 12.1 for torch 2.x compatibility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime via pip wheel (lighter than apt CUDA toolkit)
# torch 2.x ships its own CUDA libraries — no system CUDA install needed
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

ENV CUDA_VISIBLE_DEVICES=0
