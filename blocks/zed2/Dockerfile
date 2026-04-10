FROM host-gateway:5000/ros2-block-base:humble

# Zed SDK requires specific installation procedure.
# See: https://www.stereolabs.com/docs/ros2/
# The Zed ROS2 wrapper is installed from apt after adding the StereoLabs PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget zstd \
    && rm -rf /var/lib/apt/lists/*

# Install ZED SDK (requires CUDA on GPU systems, CPU-only for dev)
# Pinned to 4.x for ROS2 Humble compatibility
RUN wget -q -O /tmp/ZED_SDK.run \
    "https://download.stereolabs.com/zedsdk/4.0/cu121/ubuntu22" \
    && chmod +x /tmp/ZED_SDK.run \
    && /tmp/ZED_SDK.run -- silent skip_tools \
    && rm /tmp/ZED_SDK.run

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-zed-ros2-wrapper \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
