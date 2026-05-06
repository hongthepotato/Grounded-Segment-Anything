# ros2-block-base:humble
# Shared base for ALL platform blocks. Built ONCE, cached locally.
# Disk: ~750MB. Build time: ~8 minutes on first install.
#
# Usage in block Dockerfiles:
#   FROM host-gateway:5000/ros2-block-base:humble
#
# DO NOT add block-specific packages here. Keep this layer stable.

FROM ros:humble-ros-base

# Common ROS2 packages used by most blocks — avoid redundant installs per block
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rmw-fastrtps-cpp \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# FastDDS discovery server binary — needed for macOS/Windows sidecar
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# Python packages used across blocks
RUN pip3 install --no-cache-dir \
    rclpy \
    opencv-python-headless \
    numpy

# ROS2 workspace placeholder — blocks add their own packages on top
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Source ROS2 on every shell by default
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}
