FROM host-gateway:5000/ros2-block-base:humble

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
