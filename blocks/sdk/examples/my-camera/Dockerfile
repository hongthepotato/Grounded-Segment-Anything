FROM host-gateway:5000/ros2-block-base:humble

# Install your driver package here
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ros-humble-your-driver \
#     && rm -rf /var/lib/apt/lists/*

# Copy your ROS2 workspace source
# COPY ros2_ws/ /ros2_ws/
# RUN cd /ros2_ws && . /opt/ros/humble/setup.sh && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
