FROM host-gateway:5000/ros2-block-base:humble-cuda

RUN pip3 install --no-cache-dir \
    ultralytics \
    supervision

# Copy the ROS2 workspace from the serve/ directory
COPY ros2_ws/ /ros2_ws/
RUN cd /ros2_ws && . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install \
    && rm -rf build log

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
