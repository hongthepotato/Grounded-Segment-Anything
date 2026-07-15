FROM host-gateway:5000/ros2-block-base:humble

# Add your detection model dependencies
# RUN pip3 install --no-cache-dir your-detection-library

# COPY ros2_ws/ /ros2_ws/
# RUN cd /ros2_ws && . /opt/ros/humble/setup.sh && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
