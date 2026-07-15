#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true

DEVICE_INDEX=${PARAM_DEVICE_INDEX:-0}
FPS=${PARAM_FPS:-30}
WIDTH=${PARAM_WIDTH:-640}
HEIGHT=${PARAM_HEIGHT:-480}

# Replace my_camera_pkg and my_camera_node with your package and node
exec ros2 run my_camera_pkg my_camera_node \
  --ros-args \
  -p device_index:=${DEVICE_INDEX} \
  -p fps:=${FPS} \
  -p width:=${WIDTH} \
  -p height:=${HEIGHT} \
  $ROS2_REMAP_ARGS "$@"
