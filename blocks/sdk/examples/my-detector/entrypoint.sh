#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true

CONFIDENCE=${PARAM_CONFIDENCE:-0.5}
MODEL_PATH=${PARAM_MODEL_PATH:-/models/best.pt}

exec ros2 run my_detector_pkg my_detector_node \
  --ros-args \
  -p confidence_threshold:=${CONFIDENCE} \
  -p model_path:=${MODEL_PATH} \
  $ROS2_REMAP_ARGS "$@"
