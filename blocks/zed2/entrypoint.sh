#!/bin/bash
set -e
source /opt/ros/humble/setup.bash

CAMERA_INDEX=${PARAM_CAMERA_INDEX:-0}
RESOLUTION=${PARAM_RESOLUTION:-HD720}
FPS=${PARAM_FPS:-30}
DEPTH_MODE=${PARAM_DEPTH_MODE:-PERFORMANCE}

exec ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed2 \
  camera_index:=${CAMERA_INDEX} \
  pub_resolution:=${RESOLUTION} \
  general.grab_frame_rate:=${FPS} \
  depth.depth_mode:=${DEPTH_MODE} \
  $ROS2_REMAP_ARGS "$@"
