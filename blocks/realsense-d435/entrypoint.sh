#!/bin/bash
set -e
source /opt/ros/humble/setup.bash

SERIAL=${PARAM_SERIAL:-""}
RESOLUTION=${PARAM_RESOLUTION:-640x480}
FPS=${PARAM_FPS:-30}

WIDTH=$(echo $RESOLUTION | cut -dx -f1)
HEIGHT=$(echo $RESOLUTION | cut -dx -f2)

SERIAL_ARG=""
if [ -n "$SERIAL" ] && [ "$SERIAL" != "any" ]; then
  SERIAL_ARG="-p serial_no:=${SERIAL}"
fi

exec ros2 launch realsense2_camera rs_launch.py \
  color_width:=${WIDTH} \
  color_height:=${HEIGHT} \
  color_fps:=${FPS} \
  depth_width:=${WIDTH} \
  depth_height:=${HEIGHT} \
  depth_fps:=${FPS} \
  enable_pointcloud:=true \
  ${SERIAL_ARG} \
  $ROS2_REMAP_ARGS "$@"
