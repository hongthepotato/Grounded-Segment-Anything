#!/bin/bash
set -e
source /opt/ros/humble/setup.bash

DEVICE_INDEX=${PARAM_DEVICE_INDEX:-0}
RESOLUTION=${PARAM_RESOLUTION:-640x480}
FPS=${PARAM_FPS:-30}
PIXEL_FORMAT=${PARAM_PIXEL_FORMAT:-mjpeg}
CAMERA_INFO_URL=${PARAM_CAMERA_INFO_URL:-""}

WIDTH=$(echo $RESOLUTION | cut -dx -f1)
HEIGHT=$(echo $RESOLUTION | cut -dx -f2)

exec ros2 run usb_cam usb_cam_node_exe \
  --ros-args \
  -p video_device:=/dev/video${DEVICE_INDEX} \
  -p image_width:=${WIDTH} \
  -p image_height:=${HEIGHT} \
  -p framerate:=${FPS}.0 \
  -p pixel_format:=${PIXEL_FORMAT} \
  ${CAMERA_INFO_URL:+-p camera_info_url:=${CAMERA_INFO_URL}} \
  $ROS2_REMAP_ARGS "$@"
