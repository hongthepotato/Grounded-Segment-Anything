#!/bin/bash
# Block SDK entrypoint template.
# Copy this file to your block directory and customize the `ros2 run` command.
#
# The launch engine ALWAYS sets:
#   ROS2_REMAP_ARGS   - e.g. "--ros-args --remap image_in:=/pipeline/abc123/cam/out"
#   ROS_DOMAIN_ID     - Pipeline isolation domain (0-230)
#   PARAM_*           - Block parameters from block.json (e.g. PARAM_CONFIDENCE, PARAM_FPS)
#
# IMPORTANT: pass $ROS2_REMAP_ARGS UNQUOTED so multiple args expand correctly.

set -e

source /opt/ros/humble/setup.bash

# Build your ROS2 workspace if not pre-built in Dockerfile
if [ -d /ros2_ws/src ] && [ "$(ls -A /ros2_ws/src 2>/dev/null)" ]; then
    if [ ! -f /ros2_ws/install/setup.bash ]; then
        cd /ros2_ws
        colcon build --symlink-install 2>&1
    fi
    source /ros2_ws/install/setup.bash
fi

# --- CUSTOMIZE BELOW THIS LINE ---
# Replace my_package and my_node with your block's ROS2 package and node names.
# $ROS2_REMAP_ARGS is injected by the launch engine — do not quote it.
exec ros2 run my_package my_node $ROS2_REMAP_ARGS "$@"
