#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
exec python3 /app/node.py $ROS2_REMAP_ARGS "$@"
