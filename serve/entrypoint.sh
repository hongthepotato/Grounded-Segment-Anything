#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

exec ros2 run yolo_inference node --ros-args "$@"
