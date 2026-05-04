#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

CONFIDENCE=${PARAM_CONFIDENCE:-0.5}
MODEL_VERSION=${PARAM_MODEL_VERSION:-latest}
MODEL_PATH=${PARAM_MODEL_PATH:-/models/best.pt}
DEVICE=${PARAM_DEVICE:-cuda:0}
IOU_THRESHOLD=${PARAM_IOU_THRESHOLD:-0.45}

# Resolve model path based on version selection
if [ "$MODEL_VERSION" = "custom" ]; then
    RESOLVED_MODEL=$MODEL_PATH
elif [ "$MODEL_VERSION" = "latest" ]; then
    # Prefer a locally trained model if present
    RESOLVED_MODEL="${MODEL_PATH:-yolov8n.pt}"
else
    RESOLVED_MODEL="yolov8${MODEL_VERSION#v8}.pt"
fi

exec ros2 run yolo_inference yolo_inference_node \
  --ros-args \
  -p confidence_threshold:=${CONFIDENCE} \
  -p model_path:=${RESOLVED_MODEL} \
  -p device:=${DEVICE} \
  -p iou_threshold:=${IOU_THRESHOLD} \
  $ROS2_REMAP_ARGS "$@"
