"""
ROS2 YOLO Inference Node.

Subscribes to sensor_msgs/Image, runs YOLO inference (optionally with TensorRT),
and publishes vision_msgs/Detection2DArray. Standalone — no ml_engine dependency.

Usage (inside container):
    ros2 run yolo_inference node --ros-args -p confidence:=0.5

Topics:
    Subscribe: /camera/image_raw (sensor_msgs/msg/Image)
    Publish:   /detections (vision_msgs/msg/Detection2DArray)
               /yolo_inference/diagnostics (std_msgs/msg/String, JSON)
"""

import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D


class YoloInferenceNode(Node):
    """ROS2 node that runs YOLO inference on incoming camera images."""

    def __init__(self):
        super().__init__('yolo_inference')

        # Declare parameters
        self.declare_parameter('model_path', '/model/best.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('enable_tensorrt', True)
        self.declare_parameter('engine_cache_dir', '/model/cache')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/detections')

        # Read parameters
        self.conf = self.get_parameter('confidence').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # Load model
        self.model = self._load_model()
        self._engine_loaded = self.model is not None

        # Publishers / subscribers
        self.sub = self.create_subscription(
            Image, input_topic, self.on_image, 10
        )
        self.det_pub = self.create_publisher(Detection2DArray, output_topic, 10)
        self.diag_pub = self.create_publisher(String, '/yolo_inference/diagnostics', 10)

        # Diagnostics state
        self._frame_count = 0
        self._total_inference_ms = 0.0

        self.get_logger().info(
            'YoloInferenceNode ready. input=%s output=%s conf=%.2f',
            input_topic, output_topic, self.conf
        )

    def _load_model(self):
        """Load YOLO model, optionally converting to TensorRT on first run."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error('ultralytics not installed — cannot load model')
            return None

        model_path = self.get_parameter('model_path').value
        cache_dir = self.get_parameter('engine_cache_dir').value
        enable_trt = self.get_parameter('enable_tensorrt').value
        engine_path = Path(cache_dir) / 'best.engine'

        # Use cached TensorRT engine if available
        if enable_trt and engine_path.exists():
            self.get_logger().info('Loading cached TensorRT engine: %s', engine_path)
            return YOLO(str(engine_path))

        if not Path(model_path).exists():
            self.get_logger().error('Model file not found: %s', model_path)
            return None

        model = YOLO(model_path)

        if enable_trt:
            try:
                self.get_logger().info(
                    'First boot: converting to TensorRT (~2 min)...'
                )
                model.export(format='engine', device=0)
                exported = Path(model_path).with_suffix('.engine')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                exported.rename(engine_path)
                model = YOLO(str(engine_path))
                self.get_logger().info('TensorRT engine cached at %s', engine_path)
            except Exception as e:
                self.get_logger().warn(
                    'TensorRT export failed, falling back to PyTorch: %s', str(e)
                )

        return model

    def on_image(self, msg: Image):
        """Process incoming image message."""
        if self.model is None:
            return

        t0 = time.monotonic()

        # Convert ROS Image to numpy — handle common encodings
        dtype = np.uint8
        if msg.encoding in ('rgb8', 'bgr8'):
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                img = img[:, :, ::-1]  # RGB→BGR for YOLO
        elif msg.encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        else:
            # Attempt generic reshape; may produce garbage for exotic encodings
            channels = len(msg.data) // (msg.height * msg.width)
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

        results = self.model(img, conf=self.conf, verbose=False)

        det_array = self._to_detection2d_array(results[0], msg.header)
        self.det_pub.publish(det_array)

        # Diagnostics
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._frame_count += 1
        self._total_inference_ms += elapsed_ms
        avg_fps = self._frame_count / (self._total_inference_ms / 1000.0) if self._total_inference_ms > 0 else 0.0

        diag = {
            'fps': round(avg_fps, 1),
            'inference_time_ms': round(elapsed_ms, 1),
            'engine_loaded': self._engine_loaded,
            'detections': len(det_array.detections),
        }
        self.diag_pub.publish(String(data=json.dumps(diag)))

    def _to_detection2d_array(self, result, header) -> Detection2DArray:
        """Convert ultralytics Results to vision_msgs/Detection2DArray."""
        array = Detection2DArray()
        array.header = header

        if result.boxes is None:
            return array

        boxes = result.boxes
        for i in range(len(boxes)):
            det = Detection2D()
            det.header = header

            # Bounding box (xyxy → center + size)
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            w = float(x2 - x1)
            h = float(y2 - y1)

            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.x = cx
            bbox.center.y = cy
            bbox.size_x = w
            bbox.size_y = h
            det.bbox = bbox

            # Class + confidence
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(boxes.cls[i].item()))
            hyp.hypothesis.score = float(boxes.conf[i].item())
            det.results.append(hyp)

            array.detections.append(det)

        return array


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
