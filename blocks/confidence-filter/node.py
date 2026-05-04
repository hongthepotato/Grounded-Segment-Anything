"""Confidence Filter ROS2 Node.

Subscribes to raw detections and republishes only those whose best
hypothesis score meets the configured minimum confidence threshold.
Outputs filtered_detections (distinct alias from detections) to enforce
an intentional filter step in the graph.
"""

import os

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


class ConfidenceFilterNode(Node):
    def __init__(self):
        super().__init__("confidence_filter")

        self.min_confidence = float(os.environ.get("PARAM_MIN_CONFIDENCE", "0.7"))

        self.sub = self.create_subscription(
            Detection2DArray,
            "detections_in",
            self._on_detections,
            10,
        )
        self.pub = self.create_publisher(Detection2DArray, "filtered_out", 10)

        self.get_logger().info(f"Confidence Filter ready. min_confidence={self.min_confidence}")

    def _on_detections(self, msg: Detection2DArray):
        filtered = Detection2DArray()
        filtered.header = msg.header

        for det in msg.detections:
            # Use the highest score among result hypotheses
            max_score = max((h.score for h in det.results), default=0.0)
            if max_score >= self.min_confidence:
                filtered.detections.append(det)

        self.pub.publish(filtered)


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
