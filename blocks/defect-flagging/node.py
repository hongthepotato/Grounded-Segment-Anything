"""Defect Flagging ROS2 Node.

Subscribes to detections and publishes a Bool flag when a qualifying
detection is found. "Qualifying" means: class label matches (if configured)
AND bounding box area exceeds minimum threshold.
"""

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray


class DefectFlaggingNode(Node):
    def __init__(self):
        super().__init__("defect_flagging")

        self.flag_label = os.environ.get("PARAM_FLAG_LABEL", "").strip()
        self.min_area = float(os.environ.get("PARAM_MIN_AREA", "0.0"))

        self.sub = self.create_subscription(
            Detection2DArray,
            "detections_in",
            self._on_detections,
            10,
        )
        self.pub = self.create_publisher(Bool, "flag_out", 10)

        self.get_logger().info(
            f'Defect Flagging ready. label_filter="{self.flag_label}" min_area={self.min_area}'
        )

    def _on_detections(self, msg: Detection2DArray):
        flagged = False
        for det in msg.detections:
            # Label check — match against the highest-score result hypothesis
            if self.flag_label:
                labels = [h.class_id for h in det.results]
                if self.flag_label not in labels:
                    continue

            # Area check
            bbox = det.bbox
            area = bbox.size_x * bbox.size_y
            if area < self.min_area:
                continue

            flagged = True
            break

        out = Bool()
        out.data = flagged
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DefectFlaggingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
