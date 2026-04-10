"""Result Logger ROS2 Node.

Subscribes to detections and writes them to a file in JSONL or CSV format.
Each message produces one line per detection. File is flushed after each batch.
"""

import csv
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


class ResultLoggerNode(Node):
    def __init__(self):
        super().__init__('result_logger')

        self.output_path = Path(os.environ.get('PARAM_OUTPUT_PATH', '/data/detections.jsonl'))
        self.fmt = os.environ.get('PARAM_FORMAT', 'jsonl').lower()
        self.max_size_bytes = int(os.environ.get('PARAM_MAX_FILE_SIZE_MB', '500')) * 1024 * 1024
        rotate_on_start = os.environ.get('PARAM_ROTATE_ON_START', 'false').lower() == 'true'

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if rotate_on_start and self.output_path.exists():
            rotated = self.output_path.with_suffix(f'.{int(time.time())}{self.output_path.suffix}')
            self.output_path.rename(rotated)
            self.get_logger().info(f'Rotated existing log to {rotated}')

        self._file = None
        self._csv_writer = None
        self._open_file()

        self.sub = self.create_subscription(
            Detection2DArray,
            'detections_in',
            self._on_detections,
            10,
        )

        self.get_logger().info(
            f'Result Logger ready. output={self.output_path} format={self.fmt}'
        )

    def _open_file(self):
        mode = 'a' if self.output_path.exists() else 'w'
        self._file = open(self.output_path, mode, newline='' if self.fmt == 'csv' else None)
        if self.fmt == 'csv':
            self._csv_writer = csv.writer(self._file)
            if mode == 'w':
                self._csv_writer.writerow(['timestamp', 'detection_id', 'class_id', 'score', 'x', 'y', 'w', 'h'])

    def _check_rotation(self):
        if self.max_size_bytes > 0 and self.output_path.stat().st_size >= self.max_size_bytes:
            self._file.close()
            rotated = self.output_path.with_suffix(f'.{int(time.time())}{self.output_path.suffix}')
            self.output_path.rename(rotated)
            self._open_file()
            self.get_logger().info(f'Rotated log to {rotated}')

    def _on_detections(self, msg: Detection2DArray):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for i, det in enumerate(msg.detections):
            class_id = det.results[0].class_id if det.results else ''
            score = det.results[0].score if det.results else 0.0
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            if self.fmt == 'csv':
                self._csv_writer.writerow([ts, i, class_id, score, x, y, w, h])
            else:
                record = {
                    'timestamp': ts,
                    'detection_index': i,
                    'class_id': class_id,
                    'score': score,
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                }
                self._file.write(json.dumps(record) + '\n')

        self._file.flush()
        if self.output_path.exists():
            self._check_rotation()

    def destroy_node(self):
        if self._file:
            self._file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ResultLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
