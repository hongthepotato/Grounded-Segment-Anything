"""Live View ROS2 Node.

Subscribes to annotated_image and serves an MJPEG stream over HTTP.
Browse to http://<workstation>:<port>/stream to view the live feed.
"""

import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# cv_bridge is available from the base image
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False


class LiveViewNode(Node):
    def __init__(self):
        super().__init__('live_view')

        self.stream_port = int(os.environ.get('PARAM_STREAM_PORT', '8080'))
        self.quality = int(os.environ.get('PARAM_QUALITY', '85'))
        self.max_fps = int(os.environ.get('PARAM_MAX_FPS', '15'))
        self._min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0

        self._bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        self._frame_lock = threading.Lock()
        self._current_jpeg: bytes = b''
        self._last_frame_time = 0.0

        self.sub = self.create_subscription(
            Image,
            'image_in',
            self._on_image,
            10,
        )

        # Start MJPEG HTTP server in a background thread
        self._server = HTTPServer(('0.0.0.0', self.stream_port), self._make_handler())
        server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        server_thread.start()

        self.get_logger().info(
            f'Live View ready. Stream at http://localhost:{self.stream_port}/stream '
            f'quality={self.quality} max_fps={self.max_fps}'
        )

    def _on_image(self, msg: Image):
        now = time.monotonic()
        if self._min_interval > 0 and (now - self._last_frame_time) < self._min_interval:
            return
        self._last_frame_time = now

        try:
            if self._bridge:
                cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                # Fallback: treat as raw RGB8
                data = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = data.reshape((msg.height, msg.width, -1))

            _, jpeg = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            with self._frame_lock:
                self._current_jpeg = jpeg.tobytes()
        except Exception as e:
            self.get_logger().warn(f'Frame encode error: {e}')

    def _make_handler(self):
        node = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Suppress HTTP access log noise

            def do_GET(self):
                if self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while True:
                            with node._frame_lock:
                                frame = node._current_jpeg
                            if frame:
                                self.wfile.write(b'--frame\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(frame)
                                self.wfile.write(b'\r\n')
                            time.sleep(max(node._min_interval, 0.033))
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path == '/health':
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                else:
                    self.send_response(404)
                    self.end_headers()

        return MJPEGHandler

    def destroy_node(self):
        self._server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveViewNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
