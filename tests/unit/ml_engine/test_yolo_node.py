"""Tests for the ROS2 YoloInferenceNode (serve/ros2_ws/src/yolo_inference/node.py).

NOTE: These tests mock rclpy entirely so they run outside a ROS2 environment.
"""

import json
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out ROS2 deps so the module loads without a live ROS2 install
# ---------------------------------------------------------------------------


def _make_ros2_stubs():
    """Build minimal stubs for rclpy and sensor_msgs/vision_msgs."""
    mods = {}

    rclpy = types.ModuleType("rclpy")
    rclpy.init = MagicMock()
    rclpy.spin = MagicMock()
    rclpy.shutdown = MagicMock()

    node_mod = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, name):
            self._params = {}

        def declare_parameter(self, name, default):
            self._params[name] = default

        def get_parameter(self, name):
            m = MagicMock()
            m.value = self._params.get(name)
            return m

        def create_subscription(self, *a, **kw):
            return MagicMock()

        def create_publisher(self, *a, **kw):
            return MagicMock()

        def destroy_node(self):
            pass

        def get_logger(self):
            log = MagicMock()
            log.info = MagicMock()
            log.warn = MagicMock()
            log.warning = MagicMock()
            log.error = MagicMock()
            log.debug = MagicMock()
            return log

    node_mod.Node = FakeNode
    rclpy.node = node_mod
    mods["rclpy"] = rclpy
    mods["rclpy.node"] = node_mod

    for pkg in (
        "sensor_msgs",
        "sensor_msgs.msg",
        "std_msgs",
        "std_msgs.msg",
        "vision_msgs",
        "vision_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
    ):
        mods[pkg] = types.ModuleType(pkg)

    # Minimal message stubs
    class _Msg:
        def __init__(self, **kw):
            [setattr(self, k, v) for k, v in kw.items()]

    mods["sensor_msgs.msg"].Image = lambda: _Msg(data=b"", height=0, width=0, encoding="bgr8")
    mods["std_msgs.msg"].String = lambda data="": _Msg(data=data)
    mods["vision_msgs.msg"].Detection2DArray = lambda: _Msg(header=None, detections=[])
    mods["vision_msgs.msg"].Detection2D = lambda: _Msg(header=None, bbox=None, results=[])
    mods["vision_msgs.msg"].BoundingBox2D = lambda: _Msg(center=None, size_x=0.0, size_y=0.0)
    mods["vision_msgs.msg"].ObjectHypothesisWithPose = lambda: _Msg(hypothesis=_Msg(class_id="", score=0.0))
    mods["geometry_msgs.msg"].Pose2D = lambda: _Msg(x=0.0, y=0.0, theta=0.0)

    return mods


@pytest.fixture(autouse=True)
def ros2_stubs():
    stubs = _make_ros2_stubs()
    for name, mod in stubs.items():
        sys.modules[name] = mod
    yield
    for name in stubs:
        sys.modules.pop(name, None)
    # Also remove the node module so it gets re-imported cleanly
    sys.modules.pop("serve.ros2_ws.src.yolo_inference.yolo_inference.node", None)


def _load_node():
    """Import node.py fresh (avoids caching issues with stub injection)."""
    import importlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "yolo_node",
        Path(__file__).parent.parent.parent.parent
        / "serve/ros2_ws/src/yolo_inference/yolo_inference/node.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestToDetection2dArray:
    def test_empty_results_returns_empty_array(self):
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)
        node._params = {
            "model_path": "/model/best.pt",
            "confidence": 0.5,
            "device": "cuda",
            "enable_tensorrt": False,
            "engine_cache_dir": "/model/cache",
            "input_topic": "/camera/image_raw",
            "output_topic": "/detections",
        }
        node.conf = 0.5
        node.model = None
        node._engine_loaded = False
        node._frame_count = 0
        node._total_inference_ms = 0.0
        node.det_pub = MagicMock()
        node.diag_pub = MagicMock()
        node.sub = MagicMock()

        result = MagicMock()
        result.boxes = None
        header = MagicMock()

        det_array = node._to_detection2d_array(result, header)
        assert det_array.detections == []

    def test_single_box_mapped_correctly(self):
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)

        import torch

        result = MagicMock()
        boxes = MagicMock()
        boxes.xyxy = [torch.tensor([10.0, 20.0, 50.0, 80.0])]
        boxes.cls = [torch.tensor(2.0)]
        boxes.conf = [torch.tensor(0.87)]
        result.boxes = boxes

        header = MagicMock()
        det_array = node._to_detection2d_array(result, header)

        assert len(det_array.detections) == 1
        det = det_array.detections[0]
        assert det.bbox.center.x == pytest.approx(30.0)
        assert det.bbox.center.y == pytest.approx(50.0)
        assert det.bbox.size_x == pytest.approx(40.0)
        assert det.bbox.size_y == pytest.approx(60.0)
        assert det.results[0].hypothesis.class_id == "2"
        assert det.results[0].hypothesis.score == pytest.approx(0.87, abs=0.01)

    def test_header_passed_through(self):
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)

        result = MagicMock()
        result.boxes = None
        header = MagicMock()
        header.stamp = "now"

        det_array = node._to_detection2d_array(result, header)
        assert det_array.header is header


class TestEncodingHandling:
    def test_rgb8_converted_to_bgr(self):
        """RGB→BGR swap must happen for rgb8 encoding."""
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)
        node.model = MagicMock()
        node.model.return_value = [MagicMock(boxes=None)]
        node.conf = 0.5
        node._engine_loaded = True
        node._frame_count = 0
        node._total_inference_ms = 0.0
        node.det_pub = MagicMock()
        node.diag_pub = MagicMock()

        # 2x2 RGB image
        img_data = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 0]]], dtype=np.uint8)

        msg = MagicMock()
        msg.encoding = "rgb8"
        msg.height = 2
        msg.width = 2
        msg.data = img_data.tobytes()

        calls = []

        def capture_call(img, conf, verbose):
            calls.append(img.copy())
            return [MagicMock(boxes=None)]

        node.model.side_effect = capture_call
        node.on_image(msg)

        assert len(calls) == 1
        # First pixel should be [0, 0, 255] (BGR from original RGB [255, 0, 0])
        np.testing.assert_array_equal(calls[0][0, 0], [0, 0, 255])


class TestDiagnosticsPublished:
    def test_diagnostics_json_published_on_each_frame(self):
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)
        node.model = MagicMock()
        node.model.return_value = [MagicMock(boxes=None)]
        node.conf = 0.5
        node._engine_loaded = True
        node._frame_count = 0
        node._total_inference_ms = 0.0
        node.det_pub = MagicMock()
        node.diag_pub = MagicMock()

        msg = MagicMock()
        msg.encoding = "bgr8"
        msg.height = 2
        msg.width = 2
        msg.data = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()

        node.on_image(msg)

        node.diag_pub.publish.assert_called_once()
        published = node.diag_pub.publish.call_args[0][0]
        diag = json.loads(published.data)
        assert "fps" in diag
        assert "inference_time_ms" in diag
        assert "engine_loaded" in diag
        assert diag["engine_loaded"] is True

    def test_model_none_skips_inference(self):
        """If model failed to load, on_image must return immediately without crashing."""
        mod = _load_node()
        node = mod.YoloInferenceNode.__new__(mod.YoloInferenceNode)
        node.model = None
        node.det_pub = MagicMock()
        node.diag_pub = MagicMock()

        msg = MagicMock()
        node.on_image(msg)

        node.det_pub.publish.assert_not_called()
        node.diag_pub.publish.assert_not_called()
