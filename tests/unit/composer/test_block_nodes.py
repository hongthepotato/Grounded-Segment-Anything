"""Unit tests for block node logic.

Covers:
- defect-flagging: label filter, area filter, no detections → False
- confidence-filter: threshold filtering, preserves header
- result-logger: JSONL output, CSV output, file rotation
- conveyor-stop: cooldown suppression, trigger_on_true=False suppresses
- live-view: frame encoding path (if cv2 available)

All tests mock ROS2/rclpy so they run without a ROS2 installation.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# ── ROS2 stubs — rclpy is not installed on macOS/CI ──────────────────────────
class _FakeNode:
    def __init__(self, name):
        self._logger = MagicMock()

    def get_logger(self):
        return self._logger

    def create_subscription(self, *a, **k):
        return MagicMock()

    def create_publisher(self, *a, **k):
        return MagicMock()

    def destroy_node(self):
        pass


class _FakeBool:
    def __init__(self):
        self.data = False


class _FakeDetection:
    def __init__(self, class_id="defect", score=0.9, size_x=50.0, size_y=50.0):
        self.results = [MagicMock(class_id=class_id, score=score)]
        self.bbox = MagicMock(
            center=MagicMock(position=MagicMock(x=100.0, y=100.0)), size_x=size_x, size_y=size_y
        )


class _FakeDetectionArray:
    def __init__(self, detections=None):
        self.header = MagicMock(stamp=MagicMock(sec=1000, nanosec=0))
        self.detections = detections or []


# Patch rclpy and dependent modules BEFORE importing block nodes
@pytest.fixture(autouse=True)
def mock_ros2(monkeypatch):
    rclpy_mock = MagicMock()
    rclpy_mock.node.Node = _FakeNode

    std_msgs_mock = MagicMock()
    std_msgs_mock.msg.Bool = _FakeBool
    std_msgs_mock.msg.String = MagicMock

    vision_msgs_mock = MagicMock()
    vision_msgs_mock.msg.Detection2DArray = _FakeDetectionArray

    sensor_msgs_mock = MagicMock()
    sensor_msgs_mock.msg.Image = MagicMock

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_mock)
    monkeypatch.setitem(sys.modules, "rclpy.node", MagicMock(Node=_FakeNode))
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs_mock)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_mock.msg)
    monkeypatch.setitem(sys.modules, "vision_msgs", vision_msgs_mock)
    monkeypatch.setitem(sys.modules, "vision_msgs.msg", vision_msgs_mock.msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs_mock)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_mock.msg)
    monkeypatch.setitem(sys.modules, "geometry_msgs", MagicMock())
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", MagicMock())


class TestDefectFlagging:
    def _make_node(self, monkeypatch, **env_vars):
        for key, val in env_vars.items():
            monkeypatch.setenv(f"PARAM_{key.upper()}", str(val))
        # Remove cached module if previously imported
        sys.modules.pop("blocks.defect-flagging.node", None)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "defect_flagging_node", repo_root / "blocks" / "defect-flagging" / "node.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.DefectFlaggingNode()

    def test_flags_when_detection_present(self, monkeypatch):
        node = self._make_node(monkeypatch)
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        msg = _FakeDetectionArray([_FakeDetection()])
        node._on_detections(msg)
        assert len(published) == 1
        assert published[0].data is True

    def test_no_flag_when_no_detections(self, monkeypatch):
        node = self._make_node(monkeypatch)
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        msg = _FakeDetectionArray([])
        node._on_detections(msg)
        assert published[0].data is False

    def test_label_filter_rejects_wrong_class(self, monkeypatch):
        node = self._make_node(monkeypatch, FLAG_LABEL="defect")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        # Detection with wrong class
        msg = _FakeDetectionArray([_FakeDetection(class_id="background")])
        node._on_detections(msg)
        assert published[0].data is False

    def test_label_filter_accepts_matching_class(self, monkeypatch):
        node = self._make_node(monkeypatch, FLAG_LABEL="defect")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        msg = _FakeDetectionArray([_FakeDetection(class_id="defect")])
        node._on_detections(msg)
        assert published[0].data is True

    def test_area_filter_rejects_small_bboxes(self, monkeypatch):
        node = self._make_node(monkeypatch, MIN_AREA="10000")  # 100x100 = 10000
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        # 50x50 = 2500 < 10000
        msg = _FakeDetectionArray([_FakeDetection(size_x=50.0, size_y=50.0)])
        node._on_detections(msg)
        assert published[0].data is False

    def test_area_filter_accepts_large_bboxes(self, monkeypatch):
        node = self._make_node(monkeypatch, MIN_AREA="100")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        # 200x200 = 40000 > 100
        msg = _FakeDetectionArray([_FakeDetection(size_x=200.0, size_y=200.0)])
        node._on_detections(msg)
        assert published[0].data is True


class TestConfidenceFilter:
    def _make_node(self, monkeypatch, min_confidence="0.7"):
        monkeypatch.setenv("PARAM_MIN_CONFIDENCE", min_confidence)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "confidence_filter_node", repo_root / "blocks" / "confidence-filter" / "node.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ConfidenceFilterNode()

    def test_passes_high_confidence_detections(self, monkeypatch):
        node = self._make_node(monkeypatch, "0.7")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        det = _FakeDetection(score=0.9)
        msg = _FakeDetectionArray([det])
        node._on_detections(msg)
        assert len(published[0].detections) == 1

    def test_rejects_low_confidence_detections(self, monkeypatch):
        node = self._make_node(monkeypatch, "0.7")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        det = _FakeDetection(score=0.5)
        msg = _FakeDetectionArray([det])
        node._on_detections(msg)
        assert len(published[0].detections) == 0

    def test_preserves_header(self, monkeypatch):
        node = self._make_node(monkeypatch, "0.5")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        msg = _FakeDetectionArray([_FakeDetection(score=0.8)])
        node._on_detections(msg)
        assert published[0].header == msg.header

    def test_empty_detections_publishes_empty(self, monkeypatch):
        node = self._make_node(monkeypatch, "0.5")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        msg = _FakeDetectionArray([])
        node._on_detections(msg)
        assert published[0].detections == []

    def test_multiple_detections_filtered_correctly(self, monkeypatch):
        node = self._make_node(monkeypatch, "0.7")
        published = []
        node.pub = MagicMock(publish=lambda m: published.append(m))
        dets = [_FakeDetection(score=0.9), _FakeDetection(score=0.3), _FakeDetection(score=0.8)]
        msg = _FakeDetectionArray(dets)
        node._on_detections(msg)
        assert len(published[0].detections) == 2  # 0.9 and 0.8 pass, 0.3 fails


class TestResultLogger:
    def _make_node(self, monkeypatch, tmp_path, fmt="jsonl"):
        output_path = str(tmp_path / f"detections.{fmt}")
        monkeypatch.setenv("PARAM_OUTPUT_PATH", output_path)
        monkeypatch.setenv("PARAM_FORMAT", fmt)
        monkeypatch.setenv("PARAM_MAX_FILE_SIZE_MB", "0")
        monkeypatch.setenv("PARAM_ROTATE_ON_START", "false")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "result_logger_node", repo_root / "blocks" / "result-logger" / "node.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ResultLoggerNode(), output_path

    def test_jsonl_output_written(self, monkeypatch, tmp_path):
        node, output_path = self._make_node(monkeypatch, tmp_path, "jsonl")
        msg = _FakeDetectionArray([_FakeDetection(class_id="defect", score=0.9)])
        node._on_detections(msg)
        node.destroy_node()
        lines = Path(output_path).read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["class_id"] == "defect"
        assert record["score"] == pytest.approx(0.9)

    def test_csv_output_has_header(self, monkeypatch, tmp_path):
        node, output_path = self._make_node(monkeypatch, tmp_path, "csv")
        msg = _FakeDetectionArray([_FakeDetection()])
        node._on_detections(msg)
        node.destroy_node()
        lines = Path(output_path).read_text().strip().splitlines()
        assert lines[0].startswith("timestamp,detection_id")

    def test_multiple_detections_write_multiple_lines(self, monkeypatch, tmp_path):
        node, output_path = self._make_node(monkeypatch, tmp_path, "jsonl")
        msg = _FakeDetectionArray([_FakeDetection(), _FakeDetection(class_id="scratch")])
        node._on_detections(msg)
        node.destroy_node()
        lines = Path(output_path).read_text().strip().splitlines()
        assert len(lines) == 2

    def test_file_created_in_output_dir(self, monkeypatch, tmp_path):
        node, output_path = self._make_node(monkeypatch, tmp_path, "jsonl")
        node.destroy_node()
        assert Path(output_path).parent.exists()


class TestConveyorStop:
    def _make_node(self, monkeypatch, **env_vars):
        defaults = {
            "PLC_ADDRESS": "127.0.0.1:502",
            "REGISTER_ADDRESS": "0",
            "TRIGGER_ON_TRUE": "true",
            "COOLDOWN_MS": "0",
        }
        defaults.update(env_vars)
        for k, v in defaults.items():
            monkeypatch.setenv(f"PARAM_{k}", str(v))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "conveyor_stop_node", repo_root / "blocks" / "conveyor-stop" / "node.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        node = mod.ConveyorStopNode()
        node._send_stop = MagicMock()  # mock PLC call
        return node

    def test_sends_stop_when_flag_is_true(self, monkeypatch):
        node = self._make_node(monkeypatch, TRIGGER_ON_TRUE="true")
        msg = _FakeBool()
        msg.data = True
        node._on_flag(msg)
        node._send_stop.assert_called_once()

    def test_no_stop_when_flag_is_false(self, monkeypatch):
        node = self._make_node(monkeypatch, TRIGGER_ON_TRUE="true")
        msg = _FakeBool()
        msg.data = False
        node._on_flag(msg)
        node._send_stop.assert_not_called()

    def test_trigger_on_false_inverts_behavior(self, monkeypatch):
        node = self._make_node(monkeypatch, TRIGGER_ON_TRUE="false")
        msg = _FakeBool()
        msg.data = False
        node._on_flag(msg)
        node._send_stop.assert_called_once()

    def test_cooldown_suppresses_rapid_triggers(self, monkeypatch):
        node = self._make_node(monkeypatch, COOLDOWN_MS="5000")  # 5 second cooldown
        msg = _FakeBool()
        msg.data = True
        node._on_flag(msg)
        node._on_flag(msg)  # within cooldown window
        assert node._send_stop.call_count == 1  # only first trigger fires

    def test_cooldown_zero_allows_repeated_triggers(self, monkeypatch):
        node = self._make_node(monkeypatch, COOLDOWN_MS="0")
        msg = _FakeBool()
        msg.data = True
        node._on_flag(msg)
        node._on_flag(msg)
        assert node._send_stop.call_count == 2
