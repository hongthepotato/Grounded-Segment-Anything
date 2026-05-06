"""Unit tests for composer/registry.py.

Covers:
- aliases_compatible: matching aliases, mismatched aliases, annotated_image restriction
- scan_blocks: happy path, invalid JSON, schema validation errors, skip sdk/ and base/ dirs
- detect_hardware: command not found, timeout, success paths
"""

import json
import subprocess

# Patch the blocks dir path before importing registry so it uses a temp dir
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def blocks_dir(tmp_path):
    """Create a minimal blocks/ directory for testing."""
    return tmp_path / "blocks"


@pytest.fixture
def registry_module(blocks_dir, monkeypatch):
    """Import registry with patched _BLOCKS_DIR pointing at tmp blocks dir."""
    # Ensure platform package can be found
    repo_root = Path(__file__).parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Reload with patched path
    import composer.registry as reg

    monkeypatch.setattr(reg, "_BLOCKS_DIR", blocks_dir)
    monkeypatch.setattr(reg, "_schema", None)  # skip schema validation in most tests
    return reg


class TestAliasesCompatible:
    def test_matching_aliases_are_compatible(self, registry_module):
        reg = registry_module
        assert reg.aliases_compatible("image_frame", "image_frame") is True
        assert reg.aliases_compatible("detections", "detections") is True
        assert reg.aliases_compatible("flag_event", "flag_event") is True

    def test_mismatched_aliases_are_incompatible(self, registry_module):
        reg = registry_module
        assert reg.aliases_compatible("image_frame", "detections") is False
        assert reg.aliases_compatible("detections", "flag_event") is False

    def test_annotated_image_cannot_connect_to_image_frame(self, registry_module):
        reg = registry_module
        # annotated_image output → image_frame input is disallowed (visualization only)
        assert reg.aliases_compatible("annotated_image", "image_frame") is False
        assert reg.aliases_compatible("image_frame", "annotated_image") is False

    def test_annotated_image_connects_to_annotated_image(self, registry_module):
        reg = registry_module
        # Connecting two annotated_image ports to each other is valid
        assert reg.aliases_compatible("annotated_image", "annotated_image") is True

    def test_unknown_alias_is_incompatible(self, registry_module):
        reg = registry_module
        assert reg.aliases_compatible("unknown_alias", "image_frame") is False
        assert reg.aliases_compatible("image_frame", "totally_made_up") is False


class TestScanBlocks:
    def _make_block(self, blocks_dir, block_id, block_data):
        block_dir = blocks_dir / block_id
        block_dir.mkdir(parents=True)
        with open(block_dir / "block.json", "w") as f:
            json.dump(block_data, f)
        return block_dir

    def test_scan_finds_valid_blocks(self, registry_module, blocks_dir):
        reg = registry_module
        self._make_block(
            blocks_dir,
            "usb-camera",
            {
                "id": "usb-camera",
                "version": "1.0.0",
                "display_name": "USB Camera",
                "category": "sensor",
                "ports": {
                    "output": [
                        {
                            "id": "image_out",
                            "alias": "image_frame",
                            "ros2_type": "sensor_msgs/Image",
                            "label": "Camera",
                        }
                    ]
                },
            },
        )
        blocks = reg.scan_blocks()
        assert len(blocks) == 1
        assert blocks[0]["id"] == "usb-camera"
        assert blocks[0]["_status"] == "ok"

    def test_scan_skips_sdk_and_base_dirs(self, registry_module, blocks_dir):
        reg = registry_module
        # These dirs should be skipped
        for skip_name in ["sdk", "base", "base-cuda"]:
            d = blocks_dir / skip_name
            d.mkdir(parents=True)
            with open(d / "block.json", "w") as f:
                json.dump(
                    {
                        "id": skip_name,
                        "version": "1.0.0",
                        "display_name": "X",
                        "category": "sensor",
                        "ports": {},
                    },
                    f,
                )
        blocks = reg.scan_blocks()
        assert blocks == []

    def test_scan_skips_invalid_json(self, registry_module, blocks_dir):
        reg = registry_module
        bad_dir = blocks_dir / "bad-block"
        bad_dir.mkdir(parents=True)
        with open(bad_dir / "block.json", "w") as f:
            f.write("{this is not json}")
        blocks = reg.scan_blocks()
        assert blocks == []

    def test_scan_returns_empty_when_no_blocks(self, registry_module, blocks_dir):
        reg = registry_module
        blocks_dir.mkdir(parents=True, exist_ok=True)
        blocks = reg.scan_blocks()
        assert blocks == []

    def test_scan_multiple_blocks(self, registry_module, blocks_dir):
        reg = registry_module
        for bid in ["cam-a", "cam-b", "detector"]:
            self._make_block(
                blocks_dir,
                bid,
                {"id": bid, "version": "1.0.0", "display_name": bid, "category": "sensor", "ports": {}},
            )
        blocks = reg.scan_blocks()
        assert len(blocks) == 3

    def test_scan_marks_validation_errors(self, registry_module, blocks_dir, monkeypatch):
        reg = registry_module

        def always_fail(block):
            return ["test validation error"]

        monkeypatch.setattr(reg, "_validate_block", always_fail)
        self._make_block(
            blocks_dir,
            "bad-schema",
            {"id": "bad-schema", "version": "1.0.0", "display_name": "X", "category": "sensor", "ports": {}},
        )
        blocks = reg.scan_blocks()
        assert blocks[0]["_status"] == "error"
        assert "_validation_errors" in blocks[0]


class TestDetectHardware:
    def test_no_hardware_when_commands_missing(self, registry_module, monkeypatch):
        reg = registry_module
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=FileNotFoundError))
        import glob as glob_mod

        monkeypatch.setattr(glob_mod, "glob", MagicMock(return_value=[]))
        hw = reg.detect_hardware()
        assert hw == []

    def test_usb_cam_detected_via_dev_video(self, registry_module, monkeypatch):
        reg = registry_module
        import glob as glob_mod

        monkeypatch.setattr(glob_mod, "glob", MagicMock(return_value=["/dev/video0"]))
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=FileNotFoundError))
        hw = reg.detect_hardware()
        assert "usb_cam" in hw

    def test_realsense_detected_when_command_succeeds(self, registry_module, monkeypatch):
        reg = registry_module
        import glob as glob_mod

        monkeypatch.setattr(glob_mod, "glob", MagicMock(return_value=[]))

        def mock_run(cmd, **kwargs):
            if cmd[0] == "rs-enumerate-devices":
                return MagicMock(returncode=0, stdout="Intel RealSense D435")
            raise FileNotFoundError

        monkeypatch.setattr("subprocess.run", mock_run)
        hw = reg.detect_hardware()
        assert "realsense" in hw

    def test_timeout_does_not_crash(self, registry_module, monkeypatch):
        reg = registry_module
        import glob as glob_mod

        monkeypatch.setattr(glob_mod, "glob", MagicMock(return_value=[]))
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 3)))
        hw = reg.detect_hardware()
        assert hw == []


class TestGetBlockById:
    def test_returns_block_when_found(self, registry_module):
        reg = registry_module
        blocks = [{"id": "usb-camera"}, {"id": "yolo-inference"}]
        assert reg.get_block_by_id(blocks, "usb-camera")["id"] == "usb-camera"

    def test_returns_none_when_not_found(self, registry_module):
        reg = registry_module
        assert reg.get_block_by_id([], "anything") is None
        assert reg.get_block_by_id([{"id": "x"}], "y") is None
