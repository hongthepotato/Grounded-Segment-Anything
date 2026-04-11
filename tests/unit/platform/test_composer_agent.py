"""Unit tests for platform/composer_agent.py.

Covers:
- create_session / get_session: session creation, TTL expiry
- _execute_tool: all 7 tools (search_blocks, get_block_details, validate_connection,
  add_block, connect, get_current_graph, finalize_graph)
- validate_connection: compatible aliases, incompatible aliases, annotated_image restriction
- check_llm_connectivity: success and failure paths
"""

import json
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

BLOCKS_CATALOG = [
    {
        "id": "usb-camera", "version": "1.0.0", "display_name": "USB Camera",
        "category": "sensor",
        "ports": {
            "output": [{"id": "image_out", "alias": "image_frame", "ros2_type": "sensor_msgs/Image", "label": "Camera"}]
        },
    },
    {
        "id": "yolo-inference", "version": "1.0.0", "display_name": "YOLO",
        "category": "detection",
        "ports": {
            "input": [{"id": "image_in", "alias": "image_frame", "ros2_type": "sensor_msgs/Image", "label": "Input"}],
            "output": [
                {"id": "detections_out", "alias": "detections", "ros2_type": "vision_msgs/Detection2DArray", "label": "Dets"},
                {"id": "annotated_out", "alias": "annotated_image", "ros2_type": "sensor_msgs/Image", "label": "View"},
            ],
        },
    },
    {
        "id": "defect-flagging", "version": "1.0.0", "display_name": "Defect Flagging",
        "category": "logic",
        "ports": {
            "input": [{"id": "detections_in", "alias": "detections", "ros2_type": "vision_msgs/Detection2DArray", "label": "Dets"}],
            "output": [{"id": "flag_out", "alias": "flag_event", "ros2_type": "std_msgs/Bool", "label": "Flag"}],
        },
    },
]


@pytest.fixture
def agent(monkeypatch):
    import platform.composer_agent as ca
    monkeypatch.setattr(ca, "_sessions", {})
    return ca


def make_session(agent, prompt="test"):
    sid = agent.create_session(prompt, BLOCKS_CATALOG)
    return sid, agent.get_session(sid)


class TestSessionLifecycle:
    def test_create_session_returns_string_id(self, agent):
        sid = agent.create_session("test prompt", BLOCKS_CATALOG)
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_get_session_returns_created_session(self, agent):
        sid, session = make_session(agent)
        assert session is not None
        assert session["session_id"] == sid
        assert session["prompt"] == "test"
        assert session["finalized"] is False

    def test_get_session_returns_none_for_unknown_id(self, agent):
        assert agent.get_session("unknown-id") is None

    def test_session_expires_after_ttl(self, agent, monkeypatch):
        sid = agent.create_session("old prompt", BLOCKS_CATALOG)
        # Manually expire the session by backdating created_at
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=11)
        agent._sessions[sid]["created_at"] = expired_time
        assert agent.get_session(sid) is None


class TestExecuteTool:
    def test_search_blocks_returns_full_catalog(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("search_blocks", {"query": "camera"}, session, BLOCKS_CATALOG)
        assert "blocks" in result
        assert result["count"] == len(BLOCKS_CATALOG)

    def test_get_block_details_returns_block(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("get_block_details", {"block_id": "usb-camera"}, session, BLOCKS_CATALOG)
        assert result["id"] == "usb-camera"

    def test_get_block_details_returns_error_for_unknown(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("get_block_details", {"block_id": "nonexistent"}, session, BLOCKS_CATALOG)
        assert "error" in result

    def test_validate_connection_compatible_aliases(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("validate_connection", {
            "source_block_id": "usb-camera", "source_port_id": "image_out",
            "target_block_id": "yolo-inference", "target_port_id": "image_in",
        }, session, BLOCKS_CATALOG)
        assert result["valid"] is True

    def test_validate_connection_incompatible_aliases(self, agent):
        _, session = make_session(agent)
        # annotated_image → image_frame: disallowed
        result = agent._execute_tool("validate_connection", {
            "source_block_id": "yolo-inference", "source_port_id": "annotated_out",
            "target_block_id": "yolo-inference", "target_port_id": "image_in",
        }, session, BLOCKS_CATALOG)
        assert result["valid"] is False

    def test_validate_connection_detections_to_defect(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("validate_connection", {
            "source_block_id": "yolo-inference", "source_port_id": "detections_out",
            "target_block_id": "defect-flagging", "target_port_id": "detections_in",
        }, session, BLOCKS_CATALOG)
        assert result["valid"] is True

    def test_validate_connection_missing_source_block(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("validate_connection", {
            "source_block_id": "nonexistent", "source_port_id": "out",
            "target_block_id": "yolo-inference", "target_port_id": "image_in",
        }, session, BLOCKS_CATALOG)
        assert result["valid"] is False
        assert "not found" in result["reason"]

    def test_add_block_creates_node_in_graph(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("add_block", {"block_id": "usb-camera", "params": {"fps": 30}}, session, BLOCKS_CATALOG)
        assert "node_id" in result
        assert result["event"] == "block_added"
        assert len(session["graph"]["nodes"]) == 1
        node = session["graph"]["nodes"][0]
        assert node["block_id"] == "usb-camera"
        assert node["params"]["fps"] == 30

    def test_add_block_unknown_block_returns_error(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("add_block", {"block_id": "ghost-block"}, session, BLOCKS_CATALOG)
        assert "error" in result
        assert len(session["graph"]["nodes"]) == 0

    def test_connect_creates_edge_when_valid(self, agent):
        _, session = make_session(agent)
        # Add two nodes first
        r1 = agent._execute_tool("add_block", {"block_id": "usb-camera", "params": {}}, session, BLOCKS_CATALOG)
        r2 = agent._execute_tool("add_block", {"block_id": "yolo-inference", "params": {}}, session, BLOCKS_CATALOG)
        cam_id = r1["node_id"]
        yolo_id = r2["node_id"]
        result = agent._execute_tool("connect", {
            "source_node_id": cam_id, "source_port_id": "image_out",
            "target_node_id": yolo_id, "target_port_id": "image_in",
        }, session, BLOCKS_CATALOG)
        assert "edge_id" in result
        assert result["event"] == "connected"
        assert len(session["graph"]["edges"]) == 1

    def test_connect_fails_for_incompatible_aliases(self, agent):
        _, session = make_session(agent)
        r1 = agent._execute_tool("add_block", {"block_id": "usb-camera", "params": {}}, session, BLOCKS_CATALOG)
        r2 = agent._execute_tool("add_block", {"block_id": "defect-flagging", "params": {}}, session, BLOCKS_CATALOG)
        result = agent._execute_tool("connect", {
            "source_node_id": r1["node_id"], "source_port_id": "image_out",
            "target_node_id": r2["node_id"], "target_port_id": "detections_in",
        }, session, BLOCKS_CATALOG)
        assert "error" in result
        assert len(session["graph"]["edges"]) == 0

    def test_get_current_graph_returns_state(self, agent):
        _, session = make_session(agent)
        agent._execute_tool("add_block", {"block_id": "usb-camera", "params": {}}, session, BLOCKS_CATALOG)
        result = agent._execute_tool("get_current_graph", {}, session, BLOCKS_CATALOG)
        assert "graph" in result
        assert len(result["graph"]["nodes"]) == 1

    def test_finalize_graph_marks_finalized(self, agent):
        _, session = make_session(agent)
        assert session["finalized"] is False
        result = agent._execute_tool("finalize_graph", {}, session, BLOCKS_CATALOG)
        assert session["finalized"] is True
        assert result["event"] == "finalized"
        assert "graph" in result

    def test_unknown_tool_returns_error(self, agent):
        _, session = make_session(agent)
        result = agent._execute_tool("fake_tool", {}, session, BLOCKS_CATALOG)
        assert "error" in result


class TestConnectivity:
    def test_llm_available_when_socket_connects(self, agent, monkeypatch):
        import socket
        monkeypatch.setattr(socket, "create_connection", MagicMock(return_value=MagicMock()))
        assert agent.check_llm_connectivity() is True

    def test_llm_unavailable_on_socket_error(self, agent, monkeypatch):
        import socket
        monkeypatch.setattr(socket, "create_connection", MagicMock(side_effect=OSError))
        assert agent.check_llm_connectivity() is False
