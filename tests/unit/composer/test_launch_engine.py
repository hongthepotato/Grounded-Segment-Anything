"""Unit tests for composer/launch_engine.py.

Covers:
- get_network_mode: Linux → host, macOS → bridge
- needs_discovery_sidecar: True when bridge
- graph_to_compose: service generation, env vars, GPU allocation, sidecar inclusion
- _build_remap_args: correct --remap arguments for source and target nodes
- domain ID pool: acquire, release, exhaustion
"""

import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def engine(monkeypatch):
    import composer.launch_engine as eng

    # Reset domain pool between tests
    monkeypatch.setattr(eng, "_used_domain_ids", set())
    return eng


SAMPLE_BLOCKS = [
    {
        "id": "usb-camera",
        "version": "1.0.0",
        "display_name": "USB Camera",
        "category": "sensor",
        "requires_gpu": False,
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
    {
        "id": "yolo-inference",
        "version": "1.0.0",
        "display_name": "YOLO",
        "category": "detection",
        "requires_gpu": True,
        "ports": {
            "input": [
                {
                    "id": "image_in",
                    "alias": "image_frame",
                    "ros2_type": "sensor_msgs/Image",
                    "label": "Camera Input",
                }
            ],
            "output": [
                {
                    "id": "detections_out",
                    "alias": "detections",
                    "ros2_type": "vision_msgs/Detection2DArray",
                    "label": "Detections",
                }
            ],
        },
    },
]

SAMPLE_GRAPH = {
    "nodes": [
        {"node_id": "node-cam-1", "block_id": "usb-camera", "block_version": "1.0.0", "params": {"fps": 30}},
        {
            "node_id": "node-yolo-1",
            "block_id": "yolo-inference",
            "block_version": "1.0.0",
            "params": {"confidence": 0.5},
        },
    ],
    "edges": [
        {
            "edge_id": "e1",
            "source_node_id": "node-cam-1",
            "source_port_id": "image_out",
            "target_node_id": "node-yolo-1",
            "target_port_id": "image_in",
        },
    ],
}


class TestNetworkMode:
    def test_linux_returns_host(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        assert engine.get_network_mode() == "host"

    def test_macos_returns_bridge(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        assert engine.get_network_mode() == "bridge"

    def test_windows_returns_bridge(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Windows")
        assert engine.get_network_mode() == "bridge"

    def test_linux_no_sidecar(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        assert engine.needs_discovery_sidecar() is False

    def test_macos_needs_sidecar(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        assert engine.needs_discovery_sidecar() is True


class TestBuildRemapArgs:
    def test_source_node_gets_remap_for_output_port(self, engine):
        graph = {
            "nodes": [{"node_id": "n1"}, {"node_id": "n2"}],
            "edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_port_id": "image_out",
                    "target_node_id": "n2",
                    "target_port_id": "image_in",
                }
            ],
        }
        args = engine._build_remap_args(graph, "n1", "testpipe")
        assert "--ros-args" in args
        assert "--remap image_out:=/pipeline/testpipe/n1/image_out" in args

    def test_target_node_gets_remap_for_input_port(self, engine):
        graph = {
            "edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_port_id": "image_out",
                    "target_node_id": "n2",
                    "target_port_id": "image_in",
                }
            ],
        }
        args = engine._build_remap_args(graph, "n2", "testpipe")
        assert "--remap image_in:=/pipeline/testpipe/n1/image_out" in args

    def test_unrelated_node_gets_empty_args(self, engine):
        graph = {
            "edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_port_id": "p_out",
                    "target_node_id": "n2",
                    "target_port_id": "p_in",
                }
            ],
        }
        args = engine._build_remap_args(graph, "n_other", "testpipe")
        assert args == ""

    def test_multiple_edges_generate_multiple_remaps(self, engine):
        graph = {
            "edges": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "source_port_id": "out1",
                    "target_node_id": "n2",
                    "target_port_id": "in1",
                },
                {
                    "edge_id": "e2",
                    "source_node_id": "n1",
                    "source_port_id": "out2",
                    "target_node_id": "n3",
                    "target_port_id": "in1",
                },
            ],
        }
        args = engine._build_remap_args(graph, "n1", "pipe")
        assert "--remap out1:=" in args
        assert "--remap out2:=" in args


class TestGraphToCompose:
    def test_services_created_for_each_node(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        services = result["compose"]["services"]
        assert len(services) == 2
        service_names = list(services.keys())
        assert any("usb-camera" in n for n in service_names)
        assert any("yolo-inference" in n for n in service_names)

    def test_ros_domain_id_injected(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc in result["compose"]["services"].values():
            assert "ROS_DOMAIN_ID" in svc["environment"]

    def test_params_injected_as_env_vars(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc_name, svc in result["compose"]["services"].items():
            if "usb-camera" in svc_name:
                assert svc["environment"].get("PARAM_FPS") == "30"
            if "yolo-inference" in svc_name:
                assert svc["environment"].get("PARAM_CONFIDENCE") == "0.5"

    def test_gpu_block_gets_nvidia_reservation(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc_name, svc in result["compose"]["services"].items():
            if "yolo-inference" in svc_name:
                assert "deploy" in svc
                devices = svc["deploy"]["resources"]["reservations"]["devices"]
                assert devices[0]["driver"] == "nvidia"

    def test_non_gpu_block_no_deploy_section(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc_name, svc in result["compose"]["services"].items():
            if "usb-camera" in svc_name:
                assert "deploy" not in svc

    def test_linux_uses_host_network(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc in result["compose"]["services"].values():
            assert svc.get("network_mode") == "host"
            assert "networks" not in svc

    def test_macos_adds_discovery_sidecar(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        services = result["compose"]["services"]
        assert "discovery" in services
        discovery = services["discovery"]
        assert "fastdds_discovery_server" in discovery["command"]

    def test_macos_block_services_connect_to_sidecar(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        for svc_name, svc in result["compose"]["services"].items():
            if svc_name == "discovery":
                continue
            env = svc["environment"]
            assert env.get("ROS_DISCOVERY_SERVER") == "discovery:11811"
            assert env.get("RMW_IMPLEMENTATION") == "rmw_fastrtps_cpp"

    def test_missing_block_raises_value_error(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        graph = {"nodes": [{"node_id": "n1", "block_id": "nonexistent", "params": {}}], "edges": []}
        with pytest.raises(ValueError, match="not found in catalog"):
            engine.graph_to_compose(graph, SAMPLE_BLOCKS)

    def test_returns_pipeline_metadata(self, engine, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        result = engine.graph_to_compose(SAMPLE_GRAPH, SAMPLE_BLOCKS)
        assert "pipeline_id" in result
        assert "domain_id" in result
        assert "project_name" in result
        assert result["project_name"].startswith("pipeline-")


class TestDomainIdPool:
    def test_acquire_returns_unique_ids(self, engine):
        id1 = engine._acquire_domain_id()
        id2 = engine._acquire_domain_id()
        assert id1 != id2
        assert 1 <= id1 <= 230
        assert 1 <= id2 <= 230

    def test_release_makes_id_available_again(self, engine):
        domain_id = engine._acquire_domain_id()
        engine._release_domain_id(domain_id)
        next_id = engine._acquire_domain_id()
        # The released ID should be reusable
        assert next_id == domain_id

    def test_pool_exhaustion_raises_runtime_error(self, engine):
        # Fill the entire pool
        for _ in range(230):
            engine._acquire_domain_id()
        with pytest.raises(RuntimeError, match="No available ROS_DOMAIN_ID"):
            engine._acquire_domain_id()
