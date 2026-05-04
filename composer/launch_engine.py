"""Launch engine — converts a block graph to docker-compose.yml and runs it.

Key behaviors:
- Linux: network_mode=host (multicast DDS discovery works natively)
- macOS/Windows: bridge network + discovery server sidecar per pipeline

Topic naming: /pipeline/{uuid}/{source_node_id}/{port_id}
Each wire is injected via ROS2_REMAP_ARGS into source and target containers.

ROS_DOMAIN_ID pool: 0–230, tracked in memory. Reclaimed on pipeline stop.
"""

import asyncio
import logging
import platform
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path(__file__).parent.parent / "graphs"
COMPOSE_CACHE_DIR = Path("/tmp/ros2-platform/compose")

# In-memory domain ID pool. Lost on restart — acceptable for v1 (single workstation).
_used_domain_ids: Set[int] = set()


def _acquire_domain_id() -> int:
    for i in range(1, 231):
        if i not in _used_domain_ids:
            _used_domain_ids.add(i)
            return i
    raise RuntimeError("No available ROS_DOMAIN_ID in pool 1-230")


def _release_domain_id(domain_id: int) -> None:
    _used_domain_ids.discard(domain_id)


def get_network_mode() -> str:
    """Linux → host networking. macOS/Windows → bridge + sidecar."""
    if platform.system() == "Linux":
        return "host"
    return "bridge"


def needs_discovery_sidecar() -> bool:
    return get_network_mode() == "bridge"


def _build_remap_args(graph: Dict, node_id: str, pipeline_id: str) -> str:
    """Build the ROS2_REMAP_ARGS string for a given node.

    For each edge where this node is source or target, generates:
      --remap {local_port_id}:=/pipeline/{pipeline_id}/{source_node_id}/{source_port_id}
    """
    remaps = []
    for edge in graph.get("edges", []):
        topic_name = f"/pipeline/{pipeline_id}/{edge['source_node_id']}/{edge['source_port_id']}"
        if edge["source_node_id"] == node_id:
            remaps.append(f"--remap {edge['source_port_id']}:={topic_name}")
        elif edge["target_node_id"] == node_id:
            remaps.append(f"--remap {edge['target_port_id']}:={topic_name}")

    if remaps:
        return "--ros-args " + " ".join(remaps)
    return ""


def graph_to_compose(graph: Dict[str, Any], blocks_catalog: List[Dict]) -> Dict[str, Any]:
    """Convert a block graph to a docker-compose dict.

    Returns the compose dict (suitable for yaml.dump) and the pipeline metadata.
    """
    pipeline_id = str(uuid.uuid4())[:8]
    domain_id = _acquire_domain_id()
    network_mode = get_network_mode()
    sidecar = needs_discovery_sidecar()

    compose: Dict[str, Any] = {
        "version": "3.8",
        "services": {},
    }

    if sidecar:
        compose["networks"] = {f"pipeline-{pipeline_id}": {"driver": "bridge"}}
        compose["services"]["discovery"] = {
            "image": "host-gateway:5000/ros2-block-base:humble",
            "command": "ros2 run rmw_fastrtps_cpp fastdds_discovery_server -i 0 -p 11811",
            "networks": [f"pipeline-{pipeline_id}"],
            "restart": "unless-stopped",
        }

    # Build a map of block_id -> block definition
    block_map = {b["id"]: b for b in blocks_catalog}

    for node in graph.get("nodes", []):
        node_id = node["node_id"]
        block_id = node["block_id"]
        block = block_map.get(block_id)
        if block is None:
            raise ValueError(f"Block '{block_id}' not found in catalog")

        block_version = node.get("block_version", block.get("version", "latest"))
        image_ref = block.get("image_ref") or f"host-gateway:5000/{block_id}:{block_version}"
        remap_args = _build_remap_args(graph, node_id, pipeline_id)

        env: Dict[str, str] = {
            "ROS_DOMAIN_ID": str(domain_id),
            "ROS2_REMAP_ARGS": remap_args,
        }

        # Inject parameters as PARAM_{ID} env vars
        params = node.get("params", {})
        for key, value in params.items():
            env[f"PARAM_{key.upper()}"] = str(value)

        # Discovery server env for macOS/Windows
        if sidecar:
            env["ROS_DISCOVERY_SERVER"] = "discovery:11811"
            env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

        service: Dict[str, Any] = {
            "image": image_ref,
            "environment": env,
            "restart": "unless-stopped",
        }

        if network_mode == "host":
            service["network_mode"] = "host"
        else:
            service["networks"] = [f"pipeline-{pipeline_id}"]
            if sidecar:
                service["depends_on"] = ["discovery"]

        # GPU allocation for blocks that require it
        if block.get("requires_gpu"):
            service["deploy"] = {
                "resources": {
                    "reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}
                }
            }

        # Volume mount for result-logger (data persistence)
        if block_id == "result-logger":
            service["volumes"] = ["./data:/data"]

        service_name = f"{block_id}-{node_id[:6]}"
        compose["services"][service_name] = service

    return {
        "compose": compose,
        "pipeline_id": pipeline_id,
        "domain_id": domain_id,
        "project_name": f"pipeline-{pipeline_id}",
    }


def _write_compose_file(pipeline_id: str, compose: Dict) -> Path:
    COMPOSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = COMPOSE_CACHE_DIR / f"pipeline-{pipeline_id}.yml"
    with open(path, "w") as f:
        yaml.dump(compose, f, default_flow_style=False)
    return path


async def launch_pipeline(graph: Dict, blocks_catalog: List[Dict]) -> Dict[str, Any]:
    """Generate compose file and run the pipeline. Returns pipeline metadata."""
    result = graph_to_compose(graph, blocks_catalog)
    compose_path = _write_compose_file(result["pipeline_id"], result["compose"])

    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "-p",
        result["project_name"],
        "up",
        "-d",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        _release_domain_id(result["domain_id"])
        raise RuntimeError(f"docker compose up failed (exit {proc.returncode}): {stderr.decode()}")

    logger.info(f"Pipeline {result['pipeline_id']} started on domain {result['domain_id']}")
    return {
        "pipeline_id": result["pipeline_id"],
        "project_name": result["project_name"],
        "domain_id": result["domain_id"],
        "compose_file": str(compose_path),
    }


async def stop_pipeline(project_name: str, domain_id: Optional[int] = None) -> None:
    compose_path = COMPOSE_CACHE_DIR / f"{project_name}.yml"
    args = ["docker", "compose", "-p", project_name, "down"]
    if compose_path.exists():
        args = ["docker", "compose", "-f", str(compose_path), "-p", project_name, "down"]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    if domain_id is not None:
        _release_domain_id(domain_id)
    logger.info(f"Pipeline {project_name} stopped")


async def get_pipeline_status(project_name: str) -> List[Dict[str, Any]]:
    """Poll docker compose ps and return service status list."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-p",
        project_name,
        "ps",
        "--format",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return []

    import json

    services = []
    for line in stdout.decode().strip().splitlines():
        if line.strip():
            try:
                services.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return services


async def get_service_logs(project_name: str, service_name: str, tail: int = 100) -> str:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-p",
        project_name,
        "logs",
        "--no-log-prefix",
        "--tail",
        str(tail),
        service_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()


async def restart_service(project_name: str, service_name: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "compose",
        "-p",
        project_name,
        "restart",
        service_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
