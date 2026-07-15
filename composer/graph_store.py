"""Graph persistence — saves/loads block graphs as JSON files in graphs/.

Each graph is stored as graphs/{graph_id}.json.
No database required. Files can be backed up with cp.

Graph JSON format:
{
  "id": "<uuid>",
  "name": "Inspection Line A",
  "created_at": "<iso8601>",
  "updated_at": "<iso8601>",
  "nodes": [
    {
      "node_id": "<uuid>",
      "block_id": "usb-camera",
      "block_version": "1.0.0",
      "x": 100, "y": 200,
      "params": {"device_index": 0, "fps": 30}
    }
  ],
  "edges": [
    {
      "edge_id": "<uuid>",
      "source_node_id": "<uuid>",
      "source_port_id": "image_out",
      "target_node_id": "<uuid>",
      "target_port_id": "image_in"
    }
  ]
}
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path(__file__).parent.parent / "graphs"


def _ensure_dir():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def list_graphs() -> List[Dict[str, Any]]:
    """Return summary of all saved graphs (id, name, updated_at)."""
    _ensure_dir()
    summaries = []
    for path in sorted(GRAPHS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                g = json.load(f)
            summaries.append(
                {
                    "id": g.get("id", path.stem),
                    "name": g.get("name", path.stem),
                    "updated_at": g.get("updated_at", ""),
                    "node_count": len(g.get("nodes", [])),
                    "edge_count": len(g.get("edges", [])),
                }
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load graph {path}: {e}")
    return summaries


def load_graph(graph_id: str) -> Optional[Dict[str, Any]]:
    _ensure_dir()
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Save or update a graph. Assigns id if missing. Returns saved graph."""
    _ensure_dir()
    now = datetime.now(timezone.utc).isoformat()
    if "id" not in graph or not graph["id"]:
        graph["id"] = str(uuid.uuid4())
    if "created_at" not in graph:
        graph["created_at"] = now
    graph["updated_at"] = now

    path = GRAPHS_DIR / f"{graph['id']}.json"
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    logger.info(f"Saved graph '{graph.get('name', graph['id'])}' to {path}")
    return graph


def delete_graph(graph_id: str) -> bool:
    path = GRAPHS_DIR / f"{graph_id}.json"
    if path.exists():
        path.unlink()
        logger.info(f"Deleted graph {graph_id}")
        return True
    return False


def new_graph(name: str) -> Dict[str, Any]:
    """Create and save a blank graph."""
    graph = {
        "name": name,
        "nodes": [],
        "edges": [],
    }
    return save_graph(graph)
