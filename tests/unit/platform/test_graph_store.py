"""Unit tests for platform/graph_store.py.

Covers:
- save_graph: assigns id, sets timestamps, writes file
- load_graph: returns None for missing, returns dict for existing
- delete_graph: removes file, returns False for missing
- list_graphs: returns summaries, skips corrupt files
- new_graph: creates empty graph
"""

import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def graphs_dir(tmp_path):
    return tmp_path / "graphs"


@pytest.fixture
def graph_store(graphs_dir, monkeypatch):
    import platform.graph_store as gs
    monkeypatch.setattr(gs, "GRAPHS_DIR", graphs_dir)
    return gs


class TestSaveAndLoad:
    def test_save_assigns_id_if_missing(self, graph_store):
        gs = graph_store
        graph = {"name": "Test Pipeline", "nodes": [], "edges": []}
        saved = gs.save_graph(graph)
        assert "id" in saved
        assert saved["id"]  # non-empty

    def test_save_preserves_existing_id(self, graph_store):
        gs = graph_store
        graph = {"id": "my-custom-id", "name": "X", "nodes": [], "edges": []}
        saved = gs.save_graph(graph)
        assert saved["id"] == "my-custom-id"

    def test_save_sets_timestamps(self, graph_store):
        gs = graph_store
        graph = {"name": "T", "nodes": [], "edges": []}
        saved = gs.save_graph(graph)
        assert "created_at" in saved
        assert "updated_at" in saved

    def test_save_and_load_round_trip(self, graph_store):
        gs = graph_store
        graph = {"name": "Inspection A", "nodes": [{"node_id": "n1"}], "edges": []}
        saved = gs.save_graph(graph)
        loaded = gs.load_graph(saved["id"])
        assert loaded["name"] == "Inspection A"
        assert len(loaded["nodes"]) == 1

    def test_load_returns_none_for_missing(self, graph_store):
        gs = graph_store
        assert gs.load_graph("nonexistent-id") is None

    def test_save_writes_valid_json_file(self, graph_store, graphs_dir):
        gs = graph_store
        graph = {"name": "JSON check", "nodes": [], "edges": []}
        saved = gs.save_graph(graph)
        path = graphs_dir / f"{saved['id']}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "JSON check"

    def test_overwrite_updates_updated_at(self, graph_store):
        gs = graph_store
        graph = {"name": "V1", "nodes": [], "edges": []}
        saved = gs.save_graph(graph)
        old_ts = saved["updated_at"]
        import time; time.sleep(0.01)
        saved["name"] = "V2"
        updated = gs.save_graph(saved)
        assert updated["updated_at"] >= old_ts  # monotonic


class TestDeleteGraph:
    def test_delete_existing_returns_true(self, graph_store):
        gs = graph_store
        graph = gs.save_graph({"name": "del", "nodes": [], "edges": []})
        assert gs.delete_graph(graph["id"]) is True
        assert gs.load_graph(graph["id"]) is None

    def test_delete_nonexistent_returns_false(self, graph_store):
        gs = graph_store
        assert gs.delete_graph("does-not-exist") is False


class TestListGraphs:
    def test_list_returns_summaries(self, graph_store):
        gs = graph_store
        gs.save_graph({"name": "A", "nodes": [], "edges": []})
        gs.save_graph({"name": "B", "nodes": [{"x": 1}], "edges": [{"y": 2}]})
        summaries = gs.list_graphs()
        assert len(summaries) == 2
        names = {s["name"] for s in summaries}
        assert names == {"A", "B"}

    def test_list_includes_node_edge_counts(self, graph_store):
        gs = graph_store
        gs.save_graph({"name": "Graph", "nodes": [{"node_id": "n1"}, {"node_id": "n2"}], "edges": [{"edge_id": "e1"}]})
        summaries = gs.list_graphs()
        assert summaries[0]["node_count"] == 2
        assert summaries[0]["edge_count"] == 1

    def test_list_skips_corrupt_files(self, graph_store, graphs_dir):
        gs = graph_store
        graphs_dir.mkdir(parents=True, exist_ok=True)
        (graphs_dir / "corrupt.json").write_text("{not json}")
        gs.save_graph({"name": "Valid", "nodes": [], "edges": []})
        summaries = gs.list_graphs()
        # Only the valid graph is returned
        assert len(summaries) == 1
        assert summaries[0]["name"] == "Valid"

    def test_list_empty_when_no_graphs(self, graph_store):
        gs = graph_store
        assert gs.list_graphs() == []


class TestNewGraph:
    def test_new_graph_is_empty(self, graph_store):
        gs = graph_store
        g = gs.new_graph("Fresh Start")
        assert g["nodes"] == []
        assert g["edges"] == []
        assert g["name"] == "Fresh Start"
        assert "id" in g
