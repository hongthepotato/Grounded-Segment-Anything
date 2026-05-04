"""Tests for /build-ros2 and /deploy-info API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_job_factory(tmp_path):
    """Return a factory that creates mock Job objects."""

    def _make(status="completed", output_dir=None, ros2_image_tag=None):
        job = MagicMock()
        job.id = "abc12345-0000-0000-0000-000000000000"
        job.status.value = status
        job.output_dir = str(output_dir or tmp_path)
        job.ros2_image_tag = ros2_image_tag
        return job

    return _make


@pytest.fixture
def client(mock_job_factory):
    """TestClient with mocked JobManager."""
    from fastapi import FastAPI

    from api.routes.exports import router

    app = FastAPI()
    app.include_router(router)

    mock_manager = MagicMock()
    mock_manager.get_job.return_value = mock_job_factory()

    with patch("api.routes.exports.get_job_manager", return_value=mock_manager):
        with TestClient(app) as c:
            c._mock_manager = mock_manager
            yield c


class TestBuildRos2Endpoint:
    def test_returns_202_when_student_pt_exists(self, client, mock_job_factory, tmp_path):
        # Create best.pt
        (tmp_path / "student_model").mkdir()
        (tmp_path / "student_model" / "best.pt").write_bytes(b"weights")

        job = mock_job_factory(output_dir=tmp_path)
        client._mock_manager.get_job.return_value = job

        with patch("api.routes.exports.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            resp = client.post(f"/api/jobs/{job.id}/build-ros2")

        assert resp.status_code == 202
        assert resp.json()["data"]["status"] == "building"

    def test_returns_404_when_no_student_model(self, client, mock_job_factory, tmp_path):
        job = mock_job_factory(output_dir=tmp_path)
        client._mock_manager.get_job.return_value = job

        resp = client.post(f"/api/jobs/{job.id}/build-ros2")
        assert resp.status_code == 404
        assert "best.pt" in resp.json()["detail"]

    def test_returns_400_for_running_job(self, client, mock_job_factory):
        job = mock_job_factory(status="running")
        client._mock_manager.get_job.return_value = job

        resp = client.post(f"/api/jobs/{job.id}/build-ros2")
        assert resp.status_code == 400


class TestDeployInfoEndpoint:
    def test_returns_deploy_info_from_redis(self, client, mock_job_factory, tmp_path, monkeypatch):
        monkeypatch.setenv("REGISTRY_PUSH_URL", "localhost:5000")
        monkeypatch.setenv("REGISTRY_EXTERNAL_URL", "workstation:5000")

        job = mock_job_factory(ros2_image_tag="localhost:5000/yolo-inference-abc12345:20260406")
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "workstation:5000" in data["image_tag"]
        assert "localhost:5000" not in data["image_tag"]
        assert "pull_command" in data
        assert "run_command" in data
        assert "-v yolo-cache:/model/cache" in data["run_command"]
        assert "--ros-args" in data["run_command"]

    def test_returns_deploy_info_from_filesystem(self, client, mock_job_factory, tmp_path, monkeypatch):
        """Fallback: read image_tag.txt when Redis field is absent."""
        monkeypatch.setenv("REGISTRY_PUSH_URL", "localhost:5000")
        monkeypatch.setenv("REGISTRY_EXTERNAL_URL", "workstation:5000")

        (tmp_path / "image_tag.txt").write_text("localhost:5000/yolo-inference-abc12345:20260406")

        job = mock_job_factory(output_dir=tmp_path, ros2_image_tag=None)
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        assert resp.status_code == 200
        assert "workstation:5000" in resp.json()["data"]["image_tag"]

    def test_returns_404_when_no_container_built(self, client, mock_job_factory, tmp_path):
        job = mock_job_factory(output_dir=tmp_path, ros2_image_tag=None)
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        assert resp.status_code == 404
        assert "build-ros2" in resp.json()["detail"]

    def test_topics_included_in_response(self, client, mock_job_factory, monkeypatch):
        monkeypatch.setenv("REGISTRY_PUSH_URL", "localhost:5000")

        job = mock_job_factory(ros2_image_tag="localhost:5000/yolo-inference-abc12345:20260406")
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        data = resp.json()["data"]
        assert "topics" in data
        assert "diagnostics" in data["topics"]

    def test_setup_script_url_in_response(self, client, mock_job_factory, monkeypatch):
        monkeypatch.setenv("REGISTRY_EXTERNAL_URL", "myrobot:5000")

        job = mock_job_factory(ros2_image_tag="localhost:5000/yolo-inference-abc12345:20260406")
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        data = resp.json()["data"]
        assert "setup_script_url" in data
        assert "myrobot" in data["setup_script_url"]

    def test_trt_volume_warning_in_notes(self, client, mock_job_factory, monkeypatch):
        monkeypatch.setenv("REGISTRY_PUSH_URL", "localhost:5000")

        job = mock_job_factory(ros2_image_tag="localhost:5000/yolo-inference-abc12345:20260406")
        client._mock_manager.get_job.return_value = job

        resp = client.get(f"/api/jobs/{job.id}/deploy-info")
        notes = resp.json()["data"]["notes"]
        assert "yolo-cache" in notes
        assert "required" in notes.lower()
