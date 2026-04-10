"""Tests for ml_engine/export/container_builder.py."""

import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest


@pytest.fixture
def tmp_model(tmp_path):
    weights = tmp_path / "student_model" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake-weights")
    return weights


class TestBuildContext:
    def test_assemble_context_copies_all_files(self, tmp_path, tmp_model):
        from ml_engine.export.container_builder import _assemble_context, _SERVE_DIR

        # Stub out serve/ files so test works without real serve directory
        serve_dir = tmp_path / "serve"
        (serve_dir / "ros2_ws" / "src").mkdir(parents=True)
        (serve_dir / "Dockerfile.ros2").write_text("FROM ros:humble")
        (serve_dir / "entrypoint.sh").write_text("#!/bin/bash")

        with patch("ml_engine.export.container_builder._SERVE_DIR", serve_dir):
            ctx = tmp_path / "ctx"
            ctx.mkdir()
            _assemble_context(str(ctx), tmp_model)

        assert (ctx / "Dockerfile").exists()
        assert (ctx / "ros2_ws").is_dir()
        assert (ctx / "entrypoint.sh").exists()
        assert (ctx / "model" / "best.pt").exists()


class TestTagGeneration:
    def test_versioned_tag_format(self, tmp_model):
        """Image tag must be registry/yolo-inference-{job_id[:8]}:{YYYYMMDD}."""
        import re
        from unittest.mock import patch as _patch

        with _patch("ml_engine.export.container_builder._assemble_context"), \
             _patch("ml_engine.export.container_builder._run_buildx") as mock_buildx, \
             _patch("ml_engine.export.container_builder._notify_webhook"):

            mock_buildx.return_value = None

            from ml_engine.export.container_builder import build_ros2_container
            tag = build_ros2_container(
                model_weights=tmp_model,
                job_id="abc12345-full-uuid",
                registry_url="localhost:5000",
            )

        assert tag.startswith("localhost:5000/yolo-inference-abc12345:")
        # Date part must be YYYYMMDD
        date_part = tag.split(":")[-1]
        assert re.match(r"^\d{8}$", date_part), f"Expected YYYYMMDD, got {date_part}"

    def test_three_tags_pushed(self, tmp_model):
        """buildx must receive versioned, job-latest, and global-latest tags."""
        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._run_buildx") as mock_buildx, \
             patch("ml_engine.export.container_builder._notify_webhook"):

            mock_buildx.return_value = None

            from ml_engine.export.container_builder import build_ros2_container
            build_ros2_container(
                model_weights=tmp_model,
                job_id="testjob1",
                registry_url="reg:5000",
            )

        _, kwargs = mock_buildx.call_args
        tags = kwargs.get("tags") or mock_buildx.call_args[0][1]
        assert any("testjob1:latest" in t for t in tags), f"Missing :latest tag in {tags}"
        assert any("yolo-inference:latest" in t for t in tags), f"Missing global :latest in {tags}"
        assert len(tags) == 3


class TestCancelSupport:
    def test_cancel_event_kills_build(self, tmp_model):
        """If cancel_event is set before build, RuntimeError is raised."""
        cancel = threading.Event()
        cancel.set()

        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._notify_webhook"), \
             patch("subprocess.Popen") as mock_popen:

            mock_proc = MagicMock()
            mock_proc.stdout = iter([])
            mock_proc.poll.return_value = None
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            from ml_engine.export.container_builder import build_ros2_container
            with pytest.raises(RuntimeError, match="cancelled"):
                build_ros2_container(
                    model_weights=tmp_model,
                    job_id="canceltest",
                    registry_url="reg:5000",
                    cancel_event=cancel,
                )


class TestImageTagPersistence:
    def test_image_tag_written_to_file(self, tmp_model):
        """image_tag.txt must be written to output_dir after successful push."""
        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._run_buildx") as mock_buildx, \
             patch("ml_engine.export.container_builder._notify_webhook"):

            mock_buildx.return_value = None

            from ml_engine.export.container_builder import build_ros2_container
            tag = build_ros2_container(
                model_weights=tmp_model,
                job_id="persisttest",
                registry_url="reg:5000",
            )

        tag_file = tmp_model.parent.parent / "image_tag.txt"
        assert tag_file.exists(), "image_tag.txt not created"
        assert tag_file.read_text().strip() == tag


class TestBuildxFailure:
    def test_buildx_failure_raises(self, tmp_model):
        """RuntimeError from _run_buildx must propagate."""
        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._run_buildx",
                   side_effect=RuntimeError("buildx failed")), \
             patch("ml_engine.export.container_builder._notify_webhook"):

            from ml_engine.export.container_builder import build_ros2_container
            with pytest.raises(RuntimeError, match="buildx failed"):
                build_ros2_container(
                    model_weights=tmp_model,
                    job_id="failtest",
                    registry_url="reg:5000",
                )


class TestWebhook:
    def test_webhook_fires_with_external_url(self, tmp_model, monkeypatch):
        """Webhook payload must use REGISTRY_EXTERNAL_URL, not the push URL."""
        monkeypatch.setenv("DEPLOY_WEBHOOK_URL", "http://robot.local/hook")
        monkeypatch.setenv("REGISTRY_PUSH_URL", "localhost:5000")
        monkeypatch.setenv("REGISTRY_EXTERNAL_URL", "workstation:5000")

        posted_payloads = []

        def fake_urlopen(req, timeout=None):
            import json
            posted_payloads.append(json.loads(req.data))

        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._run_buildx"), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen):

            from ml_engine.export.container_builder import build_ros2_container
            build_ros2_container(
                model_weights=tmp_model,
                job_id="webhooktest",
                registry_url="localhost:5000",
            )

        # Give daemon thread time to fire
        import time; time.sleep(0.1)

        assert len(posted_payloads) == 1
        payload = posted_payloads[0]
        assert "workstation:5000" in payload["image_tag"]
        assert "localhost:5000" not in payload["image_tag"]
        assert payload["event"] == "model_deployed"

    def test_webhook_failure_is_nonfatal(self, tmp_model, monkeypatch):
        """A failed webhook POST must not raise an exception."""
        monkeypatch.setenv("DEPLOY_WEBHOOK_URL", "http://broken-host/hook")

        with patch("ml_engine.export.container_builder._assemble_context"), \
             patch("ml_engine.export.container_builder._run_buildx"), \
             patch("urllib.request.urlopen", side_effect=Exception("connection refused")):

            from ml_engine.export.container_builder import build_ros2_container
            # Must not raise
            build_ros2_container(
                model_weights=tmp_model,
                job_id="webhookfail",
                registry_url="localhost:5000",
            )
