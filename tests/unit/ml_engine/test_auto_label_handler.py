"""Bug-hunting tests for AutoLabelHandler's path handling.

The handler must keep TWO path forms straight:
  - the RAW/logical path the caller supplied -> handed to AutoLabeler, which
    reports it as the COCO file_name (so annotations.json stays portable);
  - the RESOLVED filesystem path (transform_image_path) -> used to check the file
    exists and to read it for visualization.

Regression trap: the handler used to append the RESOLVED path into image_paths
and pass that to label_images, leaking absolute filesystem paths like
"/srv/shared/images/upload/a.jpg" into the COCO file_name.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_engine.jobs.handlers.auto_label import AutoLabelHandler

RAW_PATHS = ["upload/2025/a.jpg", "upload/2025/b.jpg"]


@pytest.fixture
def image_root(tmp_path):
    """Real files on disk — the handler validates existence against the RESOLVED
    path, so the fake transform must point at something that exists."""
    root = tmp_path / "store"
    root.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (root / name).touch()
    return root


def _run(tmp_path, image_root, raw_paths=RAW_PATHS):
    """Run the handler with every heavy collaborator mocked.

    Returns (labeler_instance, visualize_mock, resolve_fn).
    """

    def resolve(p):  # logical -> real file on disk (deliberately NOT identity)
        return str(image_root / Path(p).name)

    labeler = MagicMock()
    labeler.label_images.return_value = [{"class_ids": [0]} for _ in raw_paths]
    exporter = MagicMock()
    exporter.export.return_value = {"images": [], "annotations": []}
    viz = MagicMock()

    with (
        patch("ml_engine.inference.AutoLabeler", MagicMock(return_value=labeler)),
        patch("ml_engine.inference.COCOExporter", exporter),
        patch("ml_engine.inference.visualize_detections", viz),
        patch("core.constants.transform_image_path", side_effect=resolve),
        patch("core.config.save_json"),
    ):
        AutoLabelHandler().run(
            job_id="job-1",
            job_config={"image_paths": list(raw_paths), "classes": ["cat"]},
            output_dir=str(tmp_path / "out"),
            progress_queue=MagicMock(),
            cancel_event=MagicMock(**{"is_set.return_value": False}),
        )
    return labeler, viz, resolve


class TestPathHandling:
    def test_labeler_receives_raw_logical_paths(self, tmp_path, image_root):
        # BUG TRAP: pre-fix the handler passed the RESOLVED absolute paths, so the
        # COCO file_name became "/.../store/a.jpg" instead of "upload/2025/a.jpg".
        labeler, _, _ = _run(tmp_path, image_root)
        passed = labeler.label_images.call_args.kwargs["image_paths"]
        assert passed == RAW_PATHS

    def test_visualization_uses_resolved_filesystem_path(self, tmp_path, image_root):
        # Visualization reads the image, so it MUST get the resolved path — the
        # fix must not "fix" file_name by breaking the image read.
        _, viz, resolve = _run(tmp_path, image_root)
        used = [c.kwargs["image_path"] for c in viz.call_args_list]
        assert used == [resolve(p) for p in RAW_PATHS]
        assert all(Path(p).exists() for p in used)

    def test_missing_resolved_file_raises(self, tmp_path, image_root):
        with pytest.raises(ValueError, match="Image path not found"):
            _run(tmp_path, image_root, raw_paths=["upload/2025/missing.jpg"])
