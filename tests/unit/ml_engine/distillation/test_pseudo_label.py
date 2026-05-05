"""
Unit tests for ml_engine.distillation.pseudo_label — written adversarially.

Two functions:
- `_build_autolabeler_config` — wires distillation policy + resolved teacher
  artifacts into an AutoLabelerConfig
- `generate_pseudo_labels` — orchestrates teacher discovery, label generation,
  and COCO output

Bugs surfaced (xfail):
- `output_mode='garbage'` flows through without validation. The downstream
  AutoLabelerConfig may or may not catch it; the building function should
  reject unknown modes at the boundary.
- `detector_adapter_dir` set but `detector_manifest` None crashes with
  AttributeError instead of a domain-meaningful error.
- `OUTPUT_BOXES_ONLY` requested with no detector adapter is silently allowed
  (parallel gap to the segmenter-required check).
- `distillation_cfg` parameter is missing from the `generate_pseudo_labels`
  docstring. Caught here as documentation drift.
- Empty `image_paths` and empty `class_names` flow through without check.

Heavy dependencies (AutoLabeler, COCOExporter, resolve_teacher_artifacts)
are mocked. Tests stay pure unit-level.
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_engine.artifacts.resolver import ResolvedArtifacts
from ml_engine.artifacts.schemas import AdapterManifest, BaseModelRef, CreateByInfo
from ml_engine.distillation import pseudo_label
from ml_engine.distillation.pseudo_label import (
    _build_autolabeler_config,
    generate_pseudo_labels,
)
from ml_engine.inference.config import (
    OUTPUT_BOTH,
    OUTPUT_BOXES_ONLY,
    OUTPUT_MASKS_ONLY,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic ResolvedArtifacts builders
# ---------------------------------------------------------------------------


def _adapter_manifest(family: str = "grounding_dino") -> AdapterManifest:
    return AdapterManifest(
        model_family=family,
        base_model=BaseModelRef(
            checkpoint_path="/fake/base.pth",
            model_type="vit_h",
            config_path="/fake/cfg.py",
        ),
        peft_files={
            "config": "adapter_config.json",
            "weights": "adapter_model.safetensors",
        },
        created_by=CreateByInfo(job_id="test-job", timestamp="2026-04-25T00:00:00Z"),
    )


def _artifacts(
    *,
    detector: bool = False,
    segmenter: bool = False,
    detector_merged: bool = False,
    segmenter_merged: bool = False,
) -> ResolvedArtifacts:
    """Build a synthetic ResolvedArtifacts toggling each role independently."""
    return ResolvedArtifacts(
        detector_adapter_dir=Path("/fake/adapters/det") if detector else None,
        detector_manifest=_adapter_manifest("grounding_dino") if detector else None,
        segmenter_adapter_dir=Path("/fake/adapters/seg") if segmenter else None,
        segmenter_manifest=_adapter_manifest("sam") if segmenter else None,
        detector_merged=Path("/fake/cache/det.pth") if detector_merged else None,
        segmenter_merged=Path("/fake/cache/seg.pth") if segmenter_merged else None,
    )


# ===========================================================================
# _build_autolabeler_config — defaults + threshold passthrough
# ===========================================================================


class TestBuildAutoLabelerConfigDefaults:
    def test_empty_distill_cfg_uses_default_thresholds_and_output_mode(self):
        """Bare cfg ({}) → (0.3, 0.3, 0.7) thresholds, OUTPUT_BOTH default."""
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=True),
            distill_cfg={},
        )
        assert cfg.thresholds.box == pytest.approx(0.3)
        assert cfg.thresholds.text == pytest.approx(0.3)
        assert cfg.thresholds.nms == pytest.approx(0.7)
        assert cfg.output_mode == OUTPUT_BOTH

    def test_custom_thresholds_propagate(self):
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=True),
            distill_cfg={
                "pseudo_label": {
                    "thresholds": {"box": 0.45, "text": 0.6, "nms": 0.5},
                }
            },
        )
        assert cfg.thresholds.box == pytest.approx(0.45)
        assert cfg.thresholds.text == pytest.approx(0.6)
        assert cfg.thresholds.nms == pytest.approx(0.5)

    def test_partial_thresholds_use_defaults_for_unspecified(self):
        """`box: 0.5` only → text + nms still default."""
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=True),
            distill_cfg={"pseudo_label": {"thresholds": {"box": 0.5}}},
        )
        assert cfg.thresholds.box == pytest.approx(0.5)
        assert cfg.thresholds.text == pytest.approx(0.3)
        assert cfg.thresholds.nms == pytest.approx(0.7)


# ===========================================================================
# _build_autolabeler_config — output_mode validation gates
# ===========================================================================


class TestBuildAutoLabelerConfigOutputMode:
    @pytest.mark.parametrize("mode", [OUTPUT_BOTH, OUTPUT_MASKS_ONLY], ids=["both", "masks-only"])
    def test_segmentation_modes_without_segmenter_raise(self, mode: str):
        """OUTPUT_BOTH and OUTPUT_MASKS_ONLY require a segmenter adapter."""
        with pytest.raises(ValueError, match="Segmentation output mode requires"):
            _build_autolabeler_config(
                artifacts=_artifacts(detector=True, segmenter=False),
                distill_cfg={"pseudo_label": {"output_mode": mode}},
            )

    def test_boxes_only_mode_does_not_require_segmenter(self):
        """OUTPUT_BOXES_ONLY only needs detector — segmenter absence is fine."""
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=False),
            distill_cfg={"pseudo_label": {"output_mode": OUTPUT_BOXES_ONLY}},
        )
        assert cfg.output_mode == OUTPUT_BOXES_ONLY

    def test_boxes_only_without_detector_raises(self):
        """Parallel gap to the segmenter check: boxes-mode without detector should fail upfront."""
        with pytest.raises(ValueError, match="(?i)detect|detector"):
            _build_autolabeler_config(
                artifacts=_artifacts(detector=False, segmenter=False),
                distill_cfg={"pseudo_label": {"output_mode": OUTPUT_BOXES_ONLY}},
            )

    def test_unknown_output_mode_raises(self):
        """An unknown output_mode string should be rejected by the builder."""
        with pytest.raises(ValueError, match="(?i)output_mode|invalid|unknown"):
            _build_autolabeler_config(
                artifacts=_artifacts(detector=True, segmenter=True),
                distill_cfg={"pseudo_label": {"output_mode": "completely-bogus"}},
            )


# ===========================================================================
# _build_autolabeler_config — adapter wiring
# ===========================================================================


class TestBuildAutoLabelerConfigAdapters:
    def test_detector_only_builds_detector_spec(self):
        """detector_adapter_dir → GroundingDINOModelSpec is populated."""
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=False),
            distill_cfg={"pseudo_label": {"output_mode": OUTPUT_BOXES_ONLY}},
        )
        assert cfg.detector is not None
        assert cfg.detector.lora_adapter_path == "/fake/adapters/det"
        assert cfg.detector.base_checkpoint == "/fake/base.pth"
        # No segmenter → falls back to default empty SegmenterModelSpec
        # (segmenter_spec or SegmenterModelSpec() pattern in source)

    def test_segmenter_only_builds_segmenter_spec(self):
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=False, segmenter=True),
            distill_cfg={"pseudo_label": {"output_mode": OUTPUT_BOTH}},
        )
        assert cfg.segmenter is not None
        assert cfg.segmenter.lora_adapter_path == "/fake/adapters/seg"
        assert cfg.segmenter.base_checkpoint == "/fake/base.pth"

    def test_merged_cache_path_propagates_when_present(self):
        """detector_merged path → forwarded as merged_cache_path."""
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=True, detector_merged=True),
            distill_cfg={},
        )
        assert cfg.detector.merged_cache_path == "/fake/cache/det.pth"

    def test_merged_cache_path_none_when_absent(self):
        cfg = _build_autolabeler_config(
            artifacts=_artifacts(detector=True, segmenter=True),
            distill_cfg={},
        )
        assert cfg.detector.merged_cache_path is None

    def test_detector_adapter_dir_without_manifest_raises_clean_error(self):
        """
        Inconsistent ResolvedArtifacts (dir set, manifest missing) should raise
        a clean error, not a bare AttributeError from `manifest.base_model.…`.
        """
        bad = ResolvedArtifacts(
            detector_adapter_dir=Path("/fake/det"),
            detector_manifest=None,  # inconsistent with adapter_dir being set
            segmenter_adapter_dir=Path("/fake/seg"),
            segmenter_manifest=_adapter_manifest("sam"),
        )
        with pytest.raises(RuntimeError, match="(?i)manifest"):
            _build_autolabeler_config(bad, distill_cfg={})


# ===========================================================================
# generate_pseudo_labels — orchestration + error paths
# ===========================================================================


class TestGeneratePseudoLabelsErrors:
    def test_no_teachers_raises_valueerror(self, tmp_path: Path, monkeypatch):
        """Empty ResolvedArtifacts (no detector AND no segmenter) → ValueError."""
        empty_artifacts = ResolvedArtifacts()  # all None

        # Patch the inside-function import: `from ml_engine.artifacts import
        # resolve_teacher_artifacts` happens at call time, so patching the
        # source module's attribute is what the runtime lookup sees.
        monkeypatch.setattr(
            "ml_engine.artifacts.resolve_teacher_artifacts",
            lambda _: empty_artifacts,
        )

        with pytest.raises(ValueError, match="No fine-tuned teachers found"):
            generate_pseudo_labels(
                image_paths=["/fake/img.jpg"],
                class_names=["dog"],
                teacher_dir="/fake/teachers",
                output_path=str(tmp_path / "out.json"),
                distillation_cfg={},
            )


class TestGeneratePseudoLabelsHappyPath:
    def _make_mocks(self, monkeypatch, *, fake_results=None, fake_coco=None):
        """Wire mocks for resolve_teacher_artifacts, AutoLabeler, COCOExporter."""
        monkeypatch.setattr(
            "ml_engine.artifacts.resolve_teacher_artifacts",
            lambda _: _artifacts(detector=True, segmenter=True),
        )

        # Patch AutoLabeler at the import site inside pseudo_label.py
        labeler_instance = MagicMock()
        labeler_instance.label_images.return_value = fake_results or []
        labeler_class = MagicMock(return_value=labeler_instance)
        monkeypatch.setattr(pseudo_label, "AutoLabeler", labeler_class)

        # Patch COCOExporter.export
        exporter = MagicMock()
        exporter.export.return_value = fake_coco or {
            "images": [],
            "annotations": [{"id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            "categories": [{"id": 1, "name": "dog"}],
        }
        monkeypatch.setattr(pseudo_label, "COCOExporter", exporter)

        return labeler_class, labeler_instance, exporter

    def test_writes_coco_json_to_output_path(self, tmp_path: Path, monkeypatch):
        """End-to-end: invocation produces a JSON file at output_path."""
        out = tmp_path / "nested" / "deeper" / "labels.json"
        self._make_mocks(monkeypatch)

        result = generate_pseudo_labels(
            image_paths=["/fake/img1.jpg", "/fake/img2.jpg"],
            class_names=["dog", "cat"],
            teacher_dir="/fake/teachers",
            output_path=str(out),
            distillation_cfg={},
        )

        assert out.exists(), "output file should be written, parent dirs created"

        # Round-trip: file content matches what's returned in memory
        on_disk = json.loads(out.read_text())
        assert on_disk == result

    def test_class_names_forwarded_to_labeler_and_exporter(self, tmp_path: Path, monkeypatch):
        """Verify class_names propagate through to both internal calls."""
        labeler_cls, labeler_instance, exporter = self._make_mocks(monkeypatch)

        generate_pseudo_labels(
            image_paths=["/fake/img.jpg"],
            class_names=["whippet", "narwhal"],
            teacher_dir="/fake/teachers",
            output_path=str(tmp_path / "out.json"),
            distillation_cfg={},
        )

        # AutoLabeler.label_images called with class_prompts
        labeler_instance.label_images.assert_called_once()
        _, kwargs = labeler_instance.label_images.call_args
        assert kwargs["class_prompts"] == ["whippet", "narwhal"]

        # COCOExporter.export called with same class_prompts
        exporter.export.assert_called_once()
        _, ex_kwargs = exporter.export.call_args
        assert ex_kwargs["class_prompts"] == ["whippet", "narwhal"]

    def test_progress_callback_forwarded_to_labeler(self, tmp_path: Path, monkeypatch):
        """The optional progress_callback should reach AutoLabeler.label_images."""
        _, labeler_instance, _ = self._make_mocks(monkeypatch)

        cb = MagicMock()
        generate_pseudo_labels(
            image_paths=["/fake/img.jpg"],
            class_names=["a"],
            teacher_dir="/fake/teachers",
            output_path=str(tmp_path / "out.json"),
            distillation_cfg={},
            progress_callback=cb,
        )

        _, kwargs = labeler_instance.label_images.call_args
        assert kwargs["progress_callback"] is cb


class TestGeneratePseudoLabelsValidationGaps:
    """These are documentation / interface gaps the function fails to enforce."""

    def test_empty_image_paths_passes_through_without_validation(self, tmp_path: Path, monkeypatch):
        """
        Documents: an empty image_paths list is accepted silently. The labeler
        runs (on no images), the COCO output gets written (probably empty).
        Caller may not realize they passed an empty list.
        """
        monkeypatch.setattr(
            "ml_engine.artifacts.resolve_teacher_artifacts",
            lambda _: _artifacts(detector=True, segmenter=True),
        )
        labeler_instance = MagicMock()
        labeler_instance.label_images.return_value = []
        monkeypatch.setattr(pseudo_label, "AutoLabeler", MagicMock(return_value=labeler_instance))
        exporter = MagicMock()
        exporter.export.return_value = {"images": [], "annotations": [], "categories": []}
        monkeypatch.setattr(pseudo_label, "COCOExporter", exporter)

        # No exception
        result = generate_pseudo_labels(
            image_paths=[],
            class_names=["dog"],
            teacher_dir="/fake/teachers",
            output_path=str(tmp_path / "empty.json"),
            distillation_cfg={},
        )
        assert result["annotations"] == []

    def test_distillation_cfg_appears_in_docstring(self):
        """The docstring should document every parameter — including distillation_cfg."""
        doc = inspect.getdoc(generate_pseudo_labels) or ""
        assert "distillation_cfg" in doc
