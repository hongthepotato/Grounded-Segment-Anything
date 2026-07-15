"""
Unit tests for ml_engine.artifacts.resolver.

resolver.py glues bundle + adapter validation together. If resolution
silently returns wrong paths, downstream loading grabs the wrong adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_engine.artifacts.resolver import resolve_teacher_artifacts
from ml_engine.artifacts.schemas import (
    AdapterManifest,
    BaseModelRef,
    BundleManifest,
    CreateByInfo,
)


def _write_adapter(dir_: Path, model_family: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text('{"r": 16}')
    (dir_ / "adapter_model.safetensors").write_text("fake-weights")
    AdapterManifest(
        model_family=model_family,
        base_model=BaseModelRef(checkpoint_path=f"pretrained/{model_family}.pth", model_type="vit_h"),
        peft_files={"config": "adapter_config.json", "weights": "adapter_model.safetensors"},
        created_by=CreateByInfo(job_id="job-abc", timestamp="2026-04-23T10:00:00Z"),
    ).save(dir_ / "adapter.manifest.json")


@pytest.fixture
def combined_teacher_dir(tmp_path: Path) -> Path:
    """A teacher_dir with both detector (grounding_dino) and segmenter (sam) adapters."""
    root = tmp_path / "teacher_training_combined"
    root.mkdir()

    _write_adapter(root / "grounding_dino" / "lora_adapters", "grounding_dino")
    _write_adapter(root / "sam" / "lora_adapters", "sam")

    BundleManifest(
        bundle_type="teacher_training_output",
        artifacts={
            "grounding_dino": "grounding_dino/lora_adapters/adapter.manifest.json",
            "sam": "sam/lora_adapters/adapter.manifest.json",
        },
        lineage={"job_id": "job-abc"},
    ).save(root / "bundle.manifest.json")
    return root


class TestResolveCombinedTeacher:
    def test_finds_both_adapters(self, combined_teacher_dir: Path) -> None:
        resolved = resolve_teacher_artifacts(str(combined_teacher_dir))

        assert resolved.has_detector
        assert resolved.has_segmenter
        assert resolved.detector_adapter_dir == combined_teacher_dir / "grounding_dino" / "lora_adapters"
        assert resolved.segmenter_adapter_dir == combined_teacher_dir / "sam" / "lora_adapters"

    def test_manifests_populated(self, combined_teacher_dir: Path) -> None:
        resolved = resolve_teacher_artifacts(str(combined_teacher_dir))

        assert resolved.detector_manifest is not None
        assert resolved.detector_manifest.model_family == "grounding_dino"
        assert resolved.segmenter_manifest is not None
        assert resolved.segmenter_manifest.model_family == "sam"


class TestResolveDetectorOnly:
    def test_only_detector_present(self, tmp_path: Path) -> None:
        root = tmp_path / "teacher_training_det"
        root.mkdir()
        _write_adapter(root / "grounding_dino" / "lora_adapters", "grounding_dino")

        BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"grounding_dino": "grounding_dino/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-det"},
        ).save(root / "bundle.manifest.json")

        resolved = resolve_teacher_artifacts(str(root))

        assert resolved.has_detector
        assert not resolved.has_segmenter
        assert resolved.segmenter_manifest is None


class TestResolveFailureModes:
    def test_missing_bundle_manifest_raises_filenotfound(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_teacher"
        empty.mkdir()

        with pytest.raises(FileNotFoundError):
            resolve_teacher_artifacts(str(empty))

    def test_unknown_model_name_in_bundle_raises_valueerror(self, tmp_path: Path) -> None:
        root = tmp_path / "teacher_unknown"
        root.mkdir()
        _write_adapter(root / "mystery_model" / "lora_adapters", "mystery_model")

        BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"mystery_model": "mystery_model/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-x"},
        ).save(root / "bundle.manifest.json")

        with pytest.raises(ValueError, match="Unknown model name"):
            resolve_teacher_artifacts(str(root))


class TestResolveMergedCheckpoints:
    def test_merged_paths_resolved(self, combined_teacher_dir: Path) -> None:
        # Add merged checkpoints to the bundle.
        (combined_teacher_dir / "grounding_dino" / "merged").mkdir()
        (combined_teacher_dir / "grounding_dino" / "merged" / "model.pth").write_text("x")
        (combined_teacher_dir / "sam" / "merged").mkdir()
        (combined_teacher_dir / "sam" / "merged" / "model.pth").write_text("x")

        BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={
                "grounding_dino": "grounding_dino/lora_adapters/adapter.manifest.json",
                "sam": "sam/lora_adapters/adapter.manifest.json",
            },
            lineage={"job_id": "job-abc"},
            merged_checkpoints={
                "grounding_dino": "grounding_dino/merged/model.pth",
                "sam": "sam/merged/model.pth",
            },
        ).save(combined_teacher_dir / "bundle.manifest.json")

        resolved = resolve_teacher_artifacts(str(combined_teacher_dir))

        assert resolved.detector_merged == combined_teacher_dir / "grounding_dino" / "merged" / "model.pth"
        assert resolved.segmenter_merged == combined_teacher_dir / "sam" / "merged" / "model.pth"
