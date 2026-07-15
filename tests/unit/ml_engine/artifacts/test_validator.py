"""
Unit tests for ml_engine.artifacts.validator.

validator.py is the preflight gate for every adapter + bundle load downstream.
If validation wrongly accepts a corrupted manifest, teacher training loads
a mismatched checkpoint and downstream training silently produces wrong results.
Tests below cover: valid pass-through, missing files, corrupted manifests,
checksum mismatches, and base/adapter mismatches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml_engine.artifacts.errors import (
    ArtifactCorruptedError,
    ArtifactNotFoundError,
    BaseAdapterMismatch,
)
from ml_engine.artifacts.schemas import (
    AdapterManifest,
    BaseModelRef,
    BundleManifest,
    CreateByInfo,
)
from ml_engine.artifacts.validator import validate_adapter, validate_bundle

# ============================================================================
# Fixtures — write manifests to disk matching the directory layout documented
# in schemas.py.
# ============================================================================


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    """Build a valid adapter directory with manifest + referenced files."""
    d = tmp_path / "lora_adapters"
    d.mkdir()

    # Write referenced peft files.
    (d / "adapter_config.json").write_text('{"r": 16}')
    (d / "adapter_model.safetensors").write_text("fake-weights")

    # Write the manifest pointing at them.
    AdapterManifest(
        model_family="sam",
        base_model=BaseModelRef(
            checkpoint_path="pretrained/sam_vit_h.pth",
            model_type="vit_h",
        ),
        peft_files={"config": "adapter_config.json", "weights": "adapter_model.safetensors"},
        created_by=CreateByInfo(job_id="job-abc", timestamp="2026-04-23T10:00:00Z"),
    ).save(d / "adapter.manifest.json")
    return d


@pytest.fixture
def bundle_dir(tmp_path: Path, adapter_dir: Path) -> Path:
    """Build a valid bundle directory referencing the adapter_dir."""
    bundle = tmp_path / "teacher_bundle"
    bundle.mkdir()

    # Re-point the adapter_dir into the bundle.
    bundle_adapter = bundle / "sam" / "lora_adapters"
    bundle_adapter.parent.mkdir(parents=True)
    bundle_adapter.symlink_to(adapter_dir)

    BundleManifest(
        bundle_type="teacher_training_output",
        artifacts={"sam": "sam/lora_adapters/adapter.manifest.json"},
        lineage={"job_id": "job-abc"},
    ).save(bundle / "bundle.manifest.json")
    return bundle


# ============================================================================
# validate_adapter
# ============================================================================


class TestValidateAdapterHappyPath:
    def test_returns_loaded_manifest(self, adapter_dir: Path) -> None:
        manifest = validate_adapter(adapter_dir)

        assert manifest.model_family == "sam"
        assert manifest.base_model.model_type == "vit_h"
        assert manifest.peft_files["config"] == "adapter_config.json"


class TestValidateAdapterMissing:
    def test_missing_manifest_raises_not_found(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ArtifactNotFoundError) as excinfo:
            validate_adapter(empty)

        assert str(empty / "adapter.manifest.json") in str(excinfo.value)

    def test_missing_peft_file_raises_not_found(self, adapter_dir: Path) -> None:
        # Remove a referenced peft file.
        (adapter_dir / "adapter_model.safetensors").unlink()

        with pytest.raises(ArtifactNotFoundError):
            validate_adapter(adapter_dir)


class TestValidateAdapterCorrupted:
    def test_malformed_manifest_json_raises_corrupted(self, adapter_dir: Path) -> None:
        (adapter_dir / "adapter.manifest.json").write_text("{not json")

        with pytest.raises(ArtifactCorruptedError):
            validate_adapter(adapter_dir)

    def test_checksum_mismatch_raises_corrupted(self, adapter_dir: Path) -> None:
        # Write a manifest whose checksum won't match the file on disk.
        AdapterManifest(
            model_family="sam",
            base_model=BaseModelRef(checkpoint_path="x", model_type="vit_h"),
            peft_files={"weights": "adapter_model.safetensors"},
            created_by=CreateByInfo(job_id="j", timestamp="t"),
            checksums={"adapter_model.safetensors": "0" * 64},  # bogus sha256
        ).save(adapter_dir / "adapter.manifest.json")

        with pytest.raises(ArtifactCorruptedError) as excinfo:
            validate_adapter(adapter_dir)

        assert "checksum" in str(excinfo.value).lower()

    def test_valid_checksum_passes(self, adapter_dir: Path) -> None:
        # Compute actual sha256 of the weights file and embed it in the manifest.
        weights = adapter_dir / "adapter_model.safetensors"
        actual_sha = hashlib.sha256(weights.read_bytes()).hexdigest()

        AdapterManifest(
            model_family="sam",
            base_model=BaseModelRef(checkpoint_path="x", model_type="vit_h"),
            peft_files={"weights": "adapter_model.safetensors"},
            created_by=CreateByInfo(job_id="j", timestamp="t"),
            checksums={"adapter_model.safetensors": actual_sha},
        ).save(adapter_dir / "adapter.manifest.json")

        # Should not raise.
        validate_adapter(adapter_dir)


class TestValidateAdapterBaseMismatch:
    def test_checkpoint_path_mismatch_raises(self, adapter_dir: Path) -> None:
        expected = BaseModelRef(checkpoint_path="different/path.pth", model_type="vit_h")

        with pytest.raises(BaseAdapterMismatch) as excinfo:
            validate_adapter(adapter_dir, expected_base=expected)

        assert "different/path.pth" in str(excinfo.value)

    def test_model_type_mismatch_raises(self, adapter_dir: Path) -> None:
        expected = BaseModelRef(
            checkpoint_path="pretrained/sam_vit_h.pth",
            model_type="vit_l",  # disagrees with manifest
        )

        with pytest.raises(BaseAdapterMismatch):
            validate_adapter(adapter_dir, expected_base=expected)

    def test_matching_expected_base_passes(self, adapter_dir: Path) -> None:
        expected = BaseModelRef(
            checkpoint_path="pretrained/sam_vit_h.pth",
            model_type="vit_h",
        )

        manifest = validate_adapter(adapter_dir, expected_base=expected)

        assert manifest.base_model.model_type == "vit_h"


# ============================================================================
# validate_bundle
# ============================================================================


class TestValidateBundleHappyPath:
    def test_returns_loaded_manifest(self, bundle_dir: Path) -> None:
        manifest = validate_bundle(bundle_dir)

        assert manifest.bundle_type == "teacher_training_output"
        assert "sam" in manifest.artifacts


class TestValidateBundleMissing:
    def test_missing_bundle_manifest_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_bundle"
        empty.mkdir()

        with pytest.raises(ArtifactNotFoundError):
            validate_bundle(empty)

    def test_missing_referenced_artifact_raises(self, bundle_dir: Path) -> None:
        # Point the bundle manifest at a file that doesn't exist.
        BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"sam": "sam/lora_adapters/nonexistent.json"},
            lineage={"job_id": "job-abc"},
        ).save(bundle_dir / "bundle.manifest.json")

        with pytest.raises(ArtifactNotFoundError):
            validate_bundle(bundle_dir)

    def test_missing_merged_checkpoint_raises(self, bundle_dir: Path) -> None:
        BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"sam": "sam/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-abc"},
            merged_checkpoints={"sam": "sam/merged/does_not_exist.pth"},
        ).save(bundle_dir / "bundle.manifest.json")

        with pytest.raises(ArtifactNotFoundError):
            validate_bundle(bundle_dir)


class TestValidateBundleCorrupted:
    def test_malformed_bundle_manifest_raises(self, bundle_dir: Path) -> None:
        (bundle_dir / "bundle.manifest.json").write_text("{not json")

        with pytest.raises(ArtifactCorruptedError):
            validate_bundle(bundle_dir)
