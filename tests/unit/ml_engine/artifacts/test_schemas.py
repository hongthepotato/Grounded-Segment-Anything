"""
Unit tests for ml_engine.artifacts.schemas.

Covers save/load roundtrip, required-field enforcement, and optional-field
handling. These schemas are the gate for every artifact read downstream,
so broken schema handling silently breaks teacher loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_engine.artifacts.schemas import (
    AdapterManifest,
    BaseModelRef,
    BundleManifest,
    CreateByInfo,
)


def _make_adapter(tmp_path: Path, *, with_checksums: bool = False) -> AdapterManifest:
    """Build a minimal valid AdapterManifest for round-trip tests."""
    return AdapterManifest(
        model_family="sam",
        base_model=BaseModelRef(
            checkpoint_path="pretrained/sam_vit_h.pth",
            model_type="vit_h",
            config_path=None,
        ),
        peft_files={"config": "adapter_config.json", "weights": "adapter_model.safetensors"},
        created_by=CreateByInfo(job_id="job-abc", timestamp="2026-04-23T10:00:00Z"),
        checksums={"adapter_model.safetensors": "abc123"} if with_checksums else None,
    )


class TestAdapterManifestRoundtrip:
    """AdapterManifest.save() then load() should produce an equal object."""

    def test_save_and_load_restores_all_fields(self, tmp_path: Path) -> None:
        original = _make_adapter(tmp_path)
        manifest_path = tmp_path / "adapter.manifest.json"
        original.save(manifest_path)

        loaded = AdapterManifest.load(manifest_path)

        assert loaded.model_family == original.model_family
        assert loaded.base_model.checkpoint_path == original.base_model.checkpoint_path
        assert loaded.base_model.model_type == original.base_model.model_type
        assert loaded.peft_files == original.peft_files
        assert loaded.created_by.job_id == original.created_by.job_id
        assert loaded.created_by.timestamp == original.created_by.timestamp
        assert loaded.checksums is None

    def test_save_preserves_checksums_when_present(self, tmp_path: Path) -> None:
        original = _make_adapter(tmp_path, with_checksums=True)
        manifest_path = tmp_path / "adapter.manifest.json"
        original.save(manifest_path)

        loaded = AdapterManifest.load(manifest_path)

        assert loaded.checksums == {"adapter_model.safetensors": "abc123"}

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        manifest = _make_adapter(tmp_path)
        manifest_path = tmp_path / "adapter.manifest.json"
        manifest.save(manifest_path)

        # File must be parseable as JSON (not pickle, not repr, etc.)
        with manifest_path.open() as f:
            data = json.load(f)

        assert data["model_family"] == "sam"
        assert data["base_model"]["checkpoint_path"] == "pretrained/sam_vit_h.pth"

    def test_load_rejects_malformed_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "adapter.manifest.json"
        manifest_path.write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            AdapterManifest.load(manifest_path)


class TestAdapterManifestRequiredFields:
    """Required dataclass fields must be present."""

    def test_missing_required_field_raises_typeerror(self) -> None:
        # peft_files is required; omitting it at construction time must fail.
        with pytest.raises(TypeError):
            AdapterManifest(  # type: ignore[call-arg]
                model_family="sam",
                base_model=BaseModelRef(checkpoint_path="x"),
                created_by=CreateByInfo(job_id="j", timestamp="t"),
            )

    def test_optional_fields_default_to_none(self) -> None:
        ref = BaseModelRef(checkpoint_path="x")

        assert ref.model_type is None
        assert ref.config_path is None


class TestBundleManifestRoundtrip:
    """BundleManifest save/load and optional merged_checkpoints."""

    def test_save_and_load_minimal(self, tmp_path: Path) -> None:
        original = BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"sam": "sam/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-abc"},
        )
        manifest_path = tmp_path / "bundle.manifest.json"
        original.save(manifest_path)

        loaded = BundleManifest.load(manifest_path)

        assert loaded.bundle_type == "teacher_training_output"
        assert loaded.artifacts == original.artifacts
        assert loaded.lineage == original.lineage
        assert loaded.merged_checkpoints is None

    def test_save_and_load_with_merged_checkpoints(self, tmp_path: Path) -> None:
        original = BundleManifest(
            bundle_type="teacher_training_output",
            artifacts={"sam": "sam/lora_adapters/adapter.manifest.json"},
            lineage={"job_id": "job-abc"},
            merged_checkpoints={"sam": "sam/merged/model.pth"},
        )
        manifest_path = tmp_path / "bundle.manifest.json"
        original.save(manifest_path)

        loaded = BundleManifest.load(manifest_path)

        assert loaded.merged_checkpoints == {"sam": "sam/merged/model.pth"}
