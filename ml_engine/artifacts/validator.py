"""
Preflight validation for adapter and bundle artifacts.

Run these checks before loading any model to get clear errors
instead of cryptic PyTorch failures.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from .errors import ArtifactCorruptedError, ArtifactNotFoundError, BaseAdapterMismatch
from .schemas import AdapterManifest, BaseModelRef, BundleManifest

logger = logging.getLogger(__name__)

ADAPTER_MANIFEST_FILE = "adapter.manifest.json"
BUNDLE_MANIFEST_FILE = "bundle.manifest.json"


def validate_adapter(
    adapter_path: Path,
    expected_base: Optional[BaseModelRef] = None,
) -> AdapterManifest:
    """Validate a single adapter artifact"""
    full_path = adapter_path / ADAPTER_MANIFEST_FILE
    if not full_path.exists():
        raise ArtifactNotFoundError(full_path)

    try:
        manifest_file = AdapterManifest.load(full_path)
    except Exception as e:
        raise ArtifactCorruptedError(full_path, str(e)) from e

    for peft_file in manifest_file.peft_files.values():
        file_path = adapter_path / peft_file
        if not file_path.exists():
            raise ArtifactNotFoundError(file_path)

    if manifest_file.checksums:
        for file_name, expected_checksum in manifest_file.checksums.items():
            file_path = adapter_path / file_name
            actual_checksum = _compute_sha256(file_path)
            if actual_checksum != expected_checksum:
                raise ArtifactCorruptedError(file_path, "Checksum mismatch")

    if expected_base:
        if manifest_file.base_model.checkpoint_path != expected_base.checkpoint_path:
            raise BaseAdapterMismatch(
                adapter_path,
                expected_base.checkpoint_path,
                manifest_file.base_model.checkpoint_path,
            )
        if manifest_file.base_model.model_type != expected_base.model_type:
            raise BaseAdapterMismatch(
                adapter_path, expected_base.model_type, manifest_file.base_model.model_type
            )
        # if manifest_file.base_model.sha256 != expected_base.sha256:
        #     raise BaseAdapterMismatch(adapter_path, expected_base.sha256, manifest_file.base_model.sha256)

    return manifest_file


def validate_bundle(bundle_path: Path) -> BundleManifest:
    """Validate a complete teacher training job production"""
    full_path = bundle_path / BUNDLE_MANIFEST_FILE
    if not full_path.exists():
        raise ArtifactNotFoundError(full_path)

    try:
        manifest_file = BundleManifest.load(full_path)
    except Exception as e:
        raise ArtifactCorruptedError(full_path, str(e)) from e

    for rel_path in manifest_file.artifacts.values():
        abs_path = bundle_path / rel_path
        if not abs_path.exists():
            raise ArtifactNotFoundError(abs_path)

    if manifest_file.merged_checkpoints:
        for rel_path in manifest_file.merged_checkpoints.values():
            abs_path = bundle_path / rel_path
            if not abs_path.exists():
                raise ArtifactNotFoundError(abs_path)

    return manifest_file


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
