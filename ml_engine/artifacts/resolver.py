"""
Artifact discovery for teacher training outputs.

Accept a teacher_dir and returns a structured set of discovered
artifacts, using manifests when available or falling back to
legacy folder probing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ml_engine.artifacts import AdapterManifest, validate_bundle, validate_adapter, BUNDLE_MANIFEST_FILE

logger = logging.getLogger(__name__)

@dataclass
class ResolvedArtifacts:
    """Result of artifact resolution for a teacher_dir."""
    detector_adapter_dir: Optional[Path] = None
    detector_manifest: Optional[AdapterManifest] = None
    segmenter_adapter_dir: Optional[Path] = None
    segmenter_manifest: Optional[AdapterManifest] = None
    detector_merged: Optional[Path] = None
    segmenter_merged: Optional[Path] = None

    @property
    def has_detector(self) -> bool:
        return self.detector_adapter_dir is not None

    @property
    def has_segmenter(self) -> bool:
        return self.segmenter_adapter_dir is not None


def resolve_teacher_artifacts(teacher_dir: str) -> ResolvedArtifacts:
    """
    Discover and validate teacher artifaces from a job output directory.

    Strategy:
    1. If bundle.manifest.json exists -> manifest-based resolution
    2. Otherwise -> legacy folder probing with warning
    """
    root = Path(teacher_dir)
    bundle_path = root / BUNDLE_MANIFEST_FILE

    if bundle_path.exists():
        return _resolve_from_bundle(root)

    # logger.warning("No bundle manifest at %s, falling back to legacy discovery", root)
    # return None

def _resolve_from_bundle(root: Path) -> ResolvedArtifacts:
    """Resolve artifacts from a bundle manifest"""
    bundle_manifest = validate_bundle(root)
    artifacts = ResolvedArtifacts()

    for model_name, rel_path in bundle_manifest.artifacts.items():
        manifest_abs_path = root / rel_path
        adapter_dir = manifest_abs_path.parent
        if model_name == "grounding_dino":
            artifacts.detector_adapter_dir = adapter_dir
            artifacts.detector_manifest = validate_adapter(adapter_dir)
        elif model_name == "sam":
            artifacts.segmenter_adapter_dir = adapter_dir
            artifacts.segmenter_manifest = validate_adapter(adapter_dir)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    if bundle_manifest.merged_checkpoints:
        for model_name, rel_path in bundle_manifest.merged_checkpoints.items():
            merged_path = root / rel_path
            if model_name == "grounding_dino":
                artifacts.detector_merged = merged_path
            elif model_name == "sam":
                artifacts.segmenter_merged = merged_path
            else:
                raise ValueError(f"Unknown model name: {model_name}")

    return artifacts
