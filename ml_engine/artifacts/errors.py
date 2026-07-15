"""
Error taxonomy for artifact resolution and validation.

All errors inherit from ArtifactError so callers can catch the
entire domain with a single except clause
"""

from pathlib import Path
from typing import Optional


class ArtifactError(Exception):
    """Base class for artifact-related errors."""

    def __init__(self, message: str, artifact_path: Optional[Path] = None):
        self.artifact_path = artifact_path
        super().__init__(message)


class ArtifactNotFoundError(ArtifactError):
    """Required artifact file or directory does not exist."""

    def __init__(self, path: Path, detail: str = ""):
        self.detail = detail
        msg = f"Artifact not found: {path}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, artifact_path=path)


class ArtifactCorruptedError(ArtifactError):
    """Artifact exists but is unreadble, has bad schema, or fails checksum"""

    def __init__(self, path: Path, reason: str = ""):
        self.reason = reason
        super().__init__(f"Corrupt artifact at {path}: {reason}", artifact_path=path)


class BaseAdapterMismatch(ArtifactError):
    """Adapter is structurally incompatible with the requested base model"""

    def __init__(self, adapter_path: Path, expected: Optional[str], actual: Optional[str]):
        # expected/actual are Optional because BaseModelRef fields
        # (checkpoint_path, model_type) are Optional — a None on either side
        # is a legitimate mismatch case (e.g., an adapter without model_type
        # metadata being checked against a base that requires it).
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Base/adapter mismatch detected: {adapter_path}:expected {expected}, got {actual}",
            artifact_path=adapter_path,
        )
