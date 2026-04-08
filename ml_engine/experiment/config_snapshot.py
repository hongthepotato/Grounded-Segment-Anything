"""
ConfigSnapshot -- in-memory config versioning for the experiment loop.

Replaces git keep/revert. No git at runtime.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigSnapshot:
    """
    Lightweight in-memory config versioning.

    Captures config dicts by snapshot ID, tracks the best-performing one,
    and persists the best config to disk so crash recovery can resume from it.

    Usage::
        snap = ConfigSnapshot(output_dir="experiments/exp_001")
        sid = snap.capture(config)          # snapshot current config
        snap.mark_best(sid, metric=0.72)    # mark it as best
        best = snap.get_best()              # retrieve best config
        restored = snap.restore(sid)        # restore any snapshot
    """

    def __init__(self, output_dir: str):
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._best_snapshot_id: Optional[str] = None
        self._best_metric: Optional[float] = None
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, config: Dict[str, Any]) -> str:
        """Snapshot current config. Returns snapshot_id."""
        snapshot_id = f"snap_{len(self._snapshots):04d}"
        self._snapshots[snapshot_id] = copy.deepcopy(config)
        logger.debug("Captured snapshot %s", snapshot_id)
        return snapshot_id

    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """Return a deep copy of the config at snapshot_id."""
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Unknown snapshot: {snapshot_id}")
        return copy.deepcopy(self._snapshots[snapshot_id])

    def mark_best(self, snapshot_id: str, metric: float) -> None:
        """
        Mark snapshot_id as the current best.

        Also persists best_config.yaml to disk for crash recovery.
        """
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Unknown snapshot: {snapshot_id}")
        self._best_snapshot_id = snapshot_id
        self._best_metric = metric

        best_path = self._output_dir / "best_config.yaml"
        with open(best_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._snapshots[snapshot_id], f)
        logger.info("New best config (metric=%.4f) written to %s", metric, best_path)

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the best config, or None if none marked."""
        if self._best_snapshot_id is None:
            return None
        return copy.deepcopy(self._snapshots[self._best_snapshot_id])

    @property
    def best_metric(self) -> Optional[float]:
        return self._best_metric

    @classmethod
    def load_best_from_disk(cls, output_dir: str) -> Optional[Dict[str, Any]]:
        """
        Load best_config.yaml from a previous run for crash recovery.

        Returns None if the file doesn't exist.
        """
        best_path = Path(output_dir) / "best_config.yaml"
        if not best_path.exists():
            return None
        with open(best_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded best config from %s (crash recovery)", best_path)
        return config
