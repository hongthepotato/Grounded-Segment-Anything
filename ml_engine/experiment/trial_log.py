"""
TrialLog -- append-only experiment history.

JSON persistence on disk. Readable by the LLM Executor at Stage 4.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrialRecord:
    r"""Data class for a single trial record in the experiment log."""
    trial_id: str
    overrides: Dict[str, Any]
    primary_metric: Optional[float]
    all_metrics: Dict[str, float]
    status: str                  # "keep", "skip", "crashed", "oom"
    description: str
    wall_time_seconds: float = 0.0
    error_message: Optional[str] = None


class TrialLog:
    """
    Append-only experiment log. Backed by experiment_log.json on disk.

    Appending persists immediately -- no data loss on crash.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str,
        budget_summary: Dict[str, Any],
        baseline_metric: Optional[float] = None,
    ):
        self.run_id = run_id
        self._path = Path(output_dir) / "experiment_log.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._trials: List[TrialRecord] = []
        self._best_trial_id: Optional[str] = None
        self._best_metric: Optional[float] = None
        self._budget_summary = budget_summary
        self._baseline_metric = baseline_metric
        self._metric_mode: str = budget_summary.get("metric_mode", "max")

    def append(self, record: TrialRecord) -> None:
        """Add trial and persist to disk immediately."""
        self._trials.append(record)

        # Track best
        if record.primary_metric is not None and record.status not in ("crashed", "oom", "skip"):
            is_better = (
                self._best_metric is None
                or (self._metric_mode == "max" and record.primary_metric > self._best_metric)
                or (self._metric_mode == "min" and record.primary_metric < self._best_metric)
            )
            if is_better:
                self._best_metric = record.primary_metric
                self._best_trial_id = record.trial_id

        self._flush()

    def get_best(self) -> Optional[TrialRecord]:
        r"""Return the best trial record, or None if no trials yet."""
        if self._best_trial_id is None:
            return None
        return next((t for t in self._trials if t.trial_id == self._best_trial_id), None)

    @property
    def best_metric(self) -> Optional[float]:
        r"""Return the best primary metric value, or None if no trials yet."""
        return self._best_metric

    @property
    def trials(self) -> List[TrialRecord]:
        r"""Return a list of all trial records in order."""
        return list(self._trials)

    def to_llm_context(self) -> str:
        """
        Compact summary readable by an LLM.

        Returns a string with trial history, best result, and trends.
        """
        lines = [
            f"Experiment: {self.run_id}",
            f"Baseline: {self._baseline_metric}",
            f"Best so far: {self._best_metric} (trial {self._best_trial_id})",
            f"Trials completed: {len(self._trials)}",
            "",
            "Trial history:",
        ]
        for t in self._trials:
            metric_str = f"{t.primary_metric:.4f}" if t.primary_metric is not None else "N/A"
            lines.append(
                f"  {t.trial_id}: metric={metric_str} status={t.status} "
                f"overrides={json.dumps(t.overrides)} -- {t.description}"
            )
        return "\n".join(lines)

    def to_feedback_record(self) -> Dict[str, Any]:
        """Export for MemoryStore at Stage 4."""
        best = self.get_best()
        return {
            "run_id": self.run_id,
            "baseline_metric": self._baseline_metric,
            "best_metric": self._best_metric,
            "best_trial_id": self._best_trial_id,
            "trial_count": len(self._trials),
            "best_overrides": best.overrides if best else {},
        }

    def _flush(self) -> None:
        payload = {
            "run_id": self.run_id,
            "budget": self._budget_summary,
            "baseline_metric": self._baseline_metric,
            "best_trial_id": self._best_trial_id,
            "best_metric": self._best_metric,
            "trials": [asdict(t) for t in self._trials],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    @classmethod
    def load(cls, output_dir: str) -> "TrialLog":
        """Load from disk for crash recovery."""
        path = Path(output_dir) / "experiment_log.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        log = cls(
            run_id=data["run_id"],
            output_dir=output_dir,
            budget_summary=data.get("budget", {}),
            baseline_metric=data.get("baseline_metric"),
        )
        log._best_trial_id = data.get("best_trial_id")
        log._best_metric = data.get("best_metric")

        _fields = {f.name for f in TrialRecord.__dataclass_fields__.values()}
        for t in data.get("trials", []):
            log._trials.append(TrialRecord(**{k: v for k, v in t.items() if k in _fields}))

        logger.info("Loaded TrialLog from %s (%d trials)", path, len(log._trials))
        return log
