"""
PipelineContract and related types.

A PipelineContract is the single agreed-upon document that defines what a
pipeline will do before any compute is spent. The Coordinator creates it,
the human approves it, and the Executor/Evaluator are bound by it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TargetSpec:
    """What to detect / segment."""
    class_names: List[str]
    output_mode: str = "detection"          # "detection" | "segmentation" | "both"
    description: str = ""                   # natural language intent from user


@dataclass
class DataSpec:
    r"""Where the data lives and how to split it."""
    data_path: str
    image_paths: List[str]
    split_config: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )


@dataclass
class BudgetSpec:
    r"""Limits on the pipeline execution to guide planning and prevent runaway jobs."""
    max_epochs: int = 50
    max_trials: int = 20                    # AutoResearch budget
    max_wall_time_seconds: Optional[int] = None
    max_retries: int = 2                    # per stage


@dataclass
class AcceptanceCriteria:
    """Per-stage metric thresholds. Evaluator uses these for pass/fail."""
    min_mAP50: float = 0.5                  # GroundingDINO detection
    min_mIoU: float = 0.4                   # SAM segmentation (if used)
    max_val_loss: Optional[float] = None    # fallback if mAP not available


@dataclass
class LineageSpec:
    r"""Lineage and metadata for traceability, versioning, and reproducibility."""
    version_hash: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_contract_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PipelineContract:
    """
    The authoritative spec for a pipeline run.

    Created by the Coordinator, approved by a human via POST /api/agent/approve,
    then carried through the entire lifecycle. Evaluator reads acceptance_criteria
    to make pass/fail decisions. Executor reads budget to bound trial counts.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target: TargetSpec = field(default_factory=lambda: TargetSpec(class_names=[]))
    data: DataSpec = field(default_factory=lambda: DataSpec(data_path="", image_paths=[]))
    acceptance_criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    stage_configs: Dict[str, Any] = field(default_factory=dict)
    lineage: LineageSpec = field(default_factory=LineageSpec)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the PipelineContract to a dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineContract:
        r"""Create a PipelineContract from a dictionary, handling nested dataclasses."""
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            target=TargetSpec(**d.get("target", {})) if isinstance(d.get("target"), dict) else d.get("target", TargetSpec(class_names=[])),
            data=DataSpec(**d.get("data", {})) if isinstance(d.get("data"), dict) else d.get("data", DataSpec(data_path="", image_paths=[])),
            acceptance_criteria=AcceptanceCriteria(**d.get("acceptance_criteria", {})) if isinstance(d.get("acceptance_criteria"), dict) else AcceptanceCriteria(),
            budget=BudgetSpec(**d.get("budget", {})) if isinstance(d.get("budget"), dict) else BudgetSpec(),
            stage_configs=d.get("stage_configs", {}),
            lineage=LineageSpec(**d.get("lineage", {})) if isinstance(d.get("lineage"), dict) else LineageSpec(),
        )


@dataclass
class GateDecision:
    """Result of an Evaluator gate check."""
    verdict: str                # "pass" | "retry" | "escalate" | "pending_approval"
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class StageSummary:
    """
    Compact summary produced at stage completion.

    Carried forward into next-stage context so the Coordinator doesn't need
    full execution history. This is the context-compaction artifact.
    """
    stage: str
    status: str                             # "pass" | "retry" | "escalate" | "pending_approval"
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    key_decisions: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    trial_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the StageSummary to a dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StageSummary:
        r"""Create a StageSummary from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
