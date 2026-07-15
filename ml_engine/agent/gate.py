"""
Pure gate evaluation logic -- no Redis, no side effects.

Shared between RequestEvaluationTool (Coordinator) and EvaluatorWorker (Stage 3).
Keeping this as a standalone module avoids circular imports between
coordinator.py and workers.py.
"""

from __future__ import annotations

from typing import Dict

from ml_engine.agent.contracts import AcceptanceCriteria, GateDecision
from ml_engine.agent.state_machine import DISTILLATION_GATE_STAGES, TEACHER_GATE_STAGES


def _retries_left_str(remaining: int) -> str:
    return f"{remaining} {'retry' if remaining == 1 else 'retries'} left"


def evaluate_gate(
    metrics: Dict[str, float],
    criteria: AcceptanceCriteria,
    retry_count: int,
    max_retries: int,
    stage: str,
) -> GateDecision:
    """
    Deterministic pass/fail gate. No LLM needed.

    Returns GateDecision with verdict in: "pass", "retry", "escalate".
    LLM is optional only for anomaly interpretation after the verdict is known.
    """
    mAP50 = next((metrics[k] for k in ("mAP50", "best_metric", "val_mAP50") if k in metrics), None)
    mIoU = next((metrics[k] for k in ("mIoU", "val_mIoU") if k in metrics), None)

    # Teacher model gate: mAP50 is the upstream signal for all teacher stages,
    # even when SAM is also being fine-tuned. GroundingDINO box quality gates the
    # whole pipeline -- bad proposals mean SAM gets bad prompts regardless.
    if stage in TEACHER_GATE_STAGES:
        if mAP50 is None:
            return GateDecision(
                verdict="escalate",
                reason="val_mAP50 missing from outcome metrics",
                metrics=metrics,
                retry_count=retry_count,
            )
        if mAP50 >= criteria.min_mAP50:
            return GateDecision(
                verdict="pass",
                reason=f"mAP50 {mAP50:.3f} >= threshold {criteria.min_mAP50:.3f}",
                metrics=metrics,
                retry_count=retry_count,
            )
        if retry_count < max_retries:
            return GateDecision(
                verdict="retry",
                reason=(
                    f"mAP50 {mAP50:.3f} below threshold {criteria.min_mAP50:.3f}, "
                    f"{_retries_left_str(max_retries - retry_count)}"
                ),
                metrics=metrics,
                retry_count=retry_count,
            )
        return GateDecision(
            verdict="escalate",
            reason=f"mAP50 {mAP50:.3f} below threshold {criteria.min_mAP50:.3f} after {retry_count} retries",
            metrics=metrics,
            retry_count=retry_count,
        )

    # Student distillation gate: either SAM (mIoU) or GroundingDINO (mAP50)
    # depending on which component is being compressed.
    # Pick whichever metric is present; escalate if neither is.
    if stage in DISTILLATION_GATE_STAGES:
        if mIoU is not None:
            primary, threshold, name = mIoU, criteria.min_mIoU, "mIoU"
        elif mAP50 is not None:
            primary, threshold, name = mAP50, criteria.min_mAP50, "mAP50"
        else:
            return GateDecision(
                verdict="escalate",
                reason="neither mIoU nor mAP50 found in distillation outcome metrics",
                metrics=metrics,
                retry_count=retry_count,
            )
        if primary >= threshold:
            return GateDecision(
                verdict="pass",
                reason=f"{name} {primary:.3f} >= threshold {threshold:.3f}",
                metrics=metrics,
                retry_count=retry_count,
            )
        if retry_count < max_retries:
            return GateDecision(
                verdict="retry",
                reason=(
                    f"{name} {primary:.3f} below {threshold:.3f}, "
                    f"{_retries_left_str(max_retries - retry_count)}"
                ),
                metrics=metrics,
                retry_count=retry_count,
            )
        return GateDecision(
            verdict="escalate",
            reason=f"{name} {primary:.3f} below threshold {threshold:.3f} after {retry_count} retries",
            metrics=metrics,
            retry_count=retry_count,
        )

    # Default: pass (e.g. auto_labeling uses a different quality bar)
    return GateDecision(
        verdict="pass",
        reason=f"no metric threshold defined for stage '{stage}' — passing by default",
        metrics=metrics,
        retry_count=retry_count,
    )
