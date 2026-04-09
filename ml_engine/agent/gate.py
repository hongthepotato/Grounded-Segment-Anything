"""
Pure gate evaluation logic -- no Redis, no side effects.

Shared between RequestEvaluationTool (Coordinator) and EvaluatorWorker (Stage 3).
Keeping this as a standalone module avoids circular imports between
coordinator.py and workers.py.
"""

from __future__ import annotations

from typing import Any, Dict

from ml_engine.agent.contracts import AcceptanceCriteria, GateDecision


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
    mAP50 = metrics.get("mAP50") or metrics.get("best_metric") or metrics.get("val_mAP50")
    mIoU = metrics.get("mIoU") or metrics.get("val_mIoU")

    # Detection gate
    if stage in ("teacher_training", "training_eval_gate", "experiment_loop"):
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
                reason=f"mAP50 {mAP50:.3f} >= threshold {criteria.min_mAP50}",
                metrics=metrics,
                retry_count=retry_count,
            )
        if retry_count < max_retries:
            return GateDecision(
                verdict="retry",
                reason=(
                    f"mAP50 {mAP50:.3f} below threshold {criteria.min_mAP50}, "
                    f"{max_retries - retry_count} retries left"
                ),
                metrics=metrics,
                retry_count=retry_count,
            )
        return GateDecision(
            verdict="escalate",
            reason=f"mAP50 {mAP50:.3f} below threshold after {retry_count} retries",
            metrics=metrics,
            retry_count=retry_count,
        )

    # Segmentation gate
    if stage in ("student_distillation", "distill_eval_gate") and mIoU is not None:
        if mIoU >= criteria.min_mIoU:
            return GateDecision(
                verdict="pass",
                reason=f"mIoU {mIoU:.3f} >= {criteria.min_mIoU}",
                metrics=metrics,
                retry_count=retry_count,
            )
        if retry_count < max_retries:
            return GateDecision(
                verdict="retry",
                reason=f"mIoU {mIoU:.3f} below {criteria.min_mIoU}",
                metrics=metrics,
                retry_count=retry_count,
            )
        return GateDecision(
            verdict="escalate",
            reason="mIoU budget exhausted",
            metrics=metrics,
            retry_count=retry_count,
        )

    # Default: pass (e.g. auto_labeling uses a different quality bar)
    return GateDecision(
        verdict="pass",
        reason="no metric threshold defined for this stage",
        metrics=metrics,
        retry_count=retry_count,
    )
