"""
Unit tests for ml_engine.agent.gate.evaluate_gate.

Pure function -- no Redis, no I/O. Exhaustive corner cases.
"""

from __future__ import annotations

import pytest

from ml_engine.agent.contracts import AcceptanceCriteria
from ml_engine.agent.gate import evaluate_gate
from ml_engine.agent.state_machine import (
    DISTILLATION_GATE_STAGES,
    STATES,
    TEACHER_GATE_STAGES,
    TERMINAL_STATES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def criteria(min_mAP50: float = 0.5, min_mIoU: float = 0.4) -> AcceptanceCriteria:
    r"""Convenience factory for AcceptanceCriteria with defaults matching our typical contract."""
    return AcceptanceCriteria(min_mAP50=min_mAP50, min_mIoU=min_mIoU)


TEACHER_STAGES = sorted(TEACHER_GATE_STAGES)
DISTILLATION_STAGES = sorted(DISTILLATION_GATE_STAGES)
OTHER_STAGES = sorted(
    STATES
    - TEACHER_GATE_STAGES
    - DISTILLATION_GATE_STAGES
    - TERMINAL_STATES
    - {"created", "planning", "pending_contract_approval", "failed_retrying"}
)


# ---------------------------------------------------------------------------
# Detection gate
# ---------------------------------------------------------------------------


class TestTeacherGate:
    r"""Tests for the teacher training gate, which uses mAP50 as the signal metric."""

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_pass_when_mAP50_meets_threshold(self, stage):
        r"""mAP50 above threshold should pass."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.72},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"
        assert "0.720" in decision.reason
        assert "threshold 0.500" in decision.reason
        assert decision.retry_count == 0

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_pass_at_exact_threshold(self, stage):
        r"""mAP50 exactly at threshold should pass."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.5},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"
        assert decision.retry_count == 0

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_retry_when_below_threshold_and_retries_left(self, stage):
        r"""mAP50 below threshold with retries left should retry."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.3},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "retry"
        assert "2 retries left" in decision.reason

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_escalate_when_retries_exhausted(self, stage):
        r"""mAP50 below threshold with no retries left should escalate."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.3},
            criteria=criteria(min_mAP50=0.5),
            retry_count=2,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "escalate"

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_escalate_when_mAP50_missing(self, stage):
        r"""If mAP50 missing from metrics, should escalate (can't retry on missing signal)."""
        decision = evaluate_gate(
            metrics={},
            criteria=criteria(),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "escalate"
        assert "missing" in decision.reason.lower()

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_accepts_best_metric_alias(self, stage):
        r"""outcome.json may use 'best_metric' instead of 'mAP50'."""
        decision = evaluate_gate(
            metrics={"best_metric": 0.65},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"

    @pytest.mark.parametrize("stage", TEACHER_STAGES)
    def test_accepts_val_mAP50_alias(self, stage):
        r"""outcome.json may use 'val_mAP50' instead of 'mAP50'."""
        decision = evaluate_gate(
            metrics={"val_mAP50": 0.55},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"

    def test_retry_count_propagated_to_decision(self):
        r"""Retry count should be included in the GateDecision for tracking and reporting."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.2},
            criteria=criteria(min_mAP50=0.5),
            retry_count=1,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.retry_count == 1
        assert "1 retry left" in decision.reason  # singular

    def test_metrics_propagated_to_decision(self):
        r"""All metrics should be included in the GateDecision for reporting."""
        m = {"mAP50": 0.45, "val_loss": 1.2}
        decision = evaluate_gate(
            metrics=m,
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.metrics["mAP50"] == 0.45
        assert decision.metrics["val_loss"] == 1.2


# ---------------------------------------------------------------------------
# Distillation gate -- SAM path (mIoU present)
# ---------------------------------------------------------------------------


class TestDistillationGateSAMPath:
    r"""Tests for the distillation gate when distilling SAM, which uses mIoU as the primary signal metric."""

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_pass_when_mIoU_meets_threshold(self, stage):
        r"""Distilling SAM: outcome has mIoU, which is the primary signal for the gate."""
        decision = evaluate_gate(
            metrics={"mIoU": 0.6},
            criteria=criteria(min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_retry_when_mIoU_below_threshold(self, stage):
        r"""Distilling SAM: mIoU below threshold with retries left should retry."""
        decision = evaluate_gate(
            metrics={"mIoU": 0.2},
            criteria=criteria(min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "retry"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_escalate_when_budget_exhausted(self, stage):
        r"""Distilling SAM: mIoU below threshold with no retries left should escalate."""
        decision = evaluate_gate(
            metrics={"mIoU": 0.2},
            criteria=criteria(min_mIoU=0.4),
            retry_count=2,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "escalate"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_accepts_val_mIoU_alias(self, stage):
        r"""Distilling SAM: outcome may use 'val_mIoU' instead of 'mIoU'."""
        decision = evaluate_gate(
            metrics={"val_mIoU": 0.5},
            criteria=criteria(min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_reason_includes_mIoU(self, stage):
        r"""Distilling SAM: reason should include mIoU value and threshold for clarity."""
        decision = evaluate_gate(
            metrics={"mIoU": 0.6},
            criteria=criteria(min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert "mIoU" in decision.reason


# ---------------------------------------------------------------------------
# Distillation gate -- GroundingDINO path (mAP50 present, no mIoU)
# ---------------------------------------------------------------------------


class TestDistillationGateDetectionPath:
    r"""Distillation gate when distilling GroundingDINO — uses mAP50 as the signal metric."""

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_pass_when_mAP50_meets_threshold(self, stage):
        r"""Distilling GroundingDINO: outcome has mAP50, no mIoU."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.65},
            criteria=criteria(min_mAP50=0.5, min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_retry_when_mAP50_below_threshold(self, stage):
        r"""Distilling GroundingDINO: mAP50 below threshold with retries left should retry."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.3},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "retry"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_escalate_when_mAP50_budget_exhausted(self, stage):
        r"""Distilling GroundingDINO: mAP50 below threshold with no retries left should escalate."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.3},
            criteria=criteria(min_mAP50=0.5),
            retry_count=2,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "escalate"

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_reason_includes_mAP50(self, stage):
        r"""Distilling GroundingDINO: reason should include mAP50 value and threshold for clarity."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.65},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert "mAP50" in decision.reason

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_mIoU_takes_priority_when_both_present(self, stage):
        r"""If both metrics present, mIoU is used (SAM path takes precedence)."""
        decision = evaluate_gate(
            metrics={"mIoU": 0.6, "mAP50": 0.2},  # mAP50 would fail
            criteria=criteria(min_mAP50=0.5, min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"
        assert "mIoU" in decision.reason

    @pytest.mark.parametrize("stage", DISTILLATION_STAGES)
    def test_escalate_when_neither_metric_present(self, stage):
        R"""No mIoU and no mAP50: escalate, not silent pass."""
        decision = evaluate_gate(
            metrics={"val_loss": 0.5},
            criteria=criteria(),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "escalate"
        assert "neither" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Default gate (other stages)
# ---------------------------------------------------------------------------


class TestDefaultGate:
    r"""Stages without specific metric thresholds — should pass by default, not escalate."""

    @pytest.mark.parametrize("stage", OTHER_STAGES)
    def test_always_passes_unknown_stage(self, stage):
        decision = evaluate_gate(
            metrics={},
            criteria=criteria(),
            retry_count=0,
            max_retries=2,
            stage=stage,
        )
        assert decision.verdict == "pass"
        assert "no metric threshold" in decision.reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    r"""Edge cases around threshold boundaries, missing metrics, and retry counts."""

    def test_mAP50_zero_retries_left_escalates(self):
        r"""mAP50 below threshold with zero retries left should escalate, not retry."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.0},
            criteria=criteria(min_mAP50=0.5),
            retry_count=2,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.verdict == "escalate"

    def test_mAP50_exactly_below_by_epsilon(self):
        r"""mAP50 just below threshold should retry if retries left."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.4999},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.verdict == "retry"

    def test_max_retries_zero_escalates_immediately(self):
        r"""If budget is 0 retries, first failure escalates."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.1},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=0,
            stage="teacher_training",
        )
        assert decision.verdict == "escalate"

    def test_mAP50_takes_priority_over_mIoU_in_detection_stage(self):
        r"""Detection stage should use mAP50, ignore mIoU."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.6, "mIoU": 0.1},
            criteria=criteria(min_mAP50=0.5, min_mIoU=0.4),
            retry_count=0,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.verdict == "pass"

    def test_zero_mAP50_is_gated_not_skipped(self):
        """mAP50=0.0 is falsy -- must still be evaluated, not fall through to next alias key."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.0, "best_metric": 0.9},
            criteria=criteria(min_mAP50=0.5),
            retry_count=0,
            max_retries=2,
            stage="teacher_training",
        )
        # Should use mAP50=0.0, not fall through to best_metric=0.9
        assert decision.verdict == "retry"
        assert "0.000" in decision.reason

    def test_high_retry_count_beyond_max_still_escalates(self):
        r"""Even if retry_count is much higher than max_retries, should still escalate, not error or retry."""
        decision = evaluate_gate(
            metrics={"mAP50": 0.2},
            criteria=criteria(min_mAP50=0.5),
            retry_count=100,
            max_retries=2,
            stage="teacher_training",
        )
        assert decision.verdict == "escalate"
