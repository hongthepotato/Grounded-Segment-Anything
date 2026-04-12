"""
Unit tests for ml_engine.agent.contracts.

Tests PipelineContract, StageSummary, GateDecision, and all nested dataclasses
-- especially to_dict/from_dict roundtrips and edge cases in from_dict.
"""

from __future__ import annotations

import pytest

from ml_engine.agent.contracts import (
    AcceptanceCriteria,
    BudgetSpec,
    DataSpec,
    GateDecision,
    LineageSpec,
    PipelineContract,
    StageSummary,
    TargetSpec,
)


# ---------------------------------------------------------------------------
# TargetSpec
# ---------------------------------------------------------------------------

class TestTargetSpec:
    r"""TargetSpec is the contract section that describes what the pipeline is trying to achieve,
    in terms of model outputs and evaluation criteria.
    It informs the entire pipeline design, from data processing
    to model architecture to evaluation metrics."""
    def test_defaults(self):
        r"""Test that default TargetSpec has empty class_names and "detection" output_mode."""
        t = TargetSpec(class_names=["defect"])
        assert t.output_mode == "detection"
        assert t.description == ""

    def test_segmentation_mode(self):
        r"""Test that we can set output_mode to "segmentation" and it is preserved."""
        t = TargetSpec(class_names=["crack"], output_mode="segmentation")
        assert t.output_mode == "segmentation"
        assert t.class_names == ["crack"]
        assert t.description == ""


# ---------------------------------------------------------------------------
# DataSpec
# ---------------------------------------------------------------------------

class TestDataSpec:
    r"""DataSpec describes where the data lives and how to split it.
    It is used by the Coordinator to plan the pipeline and by the TeacherTrainingHandler
    to load and split the dataset. The default split is 70% train, 15% val, 15% test,
    but it can be customized by providing a split_config dict.
    The DataSpec also includes a list of image paths, which can be used for more
    fine-grained control over the dataset."""
    def test_default_split(self):
        r"""Test that the default split_config is 70% train, 15% val, 15% test."""
        d = DataSpec(data_path="/data", image_paths=[])
        assert d.split_config["train"] == pytest.approx(0.7)
        assert d.split_config["val"] == pytest.approx(0.15)
        assert d.split_config["test"] == pytest.approx(0.15)

    def test_custom_split(self):
        r"""Test that a custom split_config is preserved and used."""
        d = DataSpec(data_path="/d", image_paths=[], split_config={"train": 0.8, "val": 0.1, "test": 0.1})
        assert d.split_config["train"] == pytest.approx(0.8)
        assert d.split_config["val"] == pytest.approx(0.1)
        assert d.split_config["test"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# BudgetSpec
# ---------------------------------------------------------------------------

class TestBudgetSpec:
    r"""BudgetSpec defines limits on the pipeline execution to guide planning and prevent runaway jobs."""
    def test_defaults(self):
        r"""Test that the default BudgetSpec has max_epochs=50, max_trials=20, max_retries=2, and no wall time limit."""
        b = BudgetSpec()
        assert b.max_epochs == 50
        assert b.max_trials == 20
        assert b.max_retries == 2
        assert b.max_wall_time_seconds is None


# ---------------------------------------------------------------------------
# AcceptanceCriteria
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    r"""AcceptanceCriteria defines the per-stage metric thresholds that the Evaluator uses for pass/fail decisions."""
    def test_defaults(self):
        r"""Test that the default AcceptanceCriteria has min_mAP50=0.5, min_mIoU=0.4, and no max_val_loss."""
        c = AcceptanceCriteria()
        assert c.min_mAP50 == pytest.approx(0.5)
        assert c.min_mIoU == pytest.approx(0.4)
        assert c.max_val_loss is None

    def test_custom_thresholds(self):
        r"""Test that custom thresholds are preserved and used."""
        c = AcceptanceCriteria(min_mAP50=0.7, min_mIoU=0.6)
        assert c.min_mAP50 == pytest.approx(0.7)
        assert c.min_mIoU == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# LineageSpec
# ---------------------------------------------------------------------------

class TestLineageSpec:
    r"""LineageSpec contains metadata for traceability, versioning, and reproducibility."""
    def test_version_hash_generated(self):
        r"""Test that a version_hash is automatically generated and is an 8-character string."""
        l = LineageSpec()
        assert isinstance(l.version_hash, str)
        assert len(l.version_hash) == 8

    def test_created_at_set(self):
        r"""Test that created_at is automatically set to a non-empty string."""
        l = LineageSpec()
        assert l.created_at is not None

    def test_parent_contract_id_defaults_none(self):
        r"""Test that parent_contract_id defaults to None if not provided."""
        l = LineageSpec()
        assert l.parent_contract_id is None


# ---------------------------------------------------------------------------
# PipelineContract.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestPipelineContractSerialization:
    r"""Test the to_dict and from_dict methods of PipelineContract, including nested dataclasses and edge cases."""
    def _make_contract(self) -> PipelineContract:
        return PipelineContract(
            id="contract-001",
            target=TargetSpec(class_names=["crack", "peel"], output_mode="detection"),
            data=DataSpec(data_path="/srv/data", image_paths=["img1.jpg", "img2.jpg"]),
            acceptance_criteria=AcceptanceCriteria(min_mAP50=0.65),
            budget=BudgetSpec(max_epochs=30, max_retries=3),
            lineage=LineageSpec(parent_contract_id="parent-000"),
        )

    def test_roundtrip_id(self):
        r"""Test that the id field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.id == "contract-001"

    def test_roundtrip_class_names(self):
        r"""Test that the target.class_names field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.target.class_names == ["crack", "peel"]

    def test_roundtrip_output_mode(self):
        r"""Test that the target.output_mode field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.target.output_mode == "detection"

    def test_roundtrip_data_path(self):
        r"""Test that the data.data_path field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.data.data_path == "/srv/data"

    def test_roundtrip_image_paths(self):
        r"""Test that the data.image_paths field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.data.image_paths == ["img1.jpg", "img2.jpg"]

    def test_roundtrip_acceptance_criteria(self):
        r"""Test that the acceptance_criteria fields are preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.acceptance_criteria.min_mAP50 == pytest.approx(0.65)

    def test_roundtrip_budget(self):
        r"""Test that the budget fields are preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.budget.max_epochs == 30
        assert c2.budget.max_retries == 3

    def test_roundtrip_lineage_parent(self):
        r"""Test that the lineage.parent_contract_id field is preserved through to_dict and from_dict."""
        c = self._make_contract()
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.lineage.parent_contract_id == "parent-000"

    def test_from_dict_missing_id_generates_one(self):
        r"""Test that if the id field is missing from the input dict, from_dict generates a new one."""
        d = {"target": {"class_names": ["x"]}}
        c = PipelineContract.from_dict(d)
        assert isinstance(c.id, str)
        assert len(c.id) > 0

    def test_from_dict_empty_dict_has_defaults(self):
        r"""Test that if from_dict is given an empty dict, it creates a PipelineContract with all default values."""
        c = PipelineContract.from_dict({})
        assert c.target.class_names == []
        assert c.budget.max_retries == 2

    def test_to_dict_returns_dict(self):
        r"""Test that to_dict returns a dictionary."""
        c = self._make_contract()
        d = c.to_dict()
        assert isinstance(d, dict)

    def test_stage_configs_roundtrip(self):
        r"""Test that the stage_configs field is preserved through to_dict and from_dict, even with nested dicts."""
        c = PipelineContract(
            id="x",
            target=TargetSpec(class_names=[]),
            data=DataSpec(data_path="/d", image_paths=[]),
            acceptance_criteria=AcceptanceCriteria(),
            budget=BudgetSpec(),
            stage_configs={"teacher_training": {"lora_r": 32}},
            lineage=LineageSpec(),
        )
        c2 = PipelineContract.from_dict(c.to_dict())
        assert c2.stage_configs["teacher_training"]["lora_r"] == 32


# ---------------------------------------------------------------------------
# GateDecision
# ---------------------------------------------------------------------------

class TestGateDecision:
    r"""GateDecision represents the result of an Evaluator gate check,
    including the verdict (pass/retry/fail), the reason for the decision,
    and any relevant metrics. It is used by the Evaluator to
    communicate the outcome of the gate check."""
    def test_construction(self):
        r"""Test that we can construct a GateDecision with all fields and they are set correctly."""
        d = GateDecision(verdict="pass", reason="mAP50 met", metrics={"mAP50": 0.7}, retry_count=0)
        assert d.verdict == "pass"
        assert d.metrics["mAP50"] == pytest.approx(0.7)
        assert d.retry_count == 0
        assert d.reason == "mAP50 met"

    def test_defaults(self):
        r"""Test that if we construct a GateDecision with only verdict and reason, the metrics and retry_count default to empty dict and 0."""
        d = GateDecision(verdict="retry", reason="low mAP50")
        assert d.metrics == {}
        assert d.retry_count == 0


# ---------------------------------------------------------------------------
# StageSummary.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestStageSummary:
    def test_to_dict_roundtrip(self):
        s = StageSummary(
            stage="teacher_training",
            status="pass",
            metrics={"mAP50": 0.72},
            artifacts={"checkpoint": "ckpt.pt"},
            key_decisions=["LoRA r=32"],
            duration_seconds=3600.0,
            trial_count=15,
        )
        d = s.to_dict()
        s2 = StageSummary.from_dict(d)
        assert s2.stage == "teacher_training"
        assert s2.status == "pass"
        assert s2.metrics["mAP50"] == pytest.approx(0.72)
        assert s2.trial_count == 15
        assert "checkpoint" in s2.artifacts

    def test_from_dict_ignores_unknown_fields(self):
        d = {
            "stage": "auto_labeling",
            "status": "pass",
            "unknown_future_field": "ignored",
        }
        s = StageSummary.from_dict(d)
        assert s.stage == "auto_labeling"

    def test_defaults(self):
        s = StageSummary(stage="auto_labeling", status="pass")
        assert s.metrics == {}
        assert s.artifacts == {}
        assert s.key_decisions == []
        assert s.duration_seconds == pytest.approx(0.0)
        assert s.trial_count is None
