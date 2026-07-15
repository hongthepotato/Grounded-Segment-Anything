"""
Unit tests for ml_engine.experiment.trial_log.TrialLog and TrialRecord.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_engine.experiment.trial_log import TrialLog, TrialRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUDGET = {"metric_mode": "max", "max_trials": 20}


def make_log(tmp_path, run_id: str = "run-001", baseline: float = None) -> TrialLog:
    return TrialLog(
        run_id=run_id,
        output_dir=str(tmp_path / run_id),
        budget_summary=BUDGET,
        baseline_metric=baseline,
    )


def rec(trial_id: str, metric: float, overrides: dict = None, status: str = "keep") -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        overrides=overrides or {"batch_size": 8},
        primary_metric=metric,
        all_metrics={"val_mAP50": metric},
        status=status,
        description=f"trial {trial_id}",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    r"""Tests for TrialLog constructor behavior."""

    def test_creates_output_dir(self, tmp_path):
        r"""Constructor should create output dir if it doesn't exist."""
        out_dir = tmp_path / "run-x" / "nested"
        TrialLog(run_id="x", output_dir=str(out_dir), budget_summary=BUDGET)
        assert out_dir.exists()

    def test_empty_initially(self, tmp_path):
        r"""New log should have no trials and no best metric."""
        log = make_log(tmp_path)
        assert not log.trials
        assert log.best_metric is None
        assert log.get_best() is None


# ---------------------------------------------------------------------------
# append / best tracking
# ---------------------------------------------------------------------------


class TestAppend:
    r"""Tests for append() method and best metric tracking."""

    def test_append_adds_trial(self, tmp_path):
        r"""Appending a trial should add it to the log's trials list."""
        log = make_log(tmp_path)
        assert len(log.trials) == 0
        log.append(rec("t1", 0.5))
        assert len(log.trials) == 1

    def test_append_writes_json_immediately(self, tmp_path):
        r"""Appending should persist to experiment_log.json right away."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.6))
        path = Path(str(tmp_path / "run-001")) / "experiment_log.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["trials"]) == 1

    def test_best_metric_updated(self, tmp_path):
        r"""Appending a trial with better metric should update best_metric."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.5))
        log.append(rec("t2", 0.7))
        assert log.best_metric == pytest.approx(0.7)

    def test_best_trial_is_highest_metric(self, tmp_path):
        r"""get_best() should return the trial record with the best metric."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.5))
        log.append(rec("t2", 0.8))
        log.append(rec("t3", 0.6))
        best = log.get_best()
        assert best.trial_id == "t2"

    def test_crashed_trial_not_tracked_as_best(self, tmp_path):
        r"""A trial with status "crashed" should not be considered for best metric."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.5))
        log.append(rec("t2", 0.99, status="crashed"))
        assert log.best_metric == pytest.approx(0.5)  # t2 ignored

    def test_oom_trial_not_tracked_as_best(self, tmp_path):
        r"""A trial with status "oom" should not be considered for best metric."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.4))
        log.append(rec("t2", 0.99, status="oom"))
        assert log.best_metric == pytest.approx(0.4)

    def test_skip_trial_not_tracked_as_best(self, tmp_path):
        r"""A trial with status "skip" should not be considered for best metric."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.4))
        log.append(rec("t2", 0.99, status="skip"))
        assert log.best_metric == pytest.approx(0.4)

    def test_none_metric_ignored(self, tmp_path):
        r"""A trial with primary_metric=None should not affect best_metric."""
        log = make_log(tmp_path)
        no_metric = TrialRecord(
            trial_id="t-none",
            overrides={},
            primary_metric=None,
            all_metrics={},
            status="keep",
            description="no metric",
        )
        log.append(no_metric)
        assert log.best_metric is None

    def test_min_mode_tracks_lowest(self, tmp_path):
        r"""If metric_mode is "min", best_metric should track lowest value."""
        log = TrialLog(
            run_id="min-run",
            output_dir=str(tmp_path / "min-run"),
            budget_summary={"metric_mode": "min"},
        )
        log.append(rec("t1", 0.8))
        log.append(rec("t2", 0.3))
        log.append(rec("t3", 0.5))
        assert log.best_metric == pytest.approx(0.3)
        assert log.get_best().trial_id == "t2"


# ---------------------------------------------------------------------------
# trials property
# ---------------------------------------------------------------------------


class TestTrialsProperty:
    r"""Tests for the trials property, which should return a copy of the trials list."""

    def test_returns_copy(self, tmp_path):
        r"""Accessing the trials property should return a copy, not the original list."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.5))
        trials = log.trials
        trials.clear()
        assert len(log.trials) == 1  # original unaffected


# ---------------------------------------------------------------------------
# to_llm_context
# ---------------------------------------------------------------------------


class TestToLlmContext:
    r"""Tests for to_llm_context() method, which formats log info for LLM input."""

    def test_includes_run_id(self, tmp_path):
        r"""to_llm_context output should include the run_id."""
        log = make_log(tmp_path, run_id="my-run")
        text = log.to_llm_context()
        assert "my-run" in text

    def test_includes_baseline(self, tmp_path):
        r"""to_llm_context output should include the baseline metric if set."""
        log = make_log(tmp_path, baseline=0.45)
        text = log.to_llm_context()
        assert "0.45" in text

    def test_includes_trial_history(self, tmp_path):
        r"""to_llm_context output should include a summary of all trials."""
        log = make_log(tmp_path)
        log.append(rec("trial-001", 0.6, overrides={"batch_size": 16}))
        text = log.to_llm_context()
        assert "trial-001" in text
        assert "batch_size" in text

    def test_none_metric_shown_as_na(self, tmp_path):
        r"""A trial with primary_metric=None should show "N/A" in the LLM context."""
        log = make_log(tmp_path)
        no_metric = TrialRecord(
            trial_id="t-none",
            overrides={},
            primary_metric=None,
            all_metrics={},
            status="crashed",
            description="crashed",
        )
        log.append(no_metric)
        text = log.to_llm_context()
        assert "N/A" in text

    def test_empty_log_still_renders(self, tmp_path):
        r"""to_llm_context should return a string even if no trials have been appended."""
        log = make_log(tmp_path)
        text = log.to_llm_context()
        assert isinstance(text, str)
        assert "run-001" in text


# ---------------------------------------------------------------------------
# to_feedback_record
# ---------------------------------------------------------------------------


class TestToFeedbackRecord:
    r"""Tests for to_feedback_record() method, which exports log info for MemoryStore."""

    def test_contains_required_fields(self, tmp_path):
        r"""to_feedback_record output should contain run_id, best_metric, trial_count, and best_overrides."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.7))
        fb = log.to_feedback_record()
        assert "run_id" in fb
        assert "best_metric" in fb
        assert "baseline_metric" in fb
        assert "trial_count" in fb
        assert "best_overrides" in fb

    def test_best_overrides_match_best_trial(self, tmp_path):
        r"""best_overrides in feedback record should match the overrides of the best trial."""
        log = make_log(tmp_path)
        log.append(rec("t1", 0.5, overrides={"batch_size": 8}))
        log.append(rec("t2", 0.8, overrides={"batch_size": 16}))
        fb = log.to_feedback_record()
        assert fb["best_overrides"]["batch_size"] == 16

    def test_empty_log_best_overrides_empty(self, tmp_path):
        r"""If no trials, best_overrides in feedback record should be an empty dict."""
        log = make_log(tmp_path)
        fb = log.to_feedback_record()
        assert not fb["best_overrides"]


# ---------------------------------------------------------------------------
# Persistence: _flush / load
# ---------------------------------------------------------------------------


class TestPersistenceLoad:
    r"""Tests for the load() class method, which should recover log state from disk."""

    def test_load_recovers_all_trials(self, tmp_path):
        r"""A log loaded from disk should have the same trials as were appended before saving."""
        out_dir = str(tmp_path / "load-test")
        log = TrialLog(run_id="r", output_dir=out_dir, budget_summary=BUDGET)
        for i in range(3):
            log.append(rec(f"t{i}", 0.5 + i * 0.1))

        loaded = TrialLog.load(out_dir)
        assert len(loaded.trials) == 3

    def test_load_recovers_best_metric(self, tmp_path):
        r"""A log loaded from disk should have the same best_metric as before saving."""
        out_dir = str(tmp_path / "load-best")
        log = TrialLog(run_id="r", output_dir=out_dir, budget_summary=BUDGET)
        log.append(rec("t1", 0.5))
        log.append(rec("t2", 0.9))

        loaded = TrialLog.load(out_dir)
        assert loaded.best_metric == pytest.approx(0.9)

    def test_load_recovers_best_trial_id(self, tmp_path):
        r"""A log loaded from disk should have the same best_trial_id as before saving."""
        out_dir = str(tmp_path / "load-id")
        log = TrialLog(run_id="r", output_dir=out_dir, budget_summary=BUDGET)
        log.append(rec("best-trial", 0.9))

        loaded = TrialLog.load(out_dir)
        assert loaded._best_trial_id == "best-trial"

    def test_atomic_write_no_partial_file(self, tmp_path):
        r"""_flush writes to .tmp then replaces -- no partial JSON."""
        out_dir = str(tmp_path / "atomic")
        log = TrialLog(run_id="r", output_dir=out_dir, budget_summary=BUDGET)
        for i in range(5):
            log.append(rec(f"t{i}", 0.5))
        path = Path(out_dir) / "experiment_log.json"
        data = json.loads(path.read_text())
        assert len(data["trials"]) == 5  # no partial writes
