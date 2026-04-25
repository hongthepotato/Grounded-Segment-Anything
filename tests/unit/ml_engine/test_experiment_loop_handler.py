"""
Unit tests for ml_engine.jobs.handlers.experiment_loop.ExperimentLoopHandler.

Tests are structured around the handler's two responsibilities:
  1. Config assembly: budget, guard, propose_fn wired from YAML + job_config
  2. outcome.json output: shape, keys, cancellation status, missing best_config

The handler uses late imports inside run() (subprocess isolation pattern), so
all heavy deps are patched at their source modules rather than at the handler
module level.

Patch targets:
  core.config.load_config                    <- YAML loading
  core.constants.transform_image_path       <- path transformation
  ml_engine.data.manager.DataManager        <- data loading
  ml_engine.experiment.ExperimentLoop       <- HPO loop (re-exported from __init__)
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from ml_engine.jobs.handlers.experiment_loop import ExperimentLoopHandler

# ---------------------------------------------------------------------------
# Minimal YAML defaults returned by load_config mock
# ---------------------------------------------------------------------------

_MINIMAL_DEFAULTS: Dict[str, Any] = {
    "experiment": {
        "max_trials": 10,
        "epochs_per_trial": 3,
        "max_wall_time_seconds": None,
        "metric_name": "val_mAP50",
        "metric_mode": "max",
        "use_llm_propose": False,
    },
    "mutable_keys": {
        "epochs": {"type": "int", "min": 1, "max": 50},
        "batch_size": {"type": "int", "min": 1, "max": 16},
    },
    "immutable_keys": ["num_classes", "class_names"],
}

# ml_engine.data imports pycocotools (DLL unavailable in CI/test env).
# Stub the package in sys.modules before patch() tries to import it.
_DATA_STUBS = {
    "ml_engine.data": MagicMock(),
    "ml_engine.data.manager": MagicMock(),
}


def _fake_experiment_result(
    best_metric: float = 0.85,
    trials_completed: int = 10,
    cancelled: bool = False,
    best_trial_id: str = "trial_001",
) -> MagicMock:
    """Build a minimal ExperimentResult-like mock."""
    r = MagicMock()
    r.best_metric = best_metric
    r.trials_completed = trials_completed
    r.cancelled = cancelled
    r.best_trial_id = best_trial_id
    r.run_id = "exp_abc123"
    r.wall_time_seconds = 120.0
    return r


def _run_handler(
    job_config: Dict[str, Any],
    output_dir: str,
    experiment_result=None,
    write_best_config: bool = True,
) -> None:
    """
    Run ExperimentLoopHandler.run() with all heavy deps mocked.

    Optionally creates best_config.yaml first (simulating a trial that improved).
    """
    if experiment_result is None:
        experiment_result = _fake_experiment_result()

    ctx = mp.get_context("spawn")

    with patch.dict("sys.modules", _DATA_STUBS):
        with (
            patch("core.config.load_config", return_value=_MINIMAL_DEFAULTS),
            patch("core.constants.transform_image_path", side_effect=lambda x: x),
            patch("ml_engine.data.manager.DataManager") as MockDM,
            patch("ml_engine.experiment.ExperimentLoop") as MockLoop,
        ):
            MockDM.from_file.return_value = MagicMock()
            mock_loop_instance = MagicMock()
            mock_loop_instance.run.return_value = experiment_result
            MockLoop.return_value = mock_loop_instance

            if write_best_config:
                (Path(output_dir) / "best_config.yaml").write_text("epochs: 5\n", encoding="utf-8")

            handler = ExperimentLoopHandler()
            handler.run(
                job_config=job_config,
                output_dir=output_dir,
                progress_queue=ctx.Queue(),
                cancel_event=ctx.Event(),
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job_config():
    return {
        "data_path": "/data/annotations.json",
        "image_paths": ["img1.jpg", "img2.jpg"],
    }


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# outcome.json shape
# ---------------------------------------------------------------------------


class TestOutcomeJson:
    def test_outcome_written(self, job_config, output_dir):
        _run_handler(job_config, output_dir)
        assert (Path(output_dir) / "outcome.json").exists()

    def test_outcome_status_completed(self, job_config, output_dir):
        _run_handler(job_config, output_dir, _fake_experiment_result(cancelled=False))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert outcome["status"] == "completed"

    def test_outcome_status_cancelled(self, job_config, output_dir):
        _run_handler(job_config, output_dir, _fake_experiment_result(cancelled=True))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert outcome["status"] == "cancelled"

    def test_metrics_has_best_metric(self, job_config, output_dir):
        _run_handler(job_config, output_dir, _fake_experiment_result(best_metric=0.73))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert outcome["metrics"]["best_metric"] == pytest.approx(0.73)

    def test_metrics_has_trials_completed(self, job_config, output_dir):
        _run_handler(job_config, output_dir, _fake_experiment_result(trials_completed=7))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert outcome["metrics"]["trials_completed"] == 7.0

    def test_wall_time_seconds_present(self, job_config, output_dir):
        _run_handler(job_config, output_dir)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "wall_time_seconds" in outcome
        assert isinstance(outcome["wall_time_seconds"], float)

    def test_experiment_result_at_top_level(self, job_config, output_dir):
        """JobOutcome.extra spreads via d.update(self.extra): experiment_result is top-level."""
        _run_handler(job_config, output_dir, _fake_experiment_result(best_trial_id="t42"))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "experiment_result" in outcome
        assert outcome["experiment_result"]["best_trial_id"] == "t42"

    def test_experiment_result_has_run_id(self, job_config, output_dir):
        _run_handler(job_config, output_dir)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert outcome["experiment_result"]["run_id"] == "exp_abc123"

    def test_best_metric_key_matches_gate_lookup(self, job_config, output_dir):
        """gate.py checks ("mAP50", "best_metric", "val_mAP50") -- best_metric must be present."""
        _run_handler(job_config, output_dir, _fake_experiment_result(best_metric=0.9))
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        metrics = outcome["metrics"]
        assert any(k in metrics for k in ("mAP50", "best_metric", "val_mAP50")), (
            "gate.py requires at least one of mAP50/best_metric/val_mAP50"
        )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_best_config_in_artifacts_when_file_exists(self, job_config, output_dir):
        _run_handler(job_config, output_dir, write_best_config=True)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "best_config" in outcome["artifacts"]

    def test_best_config_absent_from_artifacts_when_all_crashed(self, job_config, output_dir):
        """All-crash run: no trial improved, best_config.yaml never written."""
        _run_handler(job_config, output_dir, write_best_config=False)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "best_config" not in outcome["artifacts"]

    def test_experiment_log_always_in_artifacts(self, job_config, output_dir):
        _run_handler(job_config, output_dir)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "experiment_log" in outcome["artifacts"]

    def test_feedback_always_in_artifacts(self, job_config, output_dir):
        _run_handler(job_config, output_dir)
        outcome = json.loads((Path(output_dir) / "outcome.json").read_text())
        assert "feedback" in outcome["artifacts"]


# ---------------------------------------------------------------------------
# Budget construction: YAML defaults + job_config overrides
# ---------------------------------------------------------------------------


class TestBudgetConstruction:
    def _run_with_capture(self, job_config: Dict[str, Any], output_dir: str) -> Any:
        """Run handler and capture the budget passed to ExperimentLoop.run()."""
        captured: Dict[str, Any] = {}

        def capture(**kw):
            captured["budget"] = kw["budget"]
            return _fake_experiment_result()

        ctx = mp.get_context("spawn")
        with patch.dict("sys.modules", _DATA_STUBS):
            with (
                patch("core.config.load_config", return_value=_MINIMAL_DEFAULTS),
                patch("core.constants.transform_image_path", side_effect=lambda x: x),
                patch("ml_engine.data.manager.DataManager"),
                patch("ml_engine.experiment.ExperimentLoop") as MockLoop,
            ):
                instance = MagicMock()
                instance.run.side_effect = capture
                MockLoop.return_value = instance
                ExperimentLoopHandler().run(
                    job_config=job_config,
                    output_dir=output_dir,
                    progress_queue=ctx.Queue(),
                    cancel_event=ctx.Event(),
                )

        return captured.get("budget")

    def test_default_max_trials_from_yaml(self, output_dir):
        budget = self._run_with_capture({"data_path": "/d", "image_paths": ["x.jpg"]}, output_dir)
        assert budget.max_trials == 10

    def test_job_config_overrides_max_trials(self, output_dir):
        budget = self._run_with_capture(
            {"data_path": "/d", "image_paths": ["x.jpg"], "experiment": {"max_trials": 5}},
            output_dir,
        )
        assert budget.max_trials == 5

    def test_wall_time_int_cast(self, output_dir):
        """max_wall_time_seconds cast to int even if provided as string."""
        budget = self._run_with_capture(
            {"data_path": "/d", "image_paths": ["x.jpg"], "experiment": {"max_wall_time_seconds": "3600"}},
            output_dir,
        )
        assert budget.max_wall_time_seconds == 3600
        assert isinstance(budget.max_wall_time_seconds, int)

    def test_null_wall_time_stays_none(self, output_dir):
        budget = self._run_with_capture({"data_path": "/d", "image_paths": ["x.jpg"]}, output_dir)
        assert budget.max_wall_time_seconds is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def _run_expect_raise(self, job_config: Dict[str, Any], output_dir: str, match: str) -> None:
        ctx = mp.get_context("spawn")
        with patch.dict("sys.modules", _DATA_STUBS):
            with (
                patch("core.config.load_config", return_value=_MINIMAL_DEFAULTS),
                patch("core.constants.transform_image_path", side_effect=lambda x: x),
                patch("ml_engine.data.manager.DataManager"),
                patch("ml_engine.experiment.ExperimentLoop"),
            ):
                with pytest.raises(ValueError, match=match):
                    ExperimentLoopHandler().run(
                        job_config=job_config,
                        output_dir=output_dir,
                        progress_queue=ctx.Queue(),
                        cancel_event=ctx.Event(),
                    )

    def test_missing_data_path_raises(self, output_dir):
        self._run_expect_raise({"image_paths": ["x.jpg"]}, output_dir, "data_path required")

    def test_missing_image_paths_raises(self, output_dir):
        self._run_expect_raise(
            {"data_path": "/data/x.json", "image_paths": []}, output_dir, "image_paths required"
        )
