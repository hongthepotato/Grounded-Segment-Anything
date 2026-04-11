"""
Unit tests for ml_engine.jobs.models.

Tests Job, JobProgress, JobOutcome, WorkerInfo serialization and properties.
Pure Python -- no Redis.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ml_engine.jobs.models import (
    Job,
    JobOutcome,
    JobProgress,
    JobStatus,
    JobType,
    WorkerInfo,
)


# ---------------------------------------------------------------------------
# JobProgress
# ---------------------------------------------------------------------------

class TestJobProgress:
    r"""Tests for JobProgress properties and serialization."""
    def test_defaults(self):
        r"""Test that default JobProgress has zero epochs and empty metrics"""
        p = JobProgress()
        assert p.current_epoch == 0
        assert p.total_epochs == 0
        assert p.metrics == {}

    def test_epoch_progress_zero_when_no_steps(self):
        r"""Test that epoch_progress is zero when total_steps is zero."""
        p = JobProgress(current_step=0, total_steps=0)
        assert p.epoch_progress == 0.0

    def test_epoch_progress_calculated(self):
        r"""Test that epoch_progress is current_step / total_steps."""
        p = JobProgress(current_step=25, total_steps=100)
        assert p.epoch_progress == pytest.approx(0.25)

    def test_overall_progress_zero_when_no_epochs(self):
        r"""Test that overall_progress is zero when total_epochs is zero."""
        p = JobProgress(total_epochs=0)
        assert p.overall_progress == 0.0

    def test_overall_progress_complete(self):
        r"""Test that overall_progress is 1.0 when all epochs are completed."""
        p = JobProgress(current_epoch=10, total_epochs=10, current_step=0, total_steps=100)
        # 10 completed epochs out of 10 = 1.0, plus 0 progress in current epoch
        assert p.overall_progress == pytest.approx(1.0)

    def test_overall_progress_half(self):
        r"""Test that overall_progress is 0.5 when half the epochs are completed."""
        p = JobProgress(current_epoch=5, total_epochs=10, current_step=0, total_steps=100)
        assert p.overall_progress == pytest.approx(0.5)

    def test_to_dict_roundtrip(self):
        r"""Test that to_dict and from_dict preserve all fields."""
        p = JobProgress(current_epoch=3, total_epochs=10, metrics={"loss": 0.5})
        d = p.to_dict()
        p2 = JobProgress.from_dict(d)
        assert p2.current_epoch == 3
        assert p2.total_epochs == 10
        assert p2.metrics["loss"] == pytest.approx(0.5)

    def test_from_dict_empty_returns_default(self):
        r"""Test that from_dict with empty dict returns default JobProgress."""
        p = JobProgress.from_dict({})
        assert p.current_epoch == 0
        assert p.total_epochs == 0
        assert p.metrics == {}

    def test_from_dict_none_returns_default(self):
        r"""Test that from_dict with None returns default JobProgress."""
        p = JobProgress.from_dict(None)
        assert p.current_epoch == 0
        assert p.total_epochs == 0
        assert p.metrics == {}



# ---------------------------------------------------------------------------
# JobOutcome
# ---------------------------------------------------------------------------

class TestJobOutcome:
    r"""Tests for JobOutcome properties and serialization."""
    def test_defaults(self):
        r"""Test that default JobOutcome has status 'completed' and empty metrics."""
        o = JobOutcome()
        assert o.status == "completed"
        assert o.metrics == {}
        assert o.error_message is None

    def test_to_dict_roundtrip(self):
        r"""Test that to_dict and from_dict preserve all fields."""
        o = JobOutcome(
            status="completed",
            metrics={"mAP50": 0.72},
            artifacts={"checkpoint": "model.pt"},
            wall_time_seconds=3600.0,
        )
        d = o.to_dict()
        o2 = JobOutcome.from_dict(d)
        assert o2.metrics["mAP50"] == pytest.approx(0.72)
        assert o2.artifacts == {"checkpoint": "model.pt"}
        assert o2.wall_time_seconds == pytest.approx(3600.0)

    def test_extra_serialized_inline_and_roundtripped(self):
        r"""Extra fields are merged into the top-level dict and restored on from_dict."""
        o = JobOutcome(extra={"experiment_result": {"best_trial_id": "t42", "best_metric": 0.9}})
        d = o.to_dict()
        assert d["experiment_result"]["best_trial_id"] == "t42"
        o2 = JobOutcome.from_dict(d)
        assert o2.extra["experiment_result"]["best_metric"] == pytest.approx(0.9)

    def test_from_dict_empty(self):
        r"""Test that from_dict with empty dict returns default JobOutcome."""
        o = JobOutcome.from_dict({})
        assert o.status == "completed"

    def test_from_dict_none(self):
        r"""Test that from_dict with None returns default JobOutcome."""
        o = JobOutcome.from_dict(None)
        assert o.status == "completed"

    def test_error_message_preserved(self):
        r"""Test that error_message is preserved through to_dict and from_dict."""
        o = JobOutcome(status="failed", error_message="OOM on epoch 3")
        d = o.to_dict()
        o2 = JobOutcome.from_dict(d)
        assert o2.error_message == "OOM on epoch 3"

    def test_wall_time_coerced_to_float(self):
        r"""Redis returns strings -- from_dict must coerce."""
        o = JobOutcome.from_dict({"wall_time_seconds": "3600"})
        assert isinstance(o.wall_time_seconds, float)


# ---------------------------------------------------------------------------
# Job -- construction
# ---------------------------------------------------------------------------

class TestJobConstruction:
    r"""Tests for Job construction and properties."""
    def test_defaults(self):
        r"""Test that default Job has PENDING status, TEACHER_TRAINING type, empty config, and run_id defaults to id."""
        j = Job()
        assert j.status == JobStatus.PENDING
        assert j.type == JobType.TEACHER_TRAINING.value
        assert j.config == {}
        assert j.run_id == j.id  # standalone default

    def test_run_id_defaults_to_id(self):
        r"""Test that if run_id is not provided, it defaults to the Job's id."""
        j = Job()
        assert j.run_id == j.id

    def test_explicit_run_id(self):
        r"""Test that if run_id is provided, it is set correctly."""
        j = Job(run_id="pipeline-001")
        assert j.run_id == "pipeline-001"

    def test_created_at_set_automatically(self):
        r"""Test that created_at is set to a datetime when Job is constructed."""
        j = Job()
        assert j.created_at is not None

    def test_status_string_coerced_to_enum(self):
        r"""Test that status string is coerced to enum."""
        j = Job(status="running")
        assert j.status == JobStatus.RUNNING

    def test_is_terminal_completed(self):
        r"""Test that completed jobs are terminal."""
        j = Job(status=JobStatus.COMPLETED)
        assert j.is_terminal is True

    def test_is_terminal_failed(self):
        r"""Test that failed jobs are terminal."""
        j = Job(status=JobStatus.FAILED)
        assert j.is_terminal is True

    def test_is_terminal_cancelled(self):
        r"""Test that cancelled jobs are terminal."""
        j = Job(status=JobStatus.CANCELLED)
        assert j.is_terminal is True

    def test_not_terminal_when_pending(self):
        r"""Test that pending jobs are not terminal."""
        j = Job(status=JobStatus.PENDING)
        assert j.is_terminal is False

    def test_not_terminal_when_running(self):
        r"""Test that running jobs are not terminal."""
        j = Job(status=JobStatus.RUNNING)
        assert j.is_terminal is False

    def test_duration_none_when_not_started(self):
        r"""Test that duration is None when job has not started."""
        j = Job()
        assert j.duration_seconds is None

    def test_duration_calculated_when_started_and_finished(self):
        r"""Test that duration is calculated as finished_at - started_at."""
        now = datetime.now()
        j = Job(
            started_at=now - timedelta(seconds=300),
            finished_at=now,
        )
        assert j.duration_seconds == pytest.approx(300.0, abs=1.0)


# ---------------------------------------------------------------------------
# Job -- to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------

class TestJobSerialization:
    r"""Tests for Job to_dict and from_dict serialization."""
    def test_to_dict_roundtrip(self):
        r"""Test that to_dict and from_dict preserve all fields."""
        j = Job(
            type=JobType.EXPERIMENT_LOOP.value,
            config={"batch_size": 8, "lr": 0.001},
            run_id="pipeline-001",
            tags=["prod", "v2"],
        )
        d = j.to_dict()
        j2 = Job.from_dict(d)
        assert j2.id == j.id
        assert j2.run_id == "pipeline-001"
        assert j2.config["batch_size"] == 8
        assert j2.config["lr"] == 0.001
        assert j2.tags == ["prod", "v2"]

    def test_status_roundtrip(self):
        r"""Test that JobStatus is preserved through to_dict and from_dict."""
        j = Job(status=JobStatus.RUNNING)
        j2 = Job.from_dict(j.to_dict())
        assert j2.status == JobStatus.RUNNING

    def test_from_dict_bytes_keys(self):
        r"""Simulate Redis returning bytes."""
        j = Job(run_id="r1", config={"x": 1})
        raw = {k.encode(): v.encode() if isinstance(v, str) else str(v).encode()
               for k, v in j.to_dict().items()}
        j2 = Job.from_dict(raw)
        assert j2.run_id == "r1"

    def test_outcome_roundtrip(self):
        r"""Test that JobOutcome is preserved through to_dict and from_dict."""
        j = Job(outcome=JobOutcome(status="completed", metrics={"mAP50": 0.8}))
        j2 = Job.from_dict(j.to_dict())
        assert j2.outcome is not None
        assert j2.outcome.metrics["mAP50"] == pytest.approx(0.8)

    def test_progress_roundtrip(self):
        r"""Test that JobProgress is preserved through to_dict and from_dict."""
        j = Job(progress=JobProgress(current_epoch=3, total_epochs=10))
        j2 = Job.from_dict(j.to_dict())
        assert j2.progress.current_epoch == 3

    def test_priority_roundtrip(self):
        r"""Test that priority is preserved through to_dict and from_dict."""
        j = Job(priority=5)
        j2 = Job.from_dict(j.to_dict())
        assert j2.priority == 5

    def test_priority_invalid_string_defaults_zero(self):
        r"""Test that invalid priority string defaults to zero."""
        d = Job().to_dict()
        d["priority"] = "not_an_int"
        j = Job.from_dict(d)
        assert j.priority == 0

    def test_created_at_roundtrip(self):
        r"""Test that created_at is preserved through to_dict and from_dict."""
        j = Job()
        j2 = Job.from_dict(j.to_dict())
        assert j2.created_at is not None

    def test_empty_error_message_is_none(self):
        r"""Test that empty string error_message is converted to None."""
        j = Job()
        j2 = Job.from_dict(j.to_dict())
        assert j2.error_message is None

    def test_repr_shows_id_prefix_and_status(self):
        r"""Test that Job __repr__ includes id prefix and status."""
        j = Job(status=JobStatus.RUNNING)
        r = repr(j)
        assert "running" in r
        assert j.id[:8] in r


# ---------------------------------------------------------------------------
# WorkerInfo
# ---------------------------------------------------------------------------

class TestWorkerInfo:
    r""""""
    def test_defaults(self):
        w = WorkerInfo(id="worker-1")
        assert w.gpu_id == 0
        assert w.status == "idle"
        assert w.current_job_id is None

    def test_to_dict_roundtrip(self):
        w = WorkerInfo(id="worker-1", gpu_id=2, hostname="node-01", status="busy")
        d = w.to_dict()
        w2 = WorkerInfo.from_dict(d)
        assert w2.id == "worker-1"
        assert w2.gpu_id == 2
        assert w2.hostname == "node-01"
        assert w2.status == "busy"

    def test_from_dict_bytes(self):
        w = WorkerInfo(id="w", gpu_id=1)
        raw = {k.encode(): v.encode() if isinstance(v, str) else str(v).encode()
               for k, v in w.to_dict().items()}
        w2 = WorkerInfo.from_dict(raw)
        assert w2.gpu_id == 1

    def test_current_job_id_empty_string_is_none(self):
        w = WorkerInfo(id="w", current_job_id=None)
        d = w.to_dict()
        w2 = WorkerInfo.from_dict(d)
        assert w2.current_job_id is None

    def test_last_heartbeat_set_automatically(self):
        w = WorkerInfo(id="w")
        assert w.last_heartbeat is not None
