"""Unit tests for ml_engine.jobs.manager.JobManager.

JobManager is a thin facade over RedisJobStore; the store is mocked so the tests
target the manager's OWN decisions — job-type validation, the cancel state
machine, the delete guard, and status-filter handling. Bug-hunting, not padding:
several assertions would fail if a transition were wrong (e.g. cancelling a
RUNNING job must set CANCELLING, NOT CANCELLED; deleting a non-terminal job must
be refused — deleting a running job would be data loss).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

import ml_engine.jobs.manager as manager_mod
from ml_engine.jobs.manager import JobManager, get_job_manager
from ml_engine.jobs.models import Job, JobStatus


@pytest.fixture
def store():
    """Patch RedisJobStore so JobManager() gets a mock store instance."""
    with patch.object(manager_mod, "RedisJobStore") as StoreCls:
        yield StoreCls.return_value


@pytest.fixture
def manager(store):
    return JobManager()


def _job(status=JobStatus.PENDING, **kw):
    return Job(status=status, **kw)


# --------------------------------------------------------------------------- #
# submit_job
# --------------------------------------------------------------------------- #


class TestSubmitJob:
    def test_valid_job_is_enqueued(self, manager, store):
        job = manager.submit_job(job_type="teacher_training", config={"a": 1}, priority=2, tags=["x"])
        store.enqueue_job.assert_called_once_with(job)
        assert job.type == "teacher_training"
        assert job.status == JobStatus.PENDING
        assert job.config == {"a": 1}
        assert job.priority == 2
        assert job.tags == ["x"]

    def test_invalid_job_type_raises_and_does_not_enqueue(self, manager, store):
        with pytest.raises(ValueError, match="Invalid job type"):
            manager.submit_job(job_type="not_a_real_type", config={})
        store.enqueue_job.assert_not_called()

    def test_none_tags_becomes_empty_list(self, manager, store):
        job = manager.submit_job(job_type="teacher_training", config={})
        assert job.tags == []


# --------------------------------------------------------------------------- #
# cancel_job — the state machine
# --------------------------------------------------------------------------- #


class TestCancelJob:
    def test_not_found_returns_false_and_no_side_effects(self, manager, store):
        store.get_job.return_value = None
        assert manager.cancel_job("jid") is False
        store.update_job.assert_not_called()
        store.publish_event.assert_not_called()

    @pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
    def test_terminal_job_is_not_cancelled(self, manager, store, status):
        store.get_job.return_value = _job(status=status)
        assert manager.cancel_job("jid") is False
        store.update_job.assert_not_called()  # terminal must be a no-op
        store.publish_event.assert_not_called()

    def test_pending_marked_cancelled_with_finished_at(self, manager, store):
        store.get_job.return_value = _job(status=JobStatus.PENDING)
        assert manager.cancel_job("jid") is True
        store.update_job.assert_called_once()
        args, kwargs = store.update_job.call_args
        assert args[0] == "jid"
        assert kwargs["status"] == JobStatus.CANCELLED
        assert isinstance(kwargs["finished_at"], datetime)
        store.publish_event.assert_called_once()

    def test_pending_cancel_removes_job_from_queue_list(self, manager, store):
        # TRAP (was a real bug): cancelling a PENDING job must LREM it from the
        # queue LIST, not just flip status — otherwise the id lingers and a worker
        # could dequeue a "cancelled" job. remove_from_queue exists for this.
        store.get_job.return_value = _job(status=JobStatus.PENDING)
        manager.cancel_job("jid")
        store.remove_from_queue.assert_called_once_with("jid")

    def test_running_cancel_does_not_touch_queue(self, manager, store):
        # A RUNNING job isn't in the queue; cancel must NOT try to LREM it.
        store.get_job.return_value = _job(status=JobStatus.RUNNING)
        manager.cancel_job("jid")
        store.remove_from_queue.assert_not_called()

    def test_running_marked_cancelling_without_finished_at(self, manager, store):
        store.get_job.return_value = _job(status=JobStatus.RUNNING)
        assert manager.cancel_job("jid") is True
        # TRAP: a running job must transition to CANCELLING (graceful stop), NOT
        # straight to CANCELLED, and must NOT set finished_at yet.
        store.update_job.assert_called_once_with("jid", status=JobStatus.CANCELLING)

    def test_already_cancelling_publishes_event_but_no_update(self, manager, store):
        store.get_job.return_value = _job(status=JobStatus.CANCELLING)
        assert manager.cancel_job("jid") is True
        store.update_job.assert_not_called()  # nothing new to set
        store.publish_event.assert_called_once()  # but re-signals cancellation


# --------------------------------------------------------------------------- #
# delete_job — the terminal guard
# --------------------------------------------------------------------------- #


class TestDeleteJob:
    def test_not_found_returns_false(self, manager, store):
        store.get_job.return_value = None
        assert manager.delete_job("jid") is False
        store.delete_job.assert_not_called()

    @pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.CANCELLING])
    def test_non_terminal_job_is_not_deleted(self, manager, store, status):
        store.get_job.return_value = _job(status=status)
        # TRAP: deleting a running/pending job would be data loss — must refuse.
        assert manager.delete_job("jid") is False
        store.delete_job.assert_not_called()

    @pytest.mark.parametrize("result", [True, False])
    def test_terminal_job_delegates_and_propagates_result(self, manager, store, result):
        store.get_job.return_value = _job(status=JobStatus.COMPLETED)
        store.delete_job.return_value = result
        assert manager.delete_job("jid") is result
        store.delete_job.assert_called_once_with("jid")


# --------------------------------------------------------------------------- #
# list_jobs / get_job_count — status filtering
# --------------------------------------------------------------------------- #


class TestListAndCount:
    def test_valid_status_is_passed_as_enum(self, manager, store):
        store.list_jobs.return_value = []
        manager.list_jobs(status="running", job_type="teacher_training", limit=5, offset=2)
        store.list_jobs.assert_called_once_with(
            status=JobStatus.RUNNING, job_type="teacher_training", limit=5, offset=2
        )

    def test_invalid_status_falls_back_to_no_filter(self, manager, store, caplog):
        # DOCUMENTED GAP: a typo'd status does NOT return empty or error — it logs
        # a warning and lists ALL jobs (status=None). Callers filtering by a bad
        # status silently get everything.
        store.list_jobs.return_value = []
        with caplog.at_level("WARNING"):
            manager.list_jobs(status="bogus")
        assert any("Invalid status filter" in r.message for r in caplog.records)
        assert store.list_jobs.call_args.kwargs["status"] is None

    def test_get_job_count_counts_returned_jobs(self, manager, store):
        store.list_jobs.return_value = [_job(), _job(), _job()]
        assert manager.get_job_count("running") == 3


# --------------------------------------------------------------------------- #
# queue status + delegation
# --------------------------------------------------------------------------- #


class TestQueueAndDelegation:
    def test_get_queue_status_structure(self, manager, store):
        store.get_queue_length.return_value = 4
        store.list_workers.return_value = []
        store.list_jobs.return_value = []
        status = manager.get_queue_status()
        assert status["queue_length"] == 4
        assert status["workers"] == []
        assert set(status["job_counts"]) == {"pending", "running", "completed", "failed", "cancelled"}

    def test_simple_delegations(self, manager, store):
        manager.get_queue_length()
        store.get_queue_length.assert_called_once()
        manager.list_workers(status="idle")
        store.list_workers.assert_called_once_with(status="idle")
        manager.cleanup_stale_workers(timeout_seconds=30)
        store.cleanup_stale_workers.assert_called_once_with(30)
        manager.get_job("jid")
        store.get_job.assert_called_with("jid")
        manager.close()
        store.close.assert_called_once()


# --------------------------------------------------------------------------- #
# singleton
# --------------------------------------------------------------------------- #


class TestSingleton:
    def test_get_job_manager_returns_same_instance(self, store):
        manager_mod._default_manager = None  # reset module global
        try:
            m1 = get_job_manager("redis://a")
            m2 = get_job_manager("redis://b")  # different url is IGNORED after first call
            assert m1 is m2
            assert m1.redis_url == "redis://a"  # documents the ignored-url gotcha
        finally:
            manager_mod._default_manager = None
