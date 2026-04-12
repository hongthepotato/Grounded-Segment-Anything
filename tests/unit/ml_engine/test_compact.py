"""
Unit tests for ml_engine.agent.compact.

compact_stage is pure -- no Redis, no I/O.
"""

from __future__ import annotations

import pytest

from ml_engine.agent.compact import compact_stage, _format_summary
from ml_engine.agent.contracts import StageSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_summary(**kwargs) -> StageSummary:
    defaults = dict(
        stage="teacher_training",
        status="pass",
        metrics={"mAP50": 0.72},
        artifacts={"checkpoint": "checkpoint.pt"},
        key_decisions=["Used LoRA r=32"],
        duration_seconds=3600.0,
        trial_count=None,
    )
    defaults.update(kwargs)
    return StageSummary(**defaults)


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# _format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    r"""Tests for the _format_summary function which converts a StageSummary into a text block."""
    def test_includes_stage_name(self):
        r"""Stage name should be included in the summary text."""
        s = make_summary(stage="teacher_training")
        text = _format_summary(s)
        assert "teacher_training" in text

    def test_includes_status(self):
        r"""Status should be included in the summary text."""
        s = make_summary(status="pass")
        text = _format_summary(s)
        assert "pass" in text

    def test_includes_duration(self):
        r"""Duration should be included in the summary text."""
        s = make_summary(duration_seconds=7200.0)
        text = _format_summary(s)
        assert "7200" in text

    def test_includes_metrics_when_present(self):
        r"""Metrics should be included when the metrics dict is not empty."""
        s = make_summary(metrics={"mAP50": 0.72, "val_loss": 0.5})
        text = _format_summary(s)
        assert "Metrics: mAP50=0.7200, val_loss=0.5000" in text

    def test_no_metrics_section_when_empty(self):
        r"""If metrics dict is empty, the summary should not include a Metrics section."""
        s = make_summary(metrics={})
        text = _format_summary(s)
        assert "Metrics:" not in text

    def test_includes_trial_count_when_set(self):
        r"""Trial count should be included when trial_count is not None."""
        s = make_summary(trial_count=12)
        text = _format_summary(s)
        assert "Trials: 12" in text

    def test_no_trials_section_when_none(self):
        r"""If trial_count is None, the summary should not include a Trials section."""
        s = make_summary(trial_count=None)
        text = _format_summary(s)
        assert "Trials:" not in text

    def test_includes_artifacts(self):
        r"""Artifacts should be included when the artifacts dict is not empty."""
        s = make_summary(artifacts={"checkpoint": "model.pt", "eval_report": "eval.json"})
        text = _format_summary(s)
        assert "Artifacts: checkpoint: model.pt, eval_report: eval.json" in text

    def test_includes_key_decisions(self):
        r"""Key decisions should be listed when key_decisions is not empty."""
        s = make_summary(key_decisions=["Chose AdamW", "LoRA r=64"])
        text = _format_summary(s)
        assert "Key decisions:\n  - Chose AdamW\n  - LoRA r=64" in text


# ---------------------------------------------------------------------------
# compact_stage
# ---------------------------------------------------------------------------

class TestCompactStage:
    r"""Tests for the compact_stage function which compacts a message
    list by replacing a stage's execution history with a summary."""
    def test_returns_list(self):
        r"""The result of compact_stage should be a list of messages."""
        messages = [
            msg("user", "[EVENT] contract_approved"),
            msg("assistant", "dispatch_stage teacher_training"),  # stage_start_idx=1
            msg("user", "[EVENT] job_started teacher_training"),
            msg("user", "[EVENT] job_completed"),
        ]
        result = compact_stage(messages, make_summary(), stage_start_idx=1)
        assert isinstance(result, list)

    def test_result_ends_with_stage_complete_message(self):
        r"""The last message in the result should indicate stage completion and include the summary."""
        messages = [
            msg("user", "[EVENT] contract_approved"),
            msg("assistant", "dispatch_stage teacher_training"),  # stage_start_idx=1
            msg("user", "[EVENT] job_completed teacher_training"),
        ]
        result = compact_stage(messages, make_summary(stage="teacher_training"), stage_start_idx=1)
        last = result[-1]
        assert "[STAGE COMPLETE]" in last["content"]

    def test_summary_content_in_last_message(self):
        r"""The summary content should be included in the last message of the result."""
        messages = [
            msg("assistant", "dispatch_stage teacher_training"),  # stage_start_idx=0
            msg("user", "[EVENT] job_completed"),
        ]
        summary = make_summary(stage="teacher_training", metrics={"mAP50": 0.88})
        result = compact_stage(messages, summary, stage_start_idx=0)
        last = result[-1]
        assert "teacher_training" in last["content"]
        assert "0.8800" in last["content"]

    def test_pre_stage_messages_preserved(self):
        r"""Messages before stage_start_idx should be preserved in the result, while messages from the stage should be dropped."""
        pre = [msg("user", f"pre-context-{i}") for i in range(3)]
        stage_msgs = [
            msg("assistant", "dispatch_stage teacher_training"),
            msg("user", "[EVENT] job_completed teacher_training"),
        ]
        # stage starts at index 3 (first message after the 3 pre-stage messages)
        result = compact_stage(pre + stage_msgs, make_summary(), stage_start_idx=3)
        result_contents = [m["content"] for m in result]
        assert any("pre-context-0" in c for c in result_contents)
        assert "dispatch_stage teacher_training" not in result_contents

    def test_uses_exact_boundary(self):
        r"""stage_start_idx=2 keeps exactly messages 0 and 1, drops 2 onward."""
        messages = [
            msg("user", "pre-0"),
            msg("user", "pre-1"),
            msg("assistant", "dispatch_stage teacher_training"),  # boundary
            msg("user", "[EVENT] job_started"),
            msg("user", "[EVENT] job_completed"),
        ]
        result = compact_stage(messages, make_summary(), stage_start_idx=2)
        # pre-0 and pre-1 kept, dispatch and beyond dropped, summary appended
        assert len(result) == 3
        assert result[0]["content"] == "pre-0"
        assert result[1]["content"] == "pre-1"
        assert "[STAGE COMPLETE]" in result[2]["content"]

    def test_result_shorter_than_input(self):
        r"""Compaction must reduce message count when stage has many events."""
        messages = (
            [msg("user", "contract_approved")] +
            [msg("user", f"[EVENT] progress {i} teacher_training") for i in range(20)] +
            [msg("user", "[EVENT] job_completed teacher_training")]
        )
        # stage starts at index 1 (first progress event)
        result = compact_stage(messages, make_summary(stage="teacher_training"), stage_start_idx=1)
        assert len(result) < len(messages)

    def test_empty_messages_returns_compact_only(self):
        r"""If input messages is empty, the result should be a single message with the stage summary."""
        result = compact_stage([], make_summary(), stage_start_idx=0)
        assert len(result) == 1
        assert "[STAGE COMPLETE]" in result[0]["content"]

    def test_last_message_role_is_user(self):
        r"""The last message in the result should have role 'user' since it contains the stage summary."""
        messages = [msg("assistant", "dispatch_stage teacher_training")]
        result = compact_stage(messages, make_summary(), stage_start_idx=0)
        assert result[-1]["role"] == "user"

    def test_none_stage_start_idx_skips_compaction(self):
        r"""When stage_start_idx is None (e.g. state from before the field existed),
        compaction is skipped and the original message list is returned unchanged."""
        messages = [
            msg("user", "contract_approved"),
            msg("assistant", "dispatch_stage teacher_training"),
            msg("user", "[EVENT] job_completed"),
        ]
        result = compact_stage(messages, make_summary(), stage_start_idx=None)
        assert result is messages
