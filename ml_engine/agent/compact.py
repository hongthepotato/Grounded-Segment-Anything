"""
Context compaction at stage boundaries.

Adapted from Claude Code's autoCompact -- trigger is stage completion,
not token pressure. ML agents accumulate 10-15 events per pipeline (not 200+
tool calls), so compaction is about cognitive focus, not window management.

After each stage completes, replace the raw execution history with a
StageSummary that preserves only what the next stage needs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ml_engine.agent.contracts import StageSummary

logger = logging.getLogger(__name__)

# Token budget per stage summary (approximate)
_MAX_SUMMARY_CHARS = 1500


def compact_stage(
    messages: List[Dict[str, Any]],
    stage_summary: StageSummary,
) -> List[Dict[str, Any]]:
    """
    Replace stage execution history with a compact StageSummary.

    Keeps:
    - System-level messages (role != "user"/"assistant")
    - Messages before the stage started (prior context)
    - The compact summary as a new user message
    - Drops: all tool calls, tool results, and progress events from this stage

    Args:
        messages: Current conversation history (role/content dicts).
        stage_summary: Structured summary of the completed stage.

    Returns:
        Compacted message list.
    """
    # Find where this stage started (look for the dispatch event)
    stage_start_idx = _find_stage_start(messages, stage_summary.stage)

    # Keep everything before the stage started
    pre_stage = messages[:stage_start_idx]

    # Append the compact summary
    summary_text = _format_summary(stage_summary)
    compact_message = {
        "role": "user",
        "content": f"[STAGE COMPLETE]\n{summary_text}",
    }

    result = pre_stage + [compact_message]
    chars_saved = sum(len(str(m)) for m in messages) - sum(len(str(m)) for m in result)
    logger.info(
        "Compacted stage %s: %d -> %d messages (~%d chars saved)",
        stage_summary.stage, len(messages), len(result), chars_saved,
    )
    return result


def _find_stage_start(messages: List[Dict[str, Any]], stage: str) -> int:
    """Return the index of the first message that belongs to this stage."""
    stage_markers = {
        f'"type": "job_started"',
        f'"stage": "{stage}"',
        f'dispatch_stage',
        stage,
    }
    for i in range(len(messages) - 1, -1, -1):
        content = str(messages[i].get("content", ""))
        if any(marker in content for marker in stage_markers):
            return i
    # Fallback: keep last 3 messages as pre-stage context
    return max(0, len(messages) - 3)


def _format_summary(s: StageSummary) -> str:
    lines = [
        f"Stage: {s.stage}",
        f"Status: {s.status}",
        f"Duration: {s.duration_seconds:.0f}s",
    ]
    if s.metrics:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in s.metrics.items())
        lines.append(f"Metrics: {metrics_str}")
    if s.trial_count is not None:
        lines.append(f"Trials: {s.trial_count}")
    if s.artifacts:
        items = list(s.artifacts.items())[:3]
        lines.append("Artifacts: " + ", ".join(f"{k}: {v}" for k, v in items))
    if s.key_decisions:
        lines.append("Key decisions:")
        for d in s.key_decisions[:5]:
            lines.append(f"  - {d}")
    return "\n".join(lines)
