"""
Context compaction at stage boundaries.

Adapted from Claude Code's autoCompact -- trigger is stage completion,
not token pressure. ML agents accumulate 10-15 events per pipeline (not 200+
tool calls), so compaction is about cognitive focus, not window management.

After each stage completes, replace the raw execution history with a
StageSummary that preserves only what the next stage needs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ml_engine.agent.contracts import StageSummary

logger = logging.getLogger(__name__)


def compact_stage(
    messages: List[Dict[str, Any]],
    stage_summary: StageSummary,
    stage_start_idx: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Replace stage execution history with a compact StageSummary.

    Keeps:
    - Messages before stage_start_idx (prior context)
    - The compact summary as a new user message
    Drops: the dispatch_stage call, all tool results, and events from this stage.

    Args:
        messages: Current conversation history (role/content dicts).
        stage_summary: Structured summary of the completed stage.
        stage_start_idx: Index of the assistant message that called dispatch_stage.
            Recorded at dispatch time so compaction uses the exact boundary.
            If None (e.g. state loaded from before this field existed), keeps all messages.

    Returns:
        Compacted message list.
    """
    if stage_start_idx is None:
        logger.warning(
            "compact_stage called without stage_start_idx for stage %s — skipping compaction",
            stage_summary.stage,
        )
        return messages

    pre_stage = messages[:stage_start_idx]

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
        lines.append("Artifacts: " + ", ".join(f"{k}: {v}" for k, v in s.artifacts.items()))
    if s.key_decisions:
        lines.append("Key decisions:")
        for d in s.key_decisions:
            lines.append(f"  - {d}")
    return "\n".join(lines)
