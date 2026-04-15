"""
Lifecycle state machine for agentic pipelines.

Backed by Redis HASHes (for run state) + Redis Streams (for events).
Survives Docker restarts because all state is in Redis.

State transitions:

    created
      -> planning      (Coordinator proposes contract)
      -> pending_contract_approval  (human reviews contract)
      -> auto_labeling
      -> label_review_gate
      -> teacher_training
      -> training_eval_gate
      -> student_distillation   (optional)
      -> distill_eval_gate      (optional)
      -> pending_approval       (human approves final model before production swap)
      -> done

Failure transitions (from any non-terminal state):
    -> failed_retrying      (job crashed, retries remaining)
    -> failed_unrecoverable (budget exhausted or unrecoverable error)
    -> escalated            (metric budget exhausted, needs human)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as _aredis

logger = logging.getLogger(__name__)

# All valid states
STATES = {
    # Active
    "created",
    "planning",
    "pending_contract_approval",
    "auto_labeling",
    "label_review_gate",
    "teacher_training",
    "training_eval_gate",
    "student_distillation",
    "distill_eval_gate",
    "pending_approval",
    # Terminal
    "done",
    "failed_retrying",
    "failed_unrecoverable",
    "escalated",
    "cancelled",
}

TERMINAL_STATES = {"done", "failed_unrecoverable", "escalated", "cancelled"}

# Gate groups: which stages each gate function applies to.
# gate.py imports these so stage membership is a single source of truth.
TEACHER_GATE_STAGES = frozenset({"teacher_training", "training_eval_gate", "experiment_loop"})
DISTILLATION_GATE_STAGES = frozenset({"student_distillation", "distill_eval_gate"})

# Valid transitions: state -> set of reachable states
TRANSITIONS: Dict[str, List[str]] = {
    "created":                      ["planning"],
    "planning":                     ["pending_contract_approval", "failed_unrecoverable"],
    "pending_contract_approval":    ["auto_labeling", "teacher_training", "cancelled"],
    "auto_labeling":                ["label_review_gate", "failed_retrying", "failed_unrecoverable"],
    "label_review_gate":            ["teacher_training", "auto_labeling", "escalated"],
    "teacher_training":             ["training_eval_gate", "failed_retrying", "failed_unrecoverable"],
    "training_eval_gate":           ["student_distillation", "pending_approval", "teacher_training", "escalated"],
    "student_distillation":         ["distill_eval_gate", "failed_retrying", "failed_unrecoverable"],
    "distill_eval_gate":            ["pending_approval", "student_distillation", "escalated"],
    "pending_approval":             ["done", "teacher_training", "cancelled"],
    "failed_retrying":              ["teacher_training", "auto_labeling", "student_distillation", "failed_unrecoverable"],
    # terminal states have no outbound transitions
}


def _decode(raw: Any) -> str:
    """Return bytes-or-str raw value as str."""
    return raw.decode() if isinstance(raw, bytes) else raw


def _decode_mapping(data: Dict[Any, Any]) -> Dict[str, str]:
    return {_decode(k): _decode(v) for k, v in data.items()}


def _validate_transition(current: str, new_state: str, run_id: str) -> None:
    """Validate that current -> new_state is permitted."""
    if new_state not in STATES:
        raise ValueError(f"Unknown state: {new_state}")
    if current in TERMINAL_STATES:
        raise ValueError(f"Run {run_id} is in terminal state {current!r}")
    allowed = TRANSITIONS.get(current, [])
    if new_state not in allowed:
        raise ValueError(
            f"Invalid transition {current!r} -> {new_state!r}. "
            f"Allowed: {allowed}"
        )


def _initial_state_mapping(run_id: str, contract: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Build the initial HASH mapping for a freshly created run."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "state": "created",
        "contract_id": contract.get("id", "") if contract else "",
        "proposed_contract": json.dumps(contract or {}),
        "retry_count": "0",
        "created_at": now,
        "updated_at": now,
        "error_message": "",
        "stage_summaries": "[]",
    }


def _build_transition_updates(
    new_state: str,
    retry_count: int,
    error_message: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Build the update-dict written to Redis on a transition."""
    updates: Dict[str, str] = {
        "state": new_state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_message is not None:
        updates["error_message"] = error_message
    if new_state == "failed_retrying":
        updates["retry_count"] = str(retry_count + 1)
    if metadata:
        updates["metadata"] = json.dumps(metadata)
    return updates


class StateMachine:
    """
    Persistent lifecycle state machine for one pipeline run.

    Redis key: `run:{run_id}:state` (HASH with state, stage, metadata).
    """

    _PREFIX = "run:"

    def __init__(self, run_id: str, redis_async: _aredis.Redis):
        self._r = redis_async
        self.run_id = run_id
        self._key = f"{self._PREFIX}{run_id}:state"

    async def initialize(self, contract: Optional[Dict[str, Any]] = None) -> None:
        """
        Create the state record in Redis. Call once at pipeline start.

        Persists the proposed contract so the frontend can retrieve it via
        GET /api/agent/status/{run_id} without relying on the HTTP response
        from POST /api/agent/plan.
        """
        await self._r.hset(
            self._key, mapping=_initial_state_mapping(self.run_id, contract),
        )
        logger.info("Run %s initialized (state=created)", self.run_id)

    async def get_proposed_contract(self) -> Optional[Dict[str, Any]]:
        """Return the contract proposed at plan time, or None if not stored."""
        raw = await self._r.hget(self._key, "proposed_contract")
        if not raw:
            return None
        try:
            result = json.loads(_decode(raw))
            return result if result else None
        except json.JSONDecodeError:
            return None

    async def load(self) -> Dict[str, Any]:
        """Load raw state dict from Redis."""
        data = await self._r.hgetall(self._key)
        if not data:
            raise KeyError(f"No state found for run {self.run_id}")
        return _decode_mapping(data)

    async def current_state(self) -> str:
        """Return the current lifecycle state. Raises KeyError if not found."""
        raw = await self._r.hget(self._key, "state")
        if raw is None:
            raise KeyError(f"No state for run {self.run_id}")
        return _decode(raw)

    async def retry_count(self) -> int:
        """Return the current retry count (times we've entered failed_retrying)."""
        raw = await self._r.hget(self._key, "retry_count") or b"0"
        return int(_decode(raw))

    async def transition(
        self,
        new_state: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Move to new_state. Raises ValueError if the transition is not allowed."""
        current = await self.current_state()
        _validate_transition(current, new_state, self.run_id)
        retry_count = await self.retry_count() if new_state == "failed_retrying" else 0
        updates = _build_transition_updates(new_state, retry_count, error_message, metadata)
        await self._r.hset(self._key, mapping=updates)
        logger.info("Run %s: %s -> %s", self.run_id, current, new_state)

    async def append_stage_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Append a StageSummary dict to the stage_summaries list."""
        raw = await self._r.hget(self._key, "stage_summaries") or b"[]"
        summaries = json.loads(_decode(raw))
        summaries.append(summary_dict)
        await self._r.hset(self._key, "stage_summaries", json.dumps(summaries))

    async def get_stage_summaries(self) -> List[Dict[str, Any]]:
        """Return the list of stage summaries, or empty list if none."""
        raw = await self._r.hget(self._key, "stage_summaries") or b"[]"
        return json.loads(_decode(raw))

    @classmethod
    async def exists(cls, redis_async: _aredis.Redis, run_id: str) -> bool:
        """Check if state exists for given run_id."""
        return (await redis_async.exists(f"{cls._PREFIX}{run_id}:state")) > 0
