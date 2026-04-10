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
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis as _redis

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


class StateMachine:
    """
    Persistent lifecycle state machine for one pipeline run.

    Redis key: `run:{run_id}:state` (HASH with state, stage, metadata)
    """

    _PREFIX = "run:"

    def __init__(self, redis_client: _redis.Redis, run_id: str):
        self._r = redis_client
        self.run_id = run_id
        self._key = f"{self._PREFIX}{run_id}:state"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def initialize(self, contract: Optional[Dict[str, Any]] = None) -> None:
        """
        Create the state record in Redis. Call once at pipeline start.

        Persists the proposed contract so the frontend can retrieve it via
        GET /api/agent/status/{run_id} without relying on the HTTP response
        from POST /api/agent/plan.
        """
        self._r.hset(self._key, mapping={
            "run_id": self.run_id,
            "state": "created",
            "contract_id": contract.get("id", "") if contract else "",
            "proposed_contract": json.dumps(contract or {}),
            "retry_count": "0",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "error_message": "",
            "stage_summaries": "[]",
        })
        logger.info("Run %s initialized (state=created)", self.run_id)

    def get_proposed_contract(self) -> Optional[Dict[str, Any]]:
        """Return the contract proposed at plan time, or None if not stored."""
        raw = self._r.hget(self._key, "proposed_contract")
        if not raw:
            return None
        decoded = raw.decode() if isinstance(raw, bytes) else raw
        try:
            result = json.loads(decoded)
            return result if result else None
        except json.JSONDecodeError:
            return None

    def load(self) -> Dict[str, Any]:
        """Load raw state dict from Redis."""
        data = self._r.hgetall(self._key)
        if not data:
            raise KeyError(f"No state found for run {self.run_id}")
        return {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in data.items()
        }

    @property
    def current_state(self) -> str:
        r"""Return the current lifecycle state. Raises KeyError if state not found."""
        raw = self._r.hget(self._key, "state")
        if raw is None:
            raise KeyError(f"No state for run {self.run_id}")
        return raw.decode() if isinstance(raw, bytes) else raw

    @property
    def retry_count(self) -> int:
        r"""Return the current retry count (number of times we've entered failed_retrying)."""
        raw = self._r.hget(self._key, "retry_count") or b"0"
        return int(raw.decode() if isinstance(raw, bytes) else raw)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        new_state: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Move to new_state. Raises ValueError if the transition is not allowed.
        """
        current = self.current_state
        if new_state not in STATES:
            raise ValueError(f"Unknown state: {new_state}")
        if current in TERMINAL_STATES:
            raise ValueError(f"Run {self.run_id} is in terminal state {current!r}")
        allowed = TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {current!r} -> {new_state!r}. "
                f"Allowed: {allowed}"
            )

        updates: Dict[str, str] = {
            "state": new_state,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if error_message is not None:
            updates["error_message"] = error_message
        if new_state == "failed_retrying":
            updates["retry_count"] = str(self.retry_count + 1)
        if metadata:
            updates["metadata"] = json.dumps(metadata)

        self._r.hset(self._key, mapping=updates)
        logger.info("Run %s: %s -> %s", self.run_id, current, new_state)

    def append_stage_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Append a StageSummary dict to the stage_summaries list."""
        raw = self._r.hget(self._key, "stage_summaries") or b"[]"
        summaries = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        summaries.append(summary_dict)
        self._r.hset(self._key, "stage_summaries", json.dumps(summaries))

    def get_stage_summaries(self) -> List[Dict[str, Any]]:
        r"""Return the list of stage summaries, or empty list if none."""
        raw = self._r.hget(self._key, "stage_summaries") or b"[]"
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    @classmethod
    def exists(cls, redis_client: _redis.Redis, run_id: str) -> bool:
        r"""Check if state exists for given run_id."""
        return redis_client.exists(f"{cls._PREFIX}{run_id}:state") > 0
