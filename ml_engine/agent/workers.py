"""
Stage 2: Executor worker.
Stage 3: Evaluator worker.

The Executor is a fire-and-forget job submitter that sits between the
Coordinator and the job queue. It:

  1. Reads `dispatch_requested` events from the agent Stream (consumer group "executor")
  2. Validates contract constraints BEFORE the job enters the work queue
  3. Enqueues the job if valid, publishes `dispatch_rejected` if not

Why a separate worker (not inline in DispatchStageTool)?
  - Contract validation must happen BEFORE the job enters the queue.
    If validation is inline in the tool, the Coordinator is doing two jobs
    (deciding AND enforcing), making it harder to test and reason about.
  - The Executor can be moved to its own container later (TODOS.md TODO-3)
    without changing the Coordinator at all.
  - Consumer group separation: Coordinator reads events it cares about
    (job_completed, gate_*). Executor reads events it cares about
    (dispatch_requested). Both share the same Stream, different cursors.

Failure modes handled:
  - job_id not found in store: publishes dispatch_rejected
  - retry budget exhausted: publishes dispatch_rejected
  - contract missing: allows dispatch (pre-approval planning phase)
  - Redis failure: logs and continues (retried on next loop iteration)

Usage:
    # Typically launched by Coordinator.run() as a concurrent asyncio task.
    executor = ExecutorWorker(redis_client, run_id, store, contract)
    await executor.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import redis as _redis

from ml_engine.agent.contracts import PipelineContract, StageSummary
from ml_engine.agent.gate import evaluate_gate
from ml_engine.agent.loop import (
    STREAM_BLOCK_MS,
    ensure_consumer_group,
    publish_event,
    stream_key,
)
from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine

logger = logging.getLogger(__name__)


class ExecutorWorker:
    """
    Async fire-and-forget job executor -- Stage 2.

    Listens on the pipeline's agent Stream for `dispatch_requested` events.
    Validates contract constraints, then either enqueues the job or rejects it.

    This worker runs concurrently with the Coordinator in the same asyncio
    event loop (launched via asyncio.gather in Coordinator.run()).
    """

    CONSUMER_GROUP = "executor"
    CONSUMER_NAME = "executor-0"

    def __init__(
        self,
        redis_client: _redis.Redis,
        run_id: str,
        store,               # RedisJobStore -- typed loosely to avoid circular import
        contract: Optional[PipelineContract] = None,
    ):
        self._r = redis_client
        self.run_id = run_id
        self._store = store
        self._contract = contract

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, cancel_check=None) -> None:
        """
        Block on the agent Stream and process dispatch_requested events.

        Exits when the pipeline reaches a terminal state or cancel_check() returns True.
        """
        ensure_consumer_group(self._r, self.run_id)
        key = stream_key(self.run_id)

        logger.info("ExecutorWorker started for run %s", self.run_id)

        while True:
            if cancel_check and cancel_check():
                logger.info("ExecutorWorker cancelled for run %s", self.run_id)
                break

            # Stop when pipeline reaches terminal state
            sm = StateMachine(self._r, self.run_id)
            try:
                if sm.current_state in TERMINAL_STATES:
                    logger.info("Run %s reached terminal state, ExecutorWorker stopping", self.run_id)
                    break
            except KeyError:
                pass  # State not initialized yet -- keep running

            try:
                entries = self._r.xreadgroup(
                    groupname=self.CONSUMER_GROUP,
                    consumername=self.CONSUMER_NAME,
                    streams={key: ">"},
                    count=5,
                    block=STREAM_BLOCK_MS,
                )
            except _redis.RedisError as e:
                logger.error("ExecutorWorker Redis error: %s", e)
                await asyncio.sleep(1)
                continue

            if not entries:
                continue

            for _stream_key, messages in entries:
                for entry_id, entry_data in messages:
                    entry_id_str = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                    raw = entry_data.get(b"data", entry_data.get("data", "{}"))
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    event = json.loads(raw)

                    if event.get("type") == "dispatch_requested":
                        try:
                            await self._handle_dispatch(event)
                        except Exception as e:
                            logger.error("ExecutorWorker dispatch error: %s", e, exc_info=True)

                    # Always ACK -- poison messages must not stall the queue
                    self._r.xack(key, self.CONSUMER_GROUP, entry_id_str)

    # ------------------------------------------------------------------
    # Dispatch handler
    # ------------------------------------------------------------------

    async def _handle_dispatch(self, event: Dict[str, Any]) -> None:
        """
        Validate and enqueue a single dispatch_requested event.

        On success: publishes stage_dispatched to agent Stream.
        On failure: publishes dispatch_rejected with reason.
        """
        job_id = event.get("job_id")
        stage = event.get("stage", "unknown")

        if not job_id:
            logger.warning("dispatch_requested missing job_id, skipping")
            return

        logger.info("ExecutorWorker handling dispatch: stage=%s job=%s", stage, job_id[:8])

        # Validate contract constraints
        errors = self._validate(stage)
        if errors:
            logger.warning(
                "Dispatch rejected for job %s (stage=%s): %s",
                job_id[:8], stage, "; ".join(errors),
            )
            publish_event(self._r, self.run_id, {
                "type": "dispatch_rejected",
                "job_id": job_id,
                "stage": stage,
                "run_id": self.run_id,
                "errors": errors,
            })
            return

        # Enqueue the pre-stored job
        success = self._store.enqueue_by_id(job_id)
        if not success:
            logger.error("ExecutorWorker: enqueue_by_id failed for job %s", job_id[:8])
            publish_event(self._r, self.run_id, {
                "type": "dispatch_rejected",
                "job_id": job_id,
                "stage": stage,
                "run_id": self.run_id,
                "errors": [f"Job {job_id[:8]} not found in store"],
            })
            return

        publish_event(self._r, self.run_id, {
            "type": "stage_dispatched",
            "job_id": job_id,
            "stage": stage,
            "run_id": self.run_id,
        })
        logger.info("ExecutorWorker: enqueued %s (job=%s)", stage, job_id[:8])

    # ------------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------------

    def _validate(self, stage: str) -> list[str]:
        """
        Return a list of constraint violations. Empty list = dispatch allowed.

        Checks:
        - Retry count vs contract.budget.max_retries
        - (Future: stage ordering against state machine)
        """
        if self._contract is None:
            return []  # Pre-approval / planning phase: no constraints yet

        errors: list[str] = []

        sm = StateMachine(self._r, self.run_id)
        try:
            retry_count = sm.retry_count
            max_retries = self._contract.budget.max_retries
            if retry_count >= max_retries:
                errors.append(
                    f"Retry budget exhausted for stage {stage!r}: "
                    f"{retry_count}/{max_retries} retries used"
                )
        except KeyError:
            pass  # No state yet, allow

        return errors

    # ------------------------------------------------------------------
    # Contract update (called by Coordinator when contract is approved)
    # ------------------------------------------------------------------

    def set_contract(self, contract: PipelineContract) -> None:
        """Update the contract used for validation. Called when user approves."""
        self._contract = contract
        logger.info("ExecutorWorker: contract updated for run %s", self.run_id)


# ---------------------------------------------------------------------------
# Stage 3: EvaluatorWorker
# ---------------------------------------------------------------------------

class EvaluatorWorker:
    """
    Metric-based evaluation worker -- Stage 3.

    Listens on the pipeline's agent Stream for `evaluation_requested` events.
    Runs deterministic gate evaluation (no LLM), writes feedback memory,
    and publishes a `gate_decision` event for the Coordinator to act on.

    Why separate from RequestEvaluationTool?
    - Tool executes inline in the Coordinator's turn (LLM-driven).
      Evaluation should be independent: deterministic, testable, no LLM needed.
    - Consumer group separation: Coordinator reads job_completed / gate_decision.
      Evaluator reads evaluation_requested. Independent cursors, no head-of-line blocking.
    - At Stage 3+, Evaluator writes memory feedback that future runs learn from.
      That write belongs here, not in the Coordinator.
    """

    CONSUMER_GROUP = "evaluator"
    CONSUMER_NAME = "evaluator-0"

    def __init__(
        self,
        redis_client: _redis.Redis,
        run_id: str,
        contract: Optional[PipelineContract] = None,
    ):
        self._r = redis_client
        self.run_id = run_id
        self._contract = contract

    async def run(self, cancel_check=None) -> None:
        """
        Block on the agent Stream and process evaluation_requested events.

        Exits when the pipeline reaches a terminal state or cancel_check() returns True.
        """
        ensure_consumer_group(self._r, self.run_id)
        key = stream_key(self.run_id)

        logger.info("EvaluatorWorker started for run %s", self.run_id)

        while True:
            if cancel_check and cancel_check():
                logger.info("EvaluatorWorker cancelled for run %s", self.run_id)
                break

            sm = StateMachine(self._r, self.run_id)
            try:
                if sm.current_state in TERMINAL_STATES:
                    logger.info("Run %s reached terminal state, EvaluatorWorker stopping", self.run_id)
                    break
            except KeyError:
                pass

            try:
                entries = self._r.xreadgroup(
                    groupname=self.CONSUMER_GROUP,
                    consumername=self.CONSUMER_NAME,
                    streams={key: ">"},
                    count=5,
                    block=STREAM_BLOCK_MS,
                )
            except _redis.RedisError as e:
                logger.error("EvaluatorWorker Redis error: %s", e)
                await asyncio.sleep(1)
                continue

            if not entries:
                continue

            for _stream_key, messages in entries:
                for entry_id, entry_data in messages:
                    entry_id_str = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                    raw = entry_data.get(b"data", entry_data.get("data", "{}"))
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    event = json.loads(raw)

                    if event.get("type") == "evaluation_requested":
                        try:
                            await self._handle_evaluation(event)
                        except Exception as e:
                            logger.error("EvaluatorWorker evaluation error: %s", e, exc_info=True)
                            # Publish escalation so the pipeline doesn't silently stall
                            publish_event(self._r, self.run_id, {
                                "type": "gate_decision",
                                "stage": event.get("stage", "unknown"),
                                "verdict": "escalate",
                                "reason": f"Evaluator error: {e}",
                                "metrics": {},
                                "run_id": self.run_id,
                            })

                    # Always ACK -- poison messages must not stall the queue
                    self._r.xack(key, self.CONSUMER_GROUP, entry_id_str)

    async def _handle_evaluation(self, event: Dict[str, Any]) -> None:
        """
        Evaluate one completed stage.

        Steps:
        1. Read outcome.json from job output_dir
        2. Run deterministic gate check against contract criteria
        3. Write feedback memory record
        4. Publish gate_decision event for Coordinator
        """
        import json as _json
        from pathlib import Path

        stage = event.get("stage", "unknown")
        output_dir = event.get("output_dir", "")
        job_id = event.get("job_id", "")

        logger.info("EvaluatorWorker evaluating stage=%s job=%s", stage, job_id[:8] if job_id else "?")

        # Read outcome.json
        outcome_path = Path(output_dir) / "outcome.json"
        if not outcome_path.exists():
            logger.error("EvaluatorWorker: outcome.json not found at %s", output_dir)
            publish_event(self._r, self.run_id, {
                "type": "gate_decision",
                "stage": stage,
                "verdict": "escalate",
                "reason": f"outcome.json not found at {output_dir}",
                "metrics": {},
                "run_id": self.run_id,
            })
            return

        outcome = _json.loads(outcome_path.read_text(encoding="utf-8"))
        metrics = outcome.get("metrics", {})
        wall_time = outcome.get("wall_time_seconds", 0.0)
        artifacts = outcome.get("artifacts", {})

        # Gate evaluation (deterministic)
        criteria = self._contract.acceptance_criteria if self._contract else None
        sm = StateMachine(self._r, self.run_id)
        retry_count = sm.retry_count
        max_retries = self._contract.budget.max_retries if self._contract else 2

        if criteria is None:
            # No contract: default pass (pre-approval planning phase)
            decision_verdict = "pass"
            decision_reason = "no contract -- defaulting to pass"
        else:
            decision = evaluate_gate(metrics, criteria, retry_count, max_retries, stage)
            decision_verdict = decision.verdict
            decision_reason = decision.reason

        logger.info(
            "EvaluatorWorker gate for %s: %s (%s)",
            stage, decision_verdict, decision_reason,
        )

        # Write feedback memory -- future pipelines learn from this
        self._write_feedback_memory(
            stage=stage,
            verdict=decision_verdict,
            metrics=metrics,
            outcome=outcome,
            wall_time=wall_time,
        )

        # Publish gate_decision for Coordinator to act on
        publish_event(self._r, self.run_id, {
            "type": "gate_decision",
            "stage": stage,
            "verdict": decision_verdict,
            "reason": decision_reason,
            "metrics": metrics,
            "artifacts": artifacts,
            "wall_time_seconds": wall_time,
            "run_id": self.run_id,
        })

    def _write_feedback_memory(
        self,
        stage: str,
        verdict: str,
        metrics: Dict[str, Any],
        outcome: Dict[str, Any],
        wall_time: float,
    ) -> None:
        """
        Write structured feedback to MemoryStore.

        Future Coordinators can query this to avoid known bad configs
        and repeat known good ones.
        """
        from ml_engine.agent.memory import MemoryStore
        from datetime import datetime

        memory = MemoryStore(self._r)
        key = f"{stage}_{self.run_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        content: Dict[str, Any] = {
            "stage": stage,
            "verdict": verdict,
            "metrics": metrics,
            "wall_time_seconds": wall_time,
            "run_id": self.run_id,
        }

        # Include experiment result detail if this was an AutoResearch stage
        exp_result = outcome.get("experiment_result")
        if exp_result:
            content["experiment_result"] = exp_result
            content["note"] = (
                f"AutoResearch: {exp_result.get('trials_completed', 0)} trials, "
                f"best={exp_result.get('best_metric')}"
            )

        try:
            memory.write("feedback", key, content)
            logger.debug("EvaluatorWorker wrote feedback memory: feedback/%s", key)
        except Exception as e:
            logger.warning("EvaluatorWorker: failed to write feedback memory: %s", e)

    def set_contract(self, contract: PipelineContract) -> None:
        """Update the contract. Called when user approves."""
        self._contract = contract
        logger.info("EvaluatorWorker: contract updated for run %s", self.run_id)
