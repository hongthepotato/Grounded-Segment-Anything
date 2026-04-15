"""

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
  - Redis failure: handled by StreamConsumer base (log + 1s backoff + retry)

Usage:
    # Typically launched by Coordinator.run() as a concurrent asyncio task.
    executor = ExecutorWorker(redis_async, run_id, store, contract)
    await executor.run()

Stream-level skeleton (XREADGROUP loop, PEL recovery on restart, always-ACK
poison safety) is inherited from :class:`StreamConsumer`. Each worker below
provides only the event-type filter, the domain handler, and a terminal-state
stop check.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as _aredis

from ml_engine.agent.contracts import PipelineContract, StageSummary
from ml_engine.agent.memory import MemoryStore
from ml_engine.agent.gate import evaluate_gate
from ml_engine.agent.loop import apublish_event
from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine
from ml_engine.agent.stream_consumer import StreamConsumer

logger = logging.getLogger(__name__)


class ExecutorWorker(StreamConsumer):
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
        redis_client: _aredis.Redis,
        run_id: str,
        store,               # AsyncRedisJobStore -- typed loosely to avoid circular import
        contract: Optional[PipelineContract] = None,
    ):
        super().__init__(redis_client, run_id, self.CONSUMER_NAME)
        self._store = store
        self._contract = contract

    # ------------------------------------------------------------------
    # StreamConsumer hooks
    # ------------------------------------------------------------------

    async def should_stop(self) -> bool:
        """Stop when the pipeline run reaches a terminal state."""
        sm = StateMachine(run_id=self.run_id, redis_async=self._r)
        try:
            if await sm.current_state() in TERMINAL_STATES:
                logger.info(
                    "Run %s reached terminal state, ExecutorWorker stopping",
                    self.run_id,
                )
                return True
        except KeyError:
            pass  # State not initialized yet -- keep running
        return False

    async def handle_event(
        self, event: Dict[str, Any], entry_id_str: str
    ) -> None:
        if event.get("type") != "dispatch_requested":
            return
        await self._handle_dispatch(event, entry_id_str)

    # ------------------------------------------------------------------
    # Dispatch handler
    # ------------------------------------------------------------------

    async def _handle_dispatch(self, event: Dict[str, Any], entry_id_str: str) -> None:
        """
        Validate and enqueue a single dispatch_requested event.

        On success: publishes stage_dispatched to agent Stream.
        On failure: publishes dispatch_rejected with reason.

        Idempotency: stores ``entry_id_str`` as ``job.dispatch_event_id`` before
        enqueuing. On PEL re-delivery (same entry_id_str seen again), skips the
        enqueue and re-publishes stage_dispatched so the Coordinator is unblocked.
        This prevents double-execution of the same ML job when the process crashes
        after enqueue but before the stream message is ACKed.
        """
        job_id = event.get("job_id")
        stage = event.get("stage", "unknown")

        if not job_id:
            logger.warning("dispatch_requested missing job_id, skipping")
            return

        logger.info("ExecutorWorker handling dispatch: stage=%s job=%s", stage, job_id[:8])

        # Idempotency check: if this exact stream entry was already processed,
        # re-publish stage_dispatched (in case the Coordinator missed it) and return.
        job = await self._store.get_job(job_id)
        if job is not None and job.dispatch_event_id == entry_id_str:
            logger.info(
                "ExecutorWorker: idempotency hit for job %s (event=%s) -- skipping re-enqueue",
                job_id[:8], entry_id_str,
            )
            await apublish_event(self._r, self.run_id, {
                "type": "stage_dispatched",
                "job_id": job_id,
                "stage": stage,
                "run_id": self.run_id,
            })
            return

        # Validate contract constraints
        errors = await self._validate(stage)
        if errors:
            logger.warning(
                "Dispatch rejected for job %s (stage=%s): %s",
                job_id[:8], stage, "; ".join(errors),
            )
            await apublish_event(self._r, self.run_id, {
                "type": "dispatch_rejected",
                "job_id": job_id,
                "stage": stage,
                "run_id": self.run_id,
                "errors": errors,
            })
            return

        # Stamp the dispatch event-id BEFORE enqueuing so that a crash between
        # this write and the LPUSH is detectable on restart (job has event-id
        # but is not in the queue; the Coordinator will eventually time out and
        # retry via its retry/escalation logic).
        await self._store.update_job(job_id, dispatch_event_id=entry_id_str)

        success = await self._store.enqueue_by_id(job_id)
        if not success:
            logger.error("ExecutorWorker: enqueue_by_id failed for job %s", job_id[:8])
            await apublish_event(self._r, self.run_id, {
                "type": "dispatch_rejected",
                "job_id": job_id,
                "stage": stage,
                "run_id": self.run_id,
                "errors": [f"Job {job_id[:8]} not found in store"],
            })
            return

        await apublish_event(self._r, self.run_id, {
            "type": "stage_dispatched",
            "job_id": job_id,
            "stage": stage,
            "run_id": self.run_id,
        })
        logger.info("ExecutorWorker: enqueued %s (job=%s)", stage, job_id[:8])

    # ------------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------------

    async def _validate(self, stage: str) -> list[str]:
        """
        Return a list of constraint violations. Empty list = dispatch allowed.

        Checks:
        - Retry count vs contract.budget.max_retries
        - (Future: stage ordering against state machine)
        """
        if self._contract is None:
            return []  # Pre-approval / planning phase: no constraints yet

        errors: list[str] = []

        sm = StateMachine(run_id=self.run_id, redis_async=self._r)
        try:
            retry_count = await sm.retry_count()
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
        """
        Update the contract used for dispatch validation.

        Called by Coordinator when a human approves a contract mid-pipeline
        (pending_contract_approval flow, see TODO-10). Without this call, the
        worker would validate against a stale or None contract.
        """
        self._contract = contract
        logger.info("ExecutorWorker: contract updated for run %s", self.run_id)


# ---------------------------------------------------------------------------
# Stage 3: EvaluatorWorker
# ---------------------------------------------------------------------------

class EvaluatorWorker(StreamConsumer):
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
        redis_client: _aredis.Redis,
        run_id: str,
        contract: Optional[PipelineContract] = None,
    ):
        super().__init__(redis_client, run_id, self.CONSUMER_NAME)
        self._contract = contract
        self._memory = MemoryStore(redis_async=redis_client)

    # ------------------------------------------------------------------
    # StreamConsumer hooks
    # ------------------------------------------------------------------

    async def should_stop(self) -> bool:
        """Stop when the pipeline run reaches a terminal state."""
        sm = StateMachine(run_id=self.run_id, redis_async=self._r)
        try:
            if await sm.current_state() in TERMINAL_STATES:
                logger.info(
                    "Run %s reached terminal state, EvaluatorWorker stopping",
                    self.run_id,
                )
                return True
        except KeyError:
            pass
        return False

    async def handle_event(
        self, event: Dict[str, Any], entry_id_str: str
    ) -> None:
        if event.get("type") != "evaluation_requested":
            return
        await self._handle_evaluation(event)

    async def on_event_error(
        self, event: Dict[str, Any], entry_id_str: str, exc: BaseException
    ) -> None:
        """
        Publish an escalation gate_decision so the pipeline doesn't silently stall.

        Only fires when handle_event raised (i.e. a real evaluation failed); the
        base class has already logged the traceback before calling this hook.
        """
        if event.get("type") != "evaluation_requested":
            return
        await apublish_event(self._r, self.run_id, {
            "type": "gate_decision",
            "stage": event.get("stage", "unknown"),
            "verdict": "escalate",
            "reason": f"Evaluator error: {exc}",
            "metrics": {},
            "run_id": self.run_id,
        })

    async def _handle_evaluation(self, event: Dict[str, Any]) -> None:
        """
        Evaluate one completed stage.

        Steps:
        1. Read outcome.json from job output_dir
        2. Run deterministic gate check against contract criteria
        3. Write feedback memory record
        4. Publish gate_decision event for Coordinator
        """
        stage = event.get("stage", "unknown")
        output_dir = event.get("output_dir", "")
        job_id = event.get("job_id", "")

        logger.info("EvaluatorWorker evaluating stage=%s job=%s", stage, job_id[:8] if job_id else "?")

        # Read outcome.json
        outcome_path = Path(output_dir) / "outcome.json"
        if not outcome_path.exists():
            logger.error("EvaluatorWorker: outcome.json not found at %s", output_dir)
            await apublish_event(self._r, self.run_id, {
                "type": "gate_decision",
                "stage": stage,
                "verdict": "escalate",
                "reason": f"outcome.json not found at {output_dir}",
                "metrics": {},
                "run_id": self.run_id,
            })
            return

        # asyncio.to_thread: outcome.json may live on a Docker volume or NFS mount;
        # blocking read_text() would stall the event loop shared by all three workers.
        raw_text = await asyncio.to_thread(outcome_path.read_text, encoding="utf-8")
        outcome = json.loads(raw_text)
        metrics = outcome.get("metrics", {})
        wall_time = outcome.get("wall_time_seconds", 0.0)
        artifacts = outcome.get("artifacts", {})

        # Gate evaluation (deterministic)
        criteria = self._contract.acceptance_criteria if self._contract else None
        sm = StateMachine(run_id=self.run_id, redis_async=self._r)
        retry_count = await sm.retry_count()
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

        # Build structured summary with the real verdict (Coordinator's copy is
        # always "pass" since it fires at job_completed before evaluation)
        exp_result = outcome.get("experiment_result", {})
        summary = StageSummary(
            stage=stage,
            status=decision_verdict,
            metrics=metrics,
            artifacts=artifacts,
            duration_seconds=wall_time,
            trial_count=int(exp_result.get("trials_completed", 0)) or None,
        )

        # Write feedback memory -- future pipelines learn from this
        await self._write_feedback_memory(summary, exp_result)

        # Publish gate_decision for Coordinator to act on; attach summary so
        # any consumer gets structured stage outcome without re-reading outcome.json
        await apublish_event(self._r, self.run_id, {
            "type": "gate_decision",
            "stage": stage,
            "verdict": decision_verdict,
            "reason": decision_reason,
            "metrics": metrics,
            "artifacts": artifacts,
            "wall_time_seconds": wall_time,
            "run_id": self.run_id,
            "stage_summary": summary.to_dict(),
        })

    async def _write_feedback_memory(
        self,
        summary: StageSummary,
        exp_result: Dict[str, Any],
    ) -> None:
        """
        Write structured feedback to MemoryStore.

        Future Coordinators can query this to avoid known bad configs
        and repeat known good ones.
        """
        key = f"{summary.stage}_{self.run_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        content = summary.to_dict()
        content["run_id"] = self.run_id

        # Include experiment result detail if this was an AutoResearch stage
        if exp_result:
            content["experiment_result"] = exp_result
            content["note"] = (
                f"AutoResearch: {exp_result.get('trials_completed', 0)} trials, "
                f"best={exp_result.get('best_metric')}"
            )

        try:
            await self._memory.write("feedback", key, content)
            logger.debug("EvaluatorWorker wrote feedback memory: feedback/%s", key)
        except Exception as e:
            logger.warning("EvaluatorWorker: failed to write feedback memory: %s", e)

    def set_contract(self, contract: PipelineContract) -> None:
        """
        Update the contract used for gate evaluation and dispatch validation.

        Called by Coordinator when a human approves a contract mid-pipeline
        (pending_contract_approval flow, see TODO-10). Also updates self._memory
        is not needed here since MemoryStore is stateless w.r.t. the contract.
        """
        self._contract = contract
        logger.info("EvaluatorWorker: contract updated for run %s", self.run_id)
