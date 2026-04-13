"""
Coordinator agent -- Stage 1.

Event-driven. Sleeps between events via Redis Streams.
Wakes on contract_approved, job_completed, gate_decision, job_failed.
Makes 1-3 LLM calls per event, then persists state and sleeps.

Six tools (stripped from Claude Code's 7):
  propose_plan        -- generate PipelineContract from intent + memory
  inspect_status      -- read pipeline state, job status, metrics
  read_memory         -- load memory records
  dispatch_stage      -- launch Executor for a stage
  request_evaluation  -- launch Evaluator for completed stage
  advance_gate        -- advance state machine (auto or human-gated)

LLM unavailability: 30s asyncio.TimeoutError -> fall back to
  SimpleMutator (HPO) or publish llm_unavailable event (planning).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import redis as _redis
from pydantic import BaseModel

from ml_engine.agent.contracts import (
    AcceptanceCriteria,
    BudgetSpec,
    DataSpec,
    LineageSpec,
    PipelineContract,
    StageSummary,
    TargetSpec,
)
from ml_engine.agent.loop import AgentLoop, LoopState, publish_event
from ml_engine.agent.skills import SkillLoader
from ml_engine.agent.llm_client import LLMClient
from ml_engine.agent.memory import MemoryStore
from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine
from ml_engine.agent.tools import RunContext, Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Stages that need a human gate before proceeding
HUMAN_GATE_STAGES = {"pending_contract_approval", "pending_approval"}

# Map: event type -> next state (deterministic, no LLM needed)
_AUTO_TRANSITIONS = {
    "job_completed": None,      # needs Evaluator to decide
    "gate_approved": None,      # needs Coordinator to decide next stage
    "job_failed": "failed_retrying",
    "llm_unavailable": None,    # publish escalation event
}

# Map pipeline state -> skill file name for per-stage LLM prompts.
# SkillLoader reads configs/agent/skills/{name}.md.
_STATE_TO_SKILL = {
    "auto_labeling":        "auto_label",
    "label_review_gate":    "auto_label",
    "teacher_training":     "teacher_training",
    "training_eval_gate":   "teacher_training",
    "student_distillation": "student_distillation",
    "distill_eval_gate":    "student_distillation",
    "pending_approval":     "student_distillation",
}

# Normalize stage names from events to skill file names.
_STAGE_TO_SKILL = {
    "auto_labeling":        "auto_label",
    "auto_label":           "auto_label",
    "teacher_training":     "teacher_training",
    "student_distillation": "student_distillation",
    "experiment_loop":      "teacher_training",
}

_SYSTEM_PROMPT = """\
You are the Coordinator for an agentic ML training pipeline.

Events you will receive (in order):
  contract_approved  -- human approved the contract, you start the pipeline
  job_completed      -- a training/labeling job finished; check outcome and request evaluation
  gate_decision      -- EvaluatorWorker produced pass/retry/escalate verdict
  job_failed         -- a job crashed; handled deterministically (no LLM needed)

Your job per event:
1. Read the event and current pipeline state.
2. Decide what to do next: dispatch a stage, request evaluation, advance gate, or escalate.
3. Use your tools. Make at most 3 tool calls per event.
4. Be decisive. If the path is clear, take it. If budget is exhausted or the
   situation needs human judgment, escalate.

On contract_approved:
  - Read memory for any prior feedback on this dataset.
  - Dispatch the first stage (auto_labeling or teacher_training depending on data annotation status).

On job_completed:
  - Call request_evaluation to trigger the EvaluatorWorker.
  - Do NOT call advance_gate yet -- wait for gate_decision.

On gate_decision:
  - verdict=pass: call advance_gate to proceed to next stage or pending_approval.
  - verdict=retry: call dispatch_stage to re-run the same stage.
  - verdict=escalate: call advance_gate to move to "escalated" state.

Constraints:
- Never dispatch a stage without an approved PipelineContract.
- Never transition to done without going through pending_approval (human gate).
- Never retry a stage more than contract.budget.max_retries times.
- All decisions are bounded by the contract budget and acceptance_criteria.
"""


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------

class ProposePlanArgs(BaseModel):
    r"""Input contract for propose_plan tool."""
    intent: str
    data_path: str
    image_paths: List[str]
    class_names: List[str]
    output_mode: str = "detection"


class InspectStatusArgs(BaseModel):
    r"""Input contract for inspect_status tool."""
    run_id: str


class ReadMemoryArgs(BaseModel):
    r"""Input contract for read_memory tool."""
    types: List[str] = ["feedback", "project"]
    query: Optional[str] = None


class DispatchStageArgs(BaseModel):
    r"""Input contract for dispatch_stage tool."""
    stage: str              # "auto_labeling" | "teacher_training" | "student_distillation"
    overrides: Dict[str, Any] = {}


class RequestEvaluationArgs(BaseModel):
    r"""Input contract for request_evaluation tool."""
    stage: str
    job_id: str
    output_dir: str


class AdvanceGateArgs(BaseModel):
    r"""Input contract for advance_gate tool."""
    target_state: str
    reason: str


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ProposePlanTool(Tool[ProposePlanArgs]):
    r"""Generate a PipelineContract from user intent and memory context."""
    name = "propose_plan"
    description = "Generate a PipelineContract from user intent and memory context."
    input_schema = ProposePlanArgs

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    async def execute(self, args: ProposePlanArgs, context: RunContext) -> ToolResult:
        _ = context  # not used, but could be for more advanced memory retrieval in the future
        memory_ctx = self._memory.to_llm_context(["project", "feedback"])
        contract = PipelineContract(
            id=str(uuid.uuid4()),
            target=TargetSpec(
                class_names=args.class_names,
                output_mode=args.output_mode,
                description=args.intent,
            ),
            data=DataSpec(
                data_path=args.data_path,
                image_paths=args.image_paths,
            ),
            acceptance_criteria=AcceptanceCriteria(),
            budget=BudgetSpec(),
            lineage=LineageSpec(),
        )
        return ToolResult(success=True, output={
            "contract": contract.to_dict(),
            "memory_applied": bool(memory_ctx),
            "note": "Contract pending human approval at POST /api/agent/approve",
        })


class InspectStatusTool(Tool[InspectStatusArgs]):
    r"""Read pipeline state, job status, and latest metrics."""
    name = "inspect_status"
    description = "Read pipeline state, job status, and latest metrics."
    input_schema = InspectStatusArgs

    def __init__(self, redis_client: _redis.Redis):
        self._r = redis_client

    async def execute(self, args: InspectStatusArgs, context: RunContext) -> ToolResult:
        _ = context  # not used, but could be for more advanced status retrieval in the future
        sm = StateMachine(self._r, args.run_id)
        try:
            state = sm.load()
            summaries = sm.get_stage_summaries()
            return ToolResult(success=True, output={
                "state": state,
                "stage_summaries": summaries,
            })
        except KeyError:
            return ToolResult(success=False, error=f"No state for run {args.run_id}")


class ReadMemoryTool(Tool[ReadMemoryArgs]):
    r"""Load memory records (feedback, project, user, reference)."""
    name = "read_memory"
    description = "Load memory records (feedback, project, user, reference)."
    input_schema = ReadMemoryArgs

    def __init__(self, memory: MemoryStore):
        self._memory = memory

    async def execute(self, args: ReadMemoryArgs, context: RunContext) -> ToolResult:
        _ = context  # not used, but could be for more advanced memory retrieval in the future
        records: Dict[str, Any] = {}
        for t in args.types:
            records[t] = self._memory.read(t)
        return ToolResult(success=True, output={"memory": records})


class DispatchStageTool(Tool[DispatchStageArgs]):
    r"""Submit a job for the next pipeline stage via JobManager."""
    name = "dispatch_stage"
    description = "Submit a job for the next pipeline stage via JobManager."
    input_schema = DispatchStageArgs

    def __init__(self, redis_client: _redis.Redis, contract: Optional[PipelineContract]):
        self._r = redis_client
        self._contract = contract

    async def execute(self, args: DispatchStageArgs, context: RunContext) -> ToolResult:
        from ml_engine.jobs import get_job_manager
        from ml_engine.jobs.models import Job

        stage_to_job_type = {
            "auto_labeling": "auto_label",
            "teacher_training": "teacher_training",
            "student_distillation": "student_distillation",
            "experiment_loop": "experiment_loop",
        }
        job_type = stage_to_job_type.get(args.stage)
        if not job_type:
            return ToolResult(success=False, error=f"Unknown stage: {args.stage!r}")

        if self._contract is None:
            return ToolResult(success=False, error="No contract -- cannot dispatch without an approved contract")

        job_config: Dict[str, Any] = {
            "data_path": self._contract.data.data_path,
            "image_paths": self._contract.data.image_paths,
            "split_config": self._contract.data.split_config,
        }
        job_config.update(args.overrides)

        manager = get_job_manager(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        job = Job(
            type=job_type,
            run_id=context.run_id,
            config=job_config,
        )

        # Stage 2: store the job WITHOUT queuing it.
        # ExecutorWorker (consumer group "executor") validates contract constraints
        # then calls store.enqueue_by_id(job_id) to move it into the work queue.
        manager.store.store_job(job)

        publish_event(self._r, context.run_id, {
            "type": "dispatch_requested",
            "stage": args.stage,
            "job_id": job.id,
            "job_type": job_type,
            "run_id": context.run_id,
        })

        logger.info(
            "Dispatch requested for %s (job_id=%s, run_id=%s)",
            args.stage, job.id[:8], context.run_id,
        )
        return ToolResult(success=True, output={
            "job_id": job.id,
            "stage": args.stage,
            "status": "dispatch_requested",
            "note": "ExecutorWorker will validate and enqueue",
        })


class RequestEvaluationTool(Tool[RequestEvaluationArgs]):
    r"""Request evaluation of a completed stage by the EvaluatorWorker."""
    name = "request_evaluation"
    description = (
        "Request metric-based evaluation of a completed stage. "
        "The EvaluatorWorker will read outcome.json, run the gate check, "
        "write feedback memory, and publish a gate_decision event."
    )
    input_schema = RequestEvaluationArgs

    def __init__(self, redis_client: _redis.Redis, contract: Optional[PipelineContract]):
        self._r = redis_client
        self._contract = contract

    async def execute(self, args: RequestEvaluationArgs, context: RunContext) -> ToolResult:
        """
        Publish evaluation_requested event.

        EvaluatorWorker (Stage 3) handles the actual metric check and
        publishes gate_decision back to the Stream. The Coordinator then
        picks up gate_decision on its next event turn.
        """
        publish_event(self._r, context.run_id, {
            "type": "evaluation_requested",
            "stage": args.stage,
            "job_id": args.job_id,
            "output_dir": args.output_dir,
            "run_id": context.run_id,
        })

        logger.info(
            "Evaluation requested for stage=%s job=%s",
            args.stage, args.job_id[:8] if args.job_id else "?",
        )
        return ToolResult(success=True, output={
            "status": "evaluation_requested",
            "stage": args.stage,
            "note": "EvaluatorWorker will publish gate_decision",
        })


class AdvanceGateTool(Tool[AdvanceGateArgs]):
    r"""Advance the pipeline state machine to the next state."""
    name = "advance_gate"
    description = "Advance the pipeline state machine to the next state."
    input_schema = AdvanceGateArgs

    def __init__(self, redis_client: _redis.Redis):
        self._r = redis_client

    async def execute(self, args: AdvanceGateArgs, context: RunContext) -> ToolResult:
        sm = StateMachine(self._r, context.run_id)
        try:
            sm.transition(args.target_state)
            publish_event(self._r, context.run_id, {
                "type": "state_changed",
                "new_state": args.target_state,
                "reason": args.reason,
                "run_id": context.run_id,
            })
            return ToolResult(success=True, output={"new_state": args.target_state})
        except ValueError as e:
            return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """
    Event-driven Coordinator for Stage 1.

    Instantiate once per pipeline run. Call run() to start the event loop.
    """

    def __init__(
        self,
        redis_client: _redis.Redis,
        run_id: str,
        llm_client: Optional[LLMClient] = None,
        contract: Optional[PipelineContract] = None,
    ):
        self._r = redis_client
        self.run_id = run_id
        self._llm = llm_client or LLMClient()
        self._contract = contract
        self._memory = MemoryStore(redis_client)
        self._skills = SkillLoader()
        self._tools = self._build_registry()

    async def on_event(self, event: Dict[str, Any], state: LoopState) -> None:
        """
        Process one pipeline event. Called by AgentLoop per event.

        Makes at most MAX_TURNS_PER_EVENT LLM calls.
        Falls back gracefully on LLM timeout.
        """
        from ml_engine.agent.loop import MAX_TURNS_PER_EVENT
        from ml_engine.agent.compact import compact_stage

        event_type = event.get("type", "unknown")
        sm = StateMachine(self._r, self.run_id)

        # Check for terminal state -- nothing to do
        try:
            current = sm.current_state
        except KeyError:
            logger.warning("No state found for run %s, skipping event", self.run_id)
            return

        if current in TERMINAL_STATES:
            logger.info("Run %s is in terminal state %s, ignoring event", self.run_id, current)
            return

        # Handle job_failed deterministically
        if event_type == "job_failed":
            retry_count = sm.retry_count
            max_retries = self._contract.budget.max_retries if self._contract else 2
            if retry_count < max_retries:
                sm.transition("failed_retrying", error_message=event.get("error"))
            else:
                sm.transition("failed_unrecoverable", error_message="max retries exhausted")
            return

        context = RunContext(
            run_id=self.run_id,
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            contract=self._contract.to_dict() if self._contract else None,
        )

        # Stage-boundary compaction on job_completed
        if event_type == "job_completed" and state.stage_just_completed:
            outcome_metrics = event.get("outcome", {}).get("metrics", {})
            key_decisions = [
                f"{k}={v}" for k, v in state.stage_dispatch_overrides.items()
            ] if state.stage_dispatch_overrides else []
            summary = StageSummary(
                stage=state.stage_just_completed,
                status="pass",
                metrics=outcome_metrics,
                artifacts=event.get("outcome", {}).get("artifacts", {}),
                duration_seconds=event.get("outcome", {}).get("wall_time_seconds", 0.0),
                trial_count=int(outcome_metrics.get("trials_completed", 0)) or None,
                key_decisions=key_decisions,
            )
            state.messages = compact_stage(state.messages, summary, state.stage_start_idx)
            sm.append_stage_summary(summary.to_dict())
            state.stage_just_completed = None
            state.stage_dispatch_overrides = {}
            state.stage_start_idx = None

        system_prompt = self._build_system_prompt(event, current)

        response = None
        # LLM-driven turns
        for turn in range(MAX_TURNS_PER_EVENT):
            try:
                response = await self._llm.call(
                    system=system_prompt,
                    messages=state.messages,
                    tools=self._tools.all_schemas(),
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timeout on run %s (turn %d), publishing llm_unavailable", self.run_id, turn)
                publish_event(self._r, self.run_id, {
                    "type": "llm_unavailable",
                    "run_id": self.run_id,
                    "event_type": event_type,
                })
                return
            except Exception as e:
                logger.error("LLM error: %s", e)
                return

            # Append assistant response to history
            state.messages.append({"role": "assistant", "content": response["content"]})

            # Execute tool calls
            tool_results = []
            tool_calls = [b for b in response.get("content", []) if b.get("type") == "tool_use"]
            if not tool_calls:
                break  # LLM is done for this event

            for tc in tool_calls:
                tool_name = tc.get("name")
                tool_input = tc.get("input", {})
                try:
                    tool = self._tools.get(tool_name)
                    input_model = tool.input_schema(**tool_input)
                    errors = tool.validate(input_model)
                    if errors:
                        result = ToolResult(success=False, error="; ".join(errors))
                    else:
                        result = await tool.execute(input_model, context)
                except Exception as e:
                    result = ToolResult(success=False, error=str(e))
                    logger.error("Tool %s error: %s", tool_name, e)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id"),
                    "content": json.dumps(result.model_dump()),
                })
                logger.debug("Tool %s -> success=%s", tool_name, result.success)

                # Track which stage was dispatched so compaction fires on job_completed.
                # Record the exact message index now — len(messages)-1 is the assistant
                # message containing this dispatch_stage call.
                if tool_name == "dispatch_stage" and result.success:
                    state.stage_just_completed = tool_input.get("stage")
                    state.stage_dispatch_overrides = tool_input.get("overrides", {})
                    state.stage_start_idx = len(state.messages) - 1

            state.messages.append({"role": "user", "content": tool_results})

        # Check stop reason
        if not response:
            logger.warning("No response from LLM for run %s on event %s", self.run_id, event_type)
        if response.get("stop_reason") == "end_turn":
            logger.info("Coordinator done for event %s (run %s)", event_type, self.run_id)

    async def run(
        self,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Start the Coordinator, ExecutorWorker, and EvaluatorWorker concurrently.

        Three consumer groups share the same agent Stream, independent cursors:
        - "coordinator": processes job_completed, gate_decision, gate_* events
        - "executor": processes dispatch_requested events
        - "evaluator": processes evaluation_requested events

        Blocks until the pipeline reaches a terminal state or cancel_check() returns True.
        """
        from ml_engine.jobs import get_job_manager
        from ml_engine.agent.workers import EvaluatorWorker, ExecutorWorker

        store = get_job_manager(os.environ.get("REDIS_URL", "redis://localhost:6379")).store
        executor = ExecutorWorker(
            redis_client=self._r,
            run_id=self.run_id,
            store=store,
            contract=self._contract,
        )
        evaluator = EvaluatorWorker(
            redis_client=self._r,
            run_id=self.run_id,
            contract=self._contract,
        )

        coord_loop = AgentLoop(
            redis_client=self._r,
            run_id=self.run_id,
            on_event=self.on_event,
            consumer_name="coordinator-0",
        )

        await asyncio.gather(
            coord_loop.run(cancel_check=cancel_check),
            executor.run(cancel_check=cancel_check),
            evaluator.run(cancel_check=cancel_check),
        )

    def _build_system_prompt(self, event: Dict[str, Any], current_state: str) -> str:
        """Build system prompt with per-stage skill injected."""
        # Resolve skill name: prefer explicit stage from event, fall back to state mapping
        skill_name = None
        stage = event.get("stage")
        if stage:
            skill_name = _STAGE_TO_SKILL.get(stage)
        if skill_name is None:
            skill_name = _STATE_TO_SKILL.get(current_state)

        if skill_name is None:
            return _SYSTEM_PROMPT

        try:
            skill = self._skills.load(skill_name)
            return _SYSTEM_PROMPT + "\n\n" + skill.to_system_prompt()
        except FileNotFoundError:
            logger.debug("No skill file for %s, using base prompt", skill_name)
            return _SYSTEM_PROMPT

    def _build_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(ProposePlanTool(self._memory))
        reg.register(InspectStatusTool(self._r))
        reg.register(ReadMemoryTool(self._memory))
        reg.register(DispatchStageTool(self._r, self._contract))
        reg.register(RequestEvaluationTool(self._r, self._contract))
        reg.register(AdvanceGateTool(self._r))
        return reg
