"""
Agent (Coordinator) REST + WebSocket endpoints.

POST /api/agent/plan          -- propose a PipelineContract from user intent
POST /api/agent/approve       -- approve a proposed contract to start pipeline
GET  /api/agent/status/{rid}  -- get current pipeline state + stage summaries
POST /api/agent/gate/{rid}/{action} -- human gate: approve or reject pending_approval
WS   /ws/agent/{rid}          -- subscribe to live pipeline events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schemas import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])
ws_router = APIRouter(tags=["agent-ws"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    intent: str
    data_path: str
    image_paths: List[str]
    class_names: List[str]
    output_mode: str = "detection"


class PlanResponse(BaseModel):
    run_id: str
    contract: Dict[str, Any]
    status: str = "pending_contract_approval"
    note: str = "POST /api/agent/approve to start the pipeline"


class ApproveRequest(BaseModel):
    run_id: str
    contract: Dict[str, Any]


class GateActionRequest(BaseModel):
    reason: str = ""


# ---------------------------------------------------------------------------
# Coordinator lifecycle: one asyncio Task per run_id
# ---------------------------------------------------------------------------

# run_id -> asyncio.Task  (module-level so it survives across requests)
_coordinator_tasks: Dict[str, asyncio.Task] = {}


def _start_coordinator(run_id: str, contract_dict: Dict[str, Any]) -> None:
    """
    Start Coordinator.run() as an asyncio background task.

    Idempotent: if a task already exists for run_id and is not done, no-op.
    Called from approve_plan so the Coordinator is running before the first
    event is processed.
    """
    existing = _coordinator_tasks.get(run_id)
    if existing and not existing.done():
        logger.info("Coordinator already running for run %s", run_id)
        return

    from ml_engine.agent.contracts import PipelineContract
    from ml_engine.agent.coordinator import Coordinator
    from ml_engine.agent.llm_client import LLMClient
    from ml_engine.agent.state_machine import StateMachine

    r = _get_async_redis()
    contract = PipelineContract.from_dict(contract_dict)

    # Build LLMClient from env so the Coordinator can target alternative
    # providers (e.g. DeepSeek via OpenAI-compat) without code changes.
    llm_kwargs: Dict[str, Any] = {
        "provider": os.environ.get("LLM_PROVIDER", "anthropic"),
        "model": os.environ.get("LLM_MODEL") or None,
        "base_url": os.environ.get("LLM_BASE_URL") or None,
        "api_key_env": os.environ.get("LLM_API_KEY_ENV") or None,
    }
    _timeout_env = os.environ.get("LLM_TIMEOUT")
    if _timeout_env:
        llm_kwargs["timeout"] = float(_timeout_env)
    llm = LLMClient(**llm_kwargs)

    coordinator = Coordinator(redis_client=r, run_id=run_id, contract=contract, llm_client=llm)

    task = asyncio.create_task(
        coordinator.run(),
        name=f"coordinator-{run_id[:8]}",
    )

    def _on_done(t: asyncio.Task) -> None:
        _coordinator_tasks.pop(run_id, None)
        if t.cancelled():
            logger.info("Coordinator task for run %s was cancelled", run_id)
            return
        exc = t.exception()
        if exc:
            logger.error("Coordinator task for run %s failed: %s", run_id, exc)

            async def _mark_failed() -> None:
                try:
                    await StateMachine(run_id=run_id, redis_async=r).transition(
                        "failed_unrecoverable", error_message=str(exc)
                    )
                except Exception as mark_err:
                    logger.error(
                        "Failed to mark run %s as failed_unrecoverable: %s",
                        run_id,
                        mark_err,
                    )

            asyncio.create_task(_mark_failed(), name=f"coordinator-crash-{run_id[:8]}")
        else:
            logger.info("Coordinator task for run %s completed", run_id)

    task.add_done_callback(_on_done)
    _coordinator_tasks[run_id] = task
    logger.info("Coordinator task started for run %s", run_id)


# ---------------------------------------------------------------------------
# Orphan recovery: re-launch Coordinator tasks for non-terminal runs on startup
# ---------------------------------------------------------------------------


async def resume_orphaned_coordinators() -> None:
    """
    Scan Redis for runs in a non-terminal state with no active Coordinator task
    and re-launch each one.

    Called once from the FastAPI lifespan startup hook. Safe to call again
    at any point -- _start_coordinator is idempotent (no-ops if task running).
    """
    from ml_engine.agent.state_machine import StateMachine

    r = _get_async_redis()
    run_ids = await StateMachine.scan_non_terminal_run_ids(r)

    resumed = 0
    for run_id in run_ids:
        if run_id in _coordinator_tasks and not _coordinator_tasks[run_id].done():
            continue  # already running (e.g. called twice at startup)

        sm = StateMachine(run_id=run_id, redis_async=r)
        contract_dict = await sm.get_approved_contract()
        if contract_dict is None:
            # Run was created and planned but never approved -- no Coordinator needed yet.
            logger.info("Skipping orphaned run %s: no approved contract (pre-approve state)", run_id)
            continue

        logger.info("Auto-resuming Coordinator for orphaned run %s", run_id)
        _start_coordinator(run_id, contract_dict)
        resumed += 1

    logger.info("Startup orphan recovery: resumed %d Coordinator(s)", resumed)


# ---------------------------------------------------------------------------
# Dependency: redis client
# ---------------------------------------------------------------------------


def _get_async_redis():
    from ml_engine.agent.redis_clients import get_async_redis_client

    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return get_async_redis_client(url)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/plan", response_model=PlanResponse)
async def propose_plan(body: PlanRequest):
    """
    Propose a PipelineContract from user intent.

    Does NOT start training. Returns a contract for human review.
    Call POST /api/agent/approve to kick off the pipeline.
    """
    from ml_engine.agent.contracts import (
        AcceptanceCriteria,
        BudgetSpec,
        DataSpec,
        LineageSpec,
        PipelineContract,
        TargetSpec,
    )
    from ml_engine.agent.loop import apublish_event
    from ml_engine.agent.state_machine import StateMachine

    run_id = str(uuid.uuid4())
    contract = PipelineContract(
        id=run_id,
        target=TargetSpec(
            class_names=body.class_names,
            output_mode=body.output_mode,
            description=body.intent,
        ),
        data=DataSpec(
            data_path=body.data_path,
            image_paths=body.image_paths,
        ),
        acceptance_criteria=AcceptanceCriteria(),
        budget=BudgetSpec(),
        lineage=LineageSpec(),
    )

    r = _get_async_redis()
    sm = StateMachine(run_id=run_id, redis_async=r)
    # Persist the contract so GET /status can return it without relying on
    # the client caching the HTTP response from this endpoint.
    await sm.initialize(contract=contract.to_dict())

    await apublish_event(
        r,
        run_id,
        {
            "type": "plan_proposed",
            "run_id": run_id,
            "contract": contract.to_dict(),
        },
    )

    logger.info("Plan proposed: run_id=%s intent=%r", run_id, body.intent[:60])
    return JSONResponse(
        status_code=200,
        content=success_response(
            data=PlanResponse(
                run_id=run_id,
                contract=contract.to_dict(),
            ).model_dump()
        ),
    )


# TODO: /approve and /gate have no authentication — any caller can approve a
# contract or make a gate decision for any run_id. Add an API-key guard or
# integrate with the project's auth layer before exposing this service publicly.
@router.post("/approve")
async def approve_plan(body: ApproveRequest):
    """
    Approve a proposed contract and start the Coordinator pipeline. Idempotent.

    First call (state=created): transitions created -> planning, publishes
    contract_approved event, persists the approved contract, starts Coordinator.

    Subsequent calls (state already past created, not terminal): skips the
    transition and event publish (Coordinator resumes from stream PEL/cursor),
    updates the persisted contract, and re-launches the Coordinator task if it
    is not already running. Returns 409 if the run is in a terminal state.
    """
    from ml_engine.agent.loop import apublish_event
    from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine

    r = _get_async_redis()
    sm = StateMachine(run_id=body.run_id, redis_async=r)

    try:
        current = await sm.current_state()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {body.run_id} not found")

    if current in TERMINAL_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Run {body.run_id!r} is in terminal state {current!r} and cannot be re-approved",
        )

    # Persist the approved contract BEFORE any state transition so that orphan
    # recovery can always reconstruct the Coordinator — even if the process dies
    # between the transition write and the end of this handler.
    await sm.store_approved_contract(body.contract)

    if current == "created":
        # First approval: transition, publish the trigger event, record contract.
        try:
            await sm.transition("planning")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        await apublish_event(
            r,
            body.run_id,
            {
                "type": "contract_approved",
                "run_id": body.run_id,
                "contract": body.contract,
            },
        )
    else:
        # Idempotent re-approve: Coordinator resumes from stream PEL/cursor —
        # no need to re-publish contract_approved (already ACKed or in PEL).
        logger.info("Idempotent re-approve for run %s (current state: %s)", body.run_id, current)

    # Start or re-launch the Coordinator task (no-op if already running).
    _start_coordinator(body.run_id, body.contract)

    logger.info("Plan approved and Coordinator started: run_id=%s", body.run_id)
    return JSONResponse(
        status_code=200,
        content=success_response(
            data={"run_id": body.run_id, "status": current if current != "created" else "planning"}
        ),
    )


@router.get("/status/{run_id}")
async def get_status(run_id: str):
    """
    Get current pipeline state, stage summaries, and proposed contract for a run.

    Returns proposed_contract so the frontend can render the contract review UI
    even after a page refresh (contract is persisted in Redis at plan time).
    """
    from ml_engine.agent.state_machine import StateMachine

    r = _get_async_redis()
    sm = StateMachine(run_id=run_id, redis_async=r)

    try:
        data = await sm.load()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    state = data["state"]
    summaries = json.loads(data.get("stage_summaries", "[]"))
    proposed_raw = data.get("proposed_contract", "")
    proposed_contract = json.loads(proposed_raw) if proposed_raw else None
    proposed_contract = proposed_contract if proposed_contract else None
    retry_count = int(data.get("retry_count", "0"))

    return JSONResponse(
        status_code=200,
        content=success_response(
            data={
                "run_id": run_id,
                "state": state,
                "retry_count": retry_count,
                "stage_summaries": summaries,
                "proposed_contract": proposed_contract,
                "coordinator_active": run_id in _coordinator_tasks and not _coordinator_tasks[run_id].done(),
            }
        ),
    )


@router.post("/gate/{run_id}/{action}")
async def human_gate(run_id: str, action: str, body: GateActionRequest):
    """
    Human gate decision for any gate state.

    action: "approve" | "reject"

    pending_approval (end-of-pipeline gate):
        approve -> "done"       (event: gate_approved)
        reject  -> "cancelled"  (event: gate_rejected)

    pending_contract_approval (start-of-pipeline gate):
        approve -> "auto_labeling"  (event: contract_approved)
        reject  -> "cancelled"      (event: contract_rejected)
    """
    from ml_engine.agent.loop import apublish_event
    from ml_engine.agent.state_machine import StateMachine

    if action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    r = _get_async_redis()
    sm = StateMachine(run_id=run_id, redis_async=r)

    try:
        current = await sm.current_state()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if current == "pending_approval":
        target_state = "done" if action == "approve" else "cancelled"
        event_type = "gate_approved" if action == "approve" else "gate_rejected"
    elif current == "pending_contract_approval":
        target_state = "auto_labeling" if action == "approve" else "cancelled"
        event_type = "contract_approved" if action == "approve" else "contract_rejected"
    else:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run is in state {current!r}, not a gate state "
                "(pending_approval or pending_contract_approval)"
            ),
        )

    try:
        await sm.transition(target_state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await apublish_event(
        r,
        run_id,
        {
            "type": event_type,
            "run_id": run_id,
            "action": action,
            "reason": body.reason,
            "new_state": target_state,
        },
    )

    logger.info("Human gate %s for run %s -> %s", action, run_id, target_state)
    return JSONResponse(
        status_code=200,
        content=success_response(data={"run_id": run_id, "action": action, "new_state": target_state}),
    )


# ---------------------------------------------------------------------------
# WebSocket: live event stream
# ---------------------------------------------------------------------------


@ws_router.websocket("/ws/agent/{run_id}")
async def agent_websocket(websocket: WebSocket, run_id: str):
    """
    Stream live pipeline events for a run via WebSocket.

    Client receives JSON-encoded events from the Redis Stream.
    Connection closes when pipeline reaches a terminal state.
    """
    import redis

    await websocket.accept()

    r = _get_async_redis()

    from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine
    from ml_engine.agent.stream_consumer import stream_key

    key = stream_key(run_id)
    last_id = "0-0"

    try:
        while True:
            try:
                entries = await r.xread({key: last_id}, count=10, block=1000)
            except redis.RedisError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                break

            if entries:
                for _key, messages in entries:
                    for entry_id, data in messages:
                        last_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
                        raw = data.get(b"data", data.get("data", "{}"))
                        if isinstance(raw, bytes):
                            raw = raw.decode()
                        event = json.loads(raw)
                        await websocket.send_json(event)

            # Check for terminal state
            try:
                sm = StateMachine(run_id=run_id, redis_async=r)
                if await sm.current_state() in TERMINAL_STATES:
                    await websocket.send_json({"type": "pipeline_done", "run_id": run_id})
                    break
            except KeyError:
                pass

            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for run %s", run_id)
