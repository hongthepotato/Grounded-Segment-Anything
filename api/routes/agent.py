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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

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

    from ml_engine.agent.coordinator import Coordinator
    from ml_engine.agent.contracts import PipelineContract

    r = _get_async_redis()
    contract = PipelineContract.from_dict(contract_dict)
    coordinator = Coordinator(redis_client=r, run_id=run_id, contract=contract)

    task = asyncio.create_task(
        coordinator.run(),
        name=f"coordinator-{run_id[:8]}",
    )

    def _on_done(t: asyncio.Task) -> None:
        exc = t.exception()
        if exc:
            logger.error("Coordinator task for run %s failed: %s", run_id, exc)
        else:
            logger.info("Coordinator task for run %s completed", run_id)
        _coordinator_tasks.pop(run_id, None)

    task.add_done_callback(_on_done)
    _coordinator_tasks[run_id] = task
    logger.info("Coordinator task started for run %s", run_id)


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
    from ml_engine.agent.state_machine import StateMachine
    from ml_engine.agent.loop import apublish_event

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

    await apublish_event(r, run_id, {
        "type": "plan_proposed",
        "run_id": run_id,
        "contract": contract.to_dict(),
    })

    logger.info("Plan proposed: run_id=%s intent=%r", run_id, body.intent[:60])
    return PlanResponse(
        run_id=run_id,
        contract=contract.to_dict(),
    )


@router.post("/approve")
async def approve_plan(body: ApproveRequest):
    """
    Approve a proposed contract and start the Coordinator pipeline.

    Transitions state created -> planning, publishes contract_approved event,
    and starts the Coordinator as an asyncio background task. The Coordinator
    reads the event from the Redis Stream and begins orchestrating stages.

    The contract in the request body may differ from the proposed contract --
    the user can modify budget, acceptance_criteria, or stage_configs before
    approving.
    """
    from ml_engine.agent.state_machine import StateMachine
    from ml_engine.agent.loop import apublish_event

    r = _get_async_redis()
    sm = StateMachine(run_id=body.run_id, redis_async=r)

    try:
        await sm.transition("planning")
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    await apublish_event(r, body.run_id, {
        "type": "contract_approved",
        "run_id": body.run_id,
        "contract": body.contract,
    })

    # Start the Coordinator task (idempotent -- no-op if already running)
    _start_coordinator(body.run_id, body.contract)

    logger.info("Plan approved and Coordinator started: run_id=%s", body.run_id)
    return {"run_id": body.run_id, "status": "planning"}


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
        state = await sm.current_state()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    summaries = await sm.get_stage_summaries()
    proposed_contract = await sm.get_proposed_contract()
    retry_count = await sm.retry_count()

    return {
        "run_id": run_id,
        "state": state,
        "retry_count": retry_count,
        "stage_summaries": summaries,
        "proposed_contract": proposed_contract,  # present when state == "created"
        "coordinator_active": run_id in _coordinator_tasks and not _coordinator_tasks[run_id].done(),
    }


@router.post("/gate/{run_id}/{action}")
async def human_gate(run_id: str, action: str, body: GateActionRequest):
    """
    Human gate decision for pending_approval state.

    action: "approve" | "reject"

    approve -> transitions to "done"
    reject  -> transitions to "escalated" with reason
    """
    from ml_engine.agent.state_machine import StateMachine
    from ml_engine.agent.loop import apublish_event

    if action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    r = _get_async_redis()
    sm = StateMachine(run_id=run_id, redis_async=r)

    try:
        current = await sm.current_state()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if current != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run is in state {current!r}, not pending_approval",
        )

    target_state = "done" if action == "approve" else "escalated"
    try:
        await sm.transition(target_state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await apublish_event(r, run_id, {
        "type": "gate_approved" if action == "approve" else "gate_rejected",
        "run_id": run_id,
        "action": action,
        "reason": body.reason,
        "new_state": target_state,
    })

    logger.info("Human gate %s for run %s -> %s", action, run_id, target_state)
    return {"run_id": run_id, "action": action, "new_state": target_state}


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

    from ml_engine.agent.loop import stream_key
    from ml_engine.agent.state_machine import TERMINAL_STATES, StateMachine

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
    finally:
        await r.aclose()
