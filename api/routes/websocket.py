"""
WebSocket endpoints for job state inspection.

Provides:
- /ws/jobs/{job_id} - Get current job state, then close

Currently a degraded surface: the underlying live-event subscription
mechanism (`AsyncJobManager.subscribe_to_job_async`) was removed in
commit bfdff7f and never replaced. The route is preserved so existing
clients connecting to this URL get a clean state dump + clear error
instead of a 500 from an undefined-method crash. Live tailing during
training requires the new mechanism — tracked in TODOS.md item 15.

Until item 15 ships, clients that need progress updates should poll
GET /api/jobs/{job_id} on a sensible cadence (e.g., every 2-5 s).
"""

import logging
import os

from fastapi import APIRouter, WebSocket

from ml_engine.jobs import get_async_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/jobs/{job_id}")
async def job_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint that returns the current job state and closes.

    Originally designed to forward live Redis pub/sub events for the job's
    lifetime, but the subscription mechanism was removed (commit bfdff7f).
    This degraded version still serves a useful read of current state, so
    callers don't have to reconnect via REST just to see status — but it
    does NOT live-tail. For live progress, poll GET /api/jobs/{job_id}.

    Frames sent (then connection closes):
    - `error`: job not found (close code 4004)
    - `job_state`: current snapshot (status + progress)
    - `job_<terminal-state>`: terminal payload (only if job already done)
    - `subscription_unavailable`: live tailing not implemented (only if
        job is non-terminal — see TODOS.md item 15)

    Example (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/jobs/a1b2c3d4-...');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.type, data);
        };
    """
    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id[:8])

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    manager = get_async_job_manager(redis_url)

    job = await manager.get_job(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": f"Job {job_id} not found"})
        await websocket.close(code=4004)
        return

    # Send initial job state.
    initial_state = {
        "type": "job_state",
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress.to_dict() if job.progress else None,
    }
    await websocket.send_json(initial_state)

    # If already terminal, send the terminal payload and close cleanly.
    if job.is_terminal:
        await websocket.send_json(
            {
                "type": f"job_{job.status.value}",
                "job_id": job_id,
                "output_dir": job.output_dir,
                "error_message": job.error_message,
            }
        )
        await websocket.close()
        return

    # Non-terminal: live tailing isn't implemented (see TODOS.md item 15).
    # Tell the client explicitly so they fall back to polling instead of
    # holding a connection that will never see another frame.
    await websocket.send_json(
        {
            "type": "subscription_unavailable",
            "job_id": job_id,
            "message": (
                "Live event tailing is not implemented in this build. "
                "Poll GET /api/jobs/{job_id} for status updates."
            ),
        }
    )
    await websocket.close()
