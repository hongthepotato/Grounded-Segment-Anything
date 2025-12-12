"""
WebSocket endpoints for real-time job updates.

Provides:
- /ws/jobs/{job_id} - Subscribe to job events

Events sent to client:
- job_started: Job began execution
- progress: Training progress update
- job_completed: Job finished successfully
- job_failed: Job failed with error
- job_cancelled: Job was cancelled
- cancel_requested: Cancellation was requested
"""

import asyncio
import logging
import os
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ml_engine.jobs import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/jobs/{job_id}")
async def job_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.
    
    Connects to Redis pub/sub and forwards events to the client.
    Automatically closes when job reaches terminal state.
    
    Example (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/jobs/a1b2c3d4-...');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.type, data);
        };
    """
    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id[:8])

    # Get manager
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    manager = get_job_manager(redis_url)

    # Check if job exists
    job = manager.get_job(job_id)
    if job is None:
        await websocket.send_json({
            "type": "error",
            "message": f"Job {job_id} not found"
        })
        await websocket.close(code=4004)
        return

    # Send initial job state
    initial_state = {
        "type": "job_state",
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress.to_dict() if job.progress else None,
    }
    await websocket.send_json(initial_state)

    # If job is already terminal, send final state and close
    if job.is_terminal:
        await websocket.send_json({
            "type": f"job_{job.status.value}",
            "job_id": job_id,
            "output_dir": job.output_dir,
            "error_message": job.error_message,
        })
        await websocket.close()
        return

    # Event queue for async handling
    event_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def on_event(event: dict):
        """Callback from Redis pub/sub (runs in background thread)."""
        try:
            # Put event in async queue
            asyncio.run_coroutine_threadsafe(
                event_queue.put(event),
                asyncio.get_event_loop()
            )
        except Exception as e:
            logger.warning("Error queuing event: %s", e)

    # Start subscription in background thread
    sub_thread = manager.subscribe_to_job_async(job_id, on_event)

    try:
        while True:
            # Check for events with timeout
            try:
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=1.0
                )

                # Forward event to client
                await websocket.send_json(event)

                # Check for terminal events
                if event.get("type") in ["job_completed", "job_failed", "job_cancelled"]:
                    logger.info("Job %s reached terminal state: %s",
                              job_id[:8], event.get("type"))
                    break

            except asyncio.TimeoutError:
                # Periodic check: is job still active?
                job = manager.get_job(job_id)
                if job and job.is_terminal:
                    # Job finished but we missed the event
                    await websocket.send_json({
                        "type": f"job_{job.status.value}",
                        "job_id": job_id,
                        "output_dir": job.output_dir,
                        "error_message": job.error_message,
                    })
                    break

                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    # Connection closed
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id[:8])

    except Exception as e:
        logger.error("WebSocket error for job %s: %s", job_id[:8], e)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass

    finally:
        # Stop subscription
        stop_event.set()
        logger.info("WebSocket closed for job %s", job_id[:8])
