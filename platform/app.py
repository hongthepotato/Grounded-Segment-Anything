"""Platform FastAPI application.

Routes:
  GET  /                        → canvas (HTML)
  GET  /api/blocks              → block catalog
  POST /api/blocks/refresh      → re-scan blocks/ directory
  GET  /api/hardware            → detected hardware
  GET  /api/graphs              → saved graph list
  POST /api/graphs              → save/create graph
  GET  /api/graphs/{id}         → load graph
  DELETE /api/graphs/{id}       → delete graph
  POST /api/pipelines           → launch pipeline from graph
  GET  /api/pipelines/{id}/status → pipeline service status
  GET  /api/pipelines/{id}/logs/{service} → container logs
  POST /api/pipelines/{id}/restart/{service} → restart service
  DELETE /api/pipelines/{id}    → stop pipeline
  POST /api/compose             → start LLM composition session
  GET  /api/compose/{id}/stream → SSE stream of composition events
  GET  /api/compose/{id}/status → session state
  GET  /api/llm-available       → connectivity check
  POST /api/auth/setup          → first-run engineer account creation
  GET  /api/auth/users          → list users (engineer only)
  POST /api/auth/users          → create user (engineer only)
  DELETE /api/auth/users/{name} → delete user (engineer only)
  GET  /health                  → liveness probe
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, composer_agent, graph_store, launch_engine, registry

logger = logging.getLogger(__name__)

app = FastAPI(title="ROS2 Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── In-memory platform state ──────────────────────────────────────────────────
_blocks_catalog = []
_detected_hardware = []
_active_pipelines: Dict[str, Dict] = {}


@app.on_event("startup")
async def _startup():
    global _blocks_catalog, _detected_hardware
    _blocks_catalog = registry.scan_blocks()
    _detected_hardware = registry.detect_hardware()
    logger.info(f"Platform started. {len(_blocks_catalog)} blocks loaded.")


# ── Auth ──────────────────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    username: str
    pin: str


class UserCreateRequest(BaseModel):
    username: str
    pin: str
    role: str


@app.get("/api/auth/first-run")
def auth_first_run():
    return {"first_run": auth.is_first_run()}


@app.post("/api/auth/setup")
def auth_setup(req: SetupRequest):
    """First-run engineer account creation. Only works when auth.json is empty."""
    if not auth.is_first_run():
        raise HTTPException(status_code=400, detail="Setup already completed")
    auth.create_user(req.username, req.pin, "engineer")
    return {"message": f"Engineer account '{req.username}' created"}


@app.post("/api/auth/login")
def auth_login(user: dict = Depends(auth.require_auth)):
    return {"username": user["username"], "role": user["role"]}


@app.get("/api/auth/users")
def auth_list_users(engineer: dict = Depends(auth.require_engineer)):
    return {"users": auth.list_users()}


@app.post("/api/auth/users")
def auth_create_user(req: UserCreateRequest, engineer: dict = Depends(auth.require_engineer)):
    auth.create_user(req.username, req.pin, req.role)
    return {"message": f"User '{req.username}' created with role '{req.role}'"}


@app.delete("/api/auth/users/{username}")
def auth_delete_user(username: str, engineer: dict = Depends(auth.require_engineer)):
    if not auth.delete_user(username):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User '{username}' deleted"}


# ── Block registry ────────────────────────────────────────────────────────────

@app.get("/api/blocks")
def get_blocks(user: dict = Depends(auth.require_auth)):
    return {
        "blocks": _blocks_catalog,
        "detected_hardware": _detected_hardware,
    }


@app.post("/api/blocks/refresh")
def refresh_blocks(engineer: dict = Depends(auth.require_engineer)):
    global _blocks_catalog, _detected_hardware
    _blocks_catalog = registry.scan_blocks()
    _detected_hardware = registry.detect_hardware()
    return {"message": "Block registry refreshed", "count": len(_blocks_catalog)}


@app.get("/api/hardware")
def get_hardware(user: dict = Depends(auth.require_auth)):
    return {"detected_hardware": _detected_hardware}


# ── Graph persistence ─────────────────────────────────────────────────────────

@app.get("/api/graphs")
def list_graphs(user: dict = Depends(auth.require_auth)):
    return {"graphs": graph_store.list_graphs()}


@app.post("/api/graphs")
def save_graph(graph: Dict[str, Any], user: dict = Depends(auth.require_auth)):
    saved = graph_store.save_graph(graph)
    return saved


@app.get("/api/graphs/{graph_id}")
def get_graph(graph_id: str, user: dict = Depends(auth.require_auth)):
    g = graph_store.load_graph(graph_id)
    if g is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    return g


@app.delete("/api/graphs/{graph_id}")
def delete_graph(graph_id: str, user: dict = Depends(auth.require_auth)):
    if not graph_store.delete_graph(graph_id):
        raise HTTPException(status_code=404, detail="Graph not found")
    return {"message": f"Graph {graph_id} deleted"}


# ── Pipeline management ───────────────────────────────────────────────────────

class LaunchRequest(BaseModel):
    graph_id: Optional[str] = None
    graph: Optional[Dict[str, Any]] = None


@app.post("/api/pipelines")
async def launch_pipeline(req: LaunchRequest, user: dict = Depends(auth.require_auth)):
    if req.graph is not None:
        graph = req.graph
    elif req.graph_id is not None:
        graph = graph_store.load_graph(req.graph_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Graph not found")
    else:
        raise HTTPException(status_code=400, detail="Provide graph_id or graph")

    try:
        pipeline = await launch_engine.launch_pipeline(graph, _blocks_catalog)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    _active_pipelines[pipeline["pipeline_id"]] = pipeline
    return pipeline


@app.get("/api/pipelines")
def list_pipelines(user: dict = Depends(auth.require_auth)):
    return {"pipelines": list(_active_pipelines.values())}


@app.get("/api/pipelines/{pipeline_id}/status")
async def pipeline_status(pipeline_id: str, user: dict = Depends(auth.require_auth)):
    pipeline = _active_pipelines.get(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    services = await launch_engine.get_pipeline_status(pipeline["project_name"])
    return {"pipeline_id": pipeline_id, "services": services}


@app.get("/api/pipelines/{pipeline_id}/logs/{service_name}")
async def pipeline_logs(
    pipeline_id: str,
    service_name: str,
    tail: int = 100,
    user: dict = Depends(auth.require_auth),
):
    pipeline = _active_pipelines.get(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    logs = await launch_engine.get_service_logs(pipeline["project_name"], service_name, tail)
    return {"logs": logs}


@app.post("/api/pipelines/{pipeline_id}/restart/{service_name}")
async def restart_service(
    pipeline_id: str,
    service_name: str,
    user: dict = Depends(auth.require_auth),
):
    pipeline = _active_pipelines.get(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    await launch_engine.restart_service(pipeline["project_name"], service_name)
    return {"message": f"Service {service_name} restarted"}


@app.delete("/api/pipelines/{pipeline_id}")
async def stop_pipeline(pipeline_id: str, user: dict = Depends(auth.require_auth)):
    pipeline = _active_pipelines.pop(pipeline_id, None)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    await launch_engine.stop_pipeline(
        pipeline["project_name"],
        domain_id=pipeline.get("domain_id"),
    )
    return {"message": f"Pipeline {pipeline_id} stopped"}


# ── LLM Composition ───────────────────────────────────────────────────────────

class ComposeRequest(BaseModel):
    prompt: str


@app.post("/api/compose")
def start_compose(req: ComposeRequest, user: dict = Depends(auth.require_auth)):
    session_id = composer_agent.create_session(req.prompt, _blocks_catalog)
    asyncio.create_task(_run_composition_background(session_id))
    return {"session_id": session_id}


async def _run_composition_background(session_id: str):
    """Runs the agent loop; events are stored in session for SSE replay."""
    session = composer_agent.get_session(session_id)
    if session is None:
        return
    async for event in composer_agent.run_session(session_id):
        session.setdefault("event_buffer", []).append(event)


@app.get("/api/compose/{session_id}/stream")
async def compose_stream(session_id: str, user: dict = Depends(auth.require_auth)):
    """SSE stream. Replays buffered events and then follows live."""
    session = composer_agent.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    async def _generate():
        sent = 0
        while True:
            buffer = session.get("event_buffer", [])
            while sent < len(buffer):
                yield buffer[sent]
                sent += 1
            if session.get("finalized") or "done" in "".join(buffer[-1:] if buffer else []):
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/compose/{session_id}/status")
def compose_status(session_id: str, user: dict = Depends(auth.require_auth)):
    session = composer_agent.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {
        "session_id": session_id,
        "finalized": session["finalized"],
        "tool_call_count": session["tool_call_count"],
        "graph": session["graph"],
    }


@app.get("/api/llm-available")
def llm_available():
    return {"available": composer_agent.check_llm_connectivity()}


# ── Canvas (frontend) ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def canvas():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Platform UI not found</h1>")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "blocks": len(_blocks_catalog)}
