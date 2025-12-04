"""
FastAPI application for Training Job Manager.

This module provides:
- FastAPI application with CORS, lifespan management
- REST endpoints for job CRUD
- WebSocket for real-time updates
- Queue status endpoints

Usage:
    # Start server
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
    
    # Or programmatically
    import uvicorn
    from api.app import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.jobs import router as jobs_router, queue_router
from api.routes.websocket import router as websocket_router
from ml_engine.jobs import get_job_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown:
    - Startup: Initialize JobManager, verify Redis connection
    - Shutdown: Close Redis connections
    """
    # Startup
    logger.info("Starting Training Job Manager API...")
    
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    try:
        manager = get_job_manager(redis_url)
        logger.info("Connected to Redis at %s", redis_url)
        
        # Cleanup stale workers on startup
        removed = manager.cleanup_stale_workers(timeout_seconds=120)
        if removed > 0:
            logger.info("Removed %d stale workers", removed)
            
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Training Job Manager API...")
    manager = get_job_manager(redis_url)
    manager.close()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Training Job Manager API",
    description="""
API for managing ML training jobs.

## Features

- **Job Submission**: Submit teacher training and distillation jobs
- **Job Management**: Cancel, list, and query job status
- **Real-time Updates**: WebSocket for live progress updates
- **Queue Monitoring**: View queue status and worker availability

## Endpoints

### Jobs
- `POST /api/jobs` - Submit new job
- `GET /api/jobs` - List jobs with filtering
- `GET /api/jobs/{id}` - Get job details
- `DELETE /api/jobs/{id}` - Cancel job

### Queue
- `GET /api/queue/status` - Queue and worker status

### WebSocket
- `WS /ws/jobs/{id}` - Real-time job updates
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
# In production, replace "*" with specific origins
cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(jobs_router)
app.include_router(queue_router)
app.include_router(websocket_router)


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        {"status": "healthy", "redis": "connected"}
    """
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    try:
        manager = get_job_manager(redis_url)
        # Quick Redis check
        manager.get_queue_length()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": f"error: {str(e)}"}


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Training Job Manager API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )





