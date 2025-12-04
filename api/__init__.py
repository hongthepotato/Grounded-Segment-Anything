"""
API package for Training Job Manager.

Provides REST and WebSocket endpoints for job management.

Usage:
    # Start API server
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from api.app import app

__all__ = ["app"]





