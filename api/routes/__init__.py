"""API routes package."""

from api.routes.jobs import router as jobs_router
from api.routes.websocket import router as websocket_router
from api.routes.distillation import router as distillation_router

__all__ = ["jobs_router", "websocket_router", "distillation_router"]
