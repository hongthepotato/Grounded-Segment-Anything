"""
Simplified Tool interface for the agentic layer.

Adapted from Claude Code's Tool.ts -- dropped:
  - is_concurrency_safe (one GPU, one job, queue handles it)
  - is_read_only (not useful at multi-hour timescales)
  - check_permissions per call (boundaries enforced by PipelineContract + ConfigGuard)

Kept:
  - Typed input schema (Pydantic)
  - validate() + execute() methods
  - Flat tool registry
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ToolResult(BaseModel):
    r"""Standardized result format for tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None


class RunContext(BaseModel):
    """Context passed to tool.execute(). Injected by the agent loop."""

    run_id: str
    redis_url: str
    contract: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = {}


class Tool(ABC, Generic[T]):
    """
    Abstract base for all Coordinator tools.

    Each tool wraps one specific capability (submit a job, read state, etc.).
    Heavy dependencies must be imported inside execute() -- tools may run in
    a subprocess or across restarts.
    """

    name: str
    description: str
    input_schema: Type[T]

    def validate(self, args: T) -> List[str]:
        """Return a list of validation error strings. Empty = valid."""
        return []

    @abstractmethod
    async def execute(self, args: T, context: RunContext) -> ToolResult:
        """Execute the tool. May be long-running (submits job, returns handle)."""
        ...

    def to_llm_schema(self) -> Dict[str, Any]:
        """Anthropic tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }


class ToolRegistry:
    """Flat dict of tool_name -> Tool instance."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        r"""Register a tool instance. Tools must be registered before the agent loop starts."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool:
        r"""Lookup a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {list(self._tools)}")
        return self._tools[name]

    def all_schemas(self) -> List[Dict[str, Any]]:
        r"""Return a list of all registered tools in LLM schema format (for prompting)."""
        return [t.to_llm_schema() for t in self._tools.values()]
