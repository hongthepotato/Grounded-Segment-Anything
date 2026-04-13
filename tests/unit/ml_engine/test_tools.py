"""
Unit tests for ml_engine.agent.tools.

Tests ToolResult, RunContext, Tool (via concrete subclass), ToolRegistry.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ml_engine.agent.tools import Tool, ToolRegistry, ToolResult, RunContext


# ---------------------------------------------------------------------------
# Concrete test tool
# ---------------------------------------------------------------------------

class EchoInput(BaseModel):
    message: str
    fail: bool = False


class EchoTool(Tool[EchoInput]):
    name = "echo"
    description = "Echoes a message back"
    input_schema = EchoInput

    async def execute(self, args: EchoInput, context: RunContext) -> ToolResult:
        if args.fail:
            return ToolResult(success=False, error="forced failure")
        return ToolResult(success=True, output=args.message)


class NoDescTool(Tool[EchoInput]):
    name = "nodesc"
    description = ""
    input_schema = EchoInput

    async def execute(self, args: EchoInput, context: RunContext) -> ToolResult:
        return ToolResult(success=True)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_success_true(self):
        r = ToolResult(success=True, output="done")
        assert r.success is True
        assert r.output == "done"
        assert r.error is None

    def test_failure_with_error(self):
        r = ToolResult(success=False, error="something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"

    def test_output_can_be_dict(self):
        r = ToolResult(success=True, output={"job_id": "abc"})
        assert r.output["job_id"] == "abc"

    def test_output_defaults_to_none(self):
        r = ToolResult(success=True)
        assert r.output is None


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

class TestRunContext:
    def test_minimal_construction(self):
        ctx = RunContext(run_id="run-1", redis_url="redis://localhost:6379")
        assert ctx.run_id == "run-1"
        assert ctx.contract is None
        assert not ctx.extra

    def test_with_contract(self):
        ctx = RunContext(
            run_id="run-2",
            redis_url="redis://localhost:6379",
            contract={"id": "c1"},
        )
        assert ctx.contract["id"] == "c1"

    def test_extra_dict(self):
        ctx = RunContext(
            run_id="run-3",
            redis_url="redis://localhost:6379",
            extra={"gpu_id": 0},
        )
        assert ctx.extra["gpu_id"] == 0


# ---------------------------------------------------------------------------
# Tool.validate (default impl)
# ---------------------------------------------------------------------------

class TestToolValidate:
    def test_default_validate_returns_empty(self):
        tool = EchoTool()
        args = EchoInput(message="hello")
        assert tool.validate(args) == []


# ---------------------------------------------------------------------------
# Tool.to_llm_schema
# ---------------------------------------------------------------------------

class TestToolLlmSchema:
    def test_schema_has_name(self):
        tool = EchoTool()
        schema = tool.to_llm_schema()
        assert schema["name"] == "echo"

    def test_schema_has_description(self):
        tool = EchoTool()
        schema = tool.to_llm_schema()
        assert schema["description"] == "Echoes a message back"

    def test_schema_has_input_schema(self):
        tool = EchoTool()
        schema = tool.to_llm_schema()
        assert "input_schema" in schema
        assert isinstance(schema["input_schema"], dict)

    def test_input_schema_includes_properties(self):
        tool = EchoTool()
        schema = tool.to_llm_schema()
        props = schema["input_schema"].get("properties", {})
        assert "message" in props


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_get_unknown_raises_key_error(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            registry.get("nonexistent")

    def test_all_schemas_returns_list(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(NoDescTool())
        schemas = registry.all_schemas()
        assert len(schemas) == 2

    def test_all_schemas_names(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(NoDescTool())
        names = {s["name"] for s in registry.all_schemas()}
        assert names == {"echo", "nodesc"}

    def test_register_overwrites_same_name(self):
        """Registering the same name twice replaces the first."""
        registry = ToolRegistry()
        t1 = EchoTool()
        t2 = EchoTool()
        registry.register(t1)
        registry.register(t2)
        assert registry.get("echo") is t2

    def test_empty_registry_all_schemas_is_empty(self):
        registry = ToolRegistry()
        assert registry.all_schemas() == []

    def test_error_message_includes_available_tools(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        with pytest.raises(KeyError) as exc_info:
            registry.get("missing")
        assert "echo" in str(exc_info.value)
