"""
Unit tests for ml_engine.agent.llm_client.LLMClient.

All SDK calls are mocked -- no real API calls are made.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ml_engine.agent.llm_client import LLMClient, LLM_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Default model selection
# ---------------------------------------------------------------------------

class TestDefaultModel:
    r"""egression tests to ensure default model selection logic doesn't accidentally break."""
    def test_anthropic_default(self):
        r"""By default, should select a model with "claude" in the name for Anthropic."""
        c = LLMClient(provider="anthropic")
        assert "claude" in c.model.lower() or c.model.startswith("claude")

    def test_openai_default(self):
        r"""By default, should select a model with "gpt" in the name for OpenAI."""
        c = LLMClient(provider="openai")
        assert "gpt" in c.model.lower()

    def test_env_var_override_anthropic(self, monkeypatch):
        r"""Should use ANTHROPIC_MODEL env var if set."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
        c = LLMClient(provider="anthropic")
        assert c.model == "claude-test-model"

    def test_env_var_override_openai(self, monkeypatch):
        r"""Should use OPENAI_MODEL env var if set."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
        c = LLMClient(provider="openai")
        assert c.model == "gpt-test-model"

    def test_explicit_model_overrides_env(self, monkeypatch):
        r"""Model passed explicitly to constructor should override env var."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-from-env")
        c = LLMClient(provider="anthropic", model="claude-explicit")
        assert c.model == "claude-explicit"


# ---------------------------------------------------------------------------
# Constructor params
# ---------------------------------------------------------------------------

class TestConstructorParams:
    r"""Tests to verify that constructor parameters are stored correctly on the instance."""
    def test_default_timeout(self):
        r"""By default, should have a timeout of LLM_TIMEOUT_SECONDS."""
        c = LLMClient()
        assert c.timeout == LLM_TIMEOUT_SECONDS

    def test_custom_timeout(self):
        r"""Should store custom timeout passed to constructor."""
        c = LLMClient(timeout=10.0)
        assert c.timeout == 10.0

    def test_base_url_stored(self):
        r"""Should store base_url passed to constructor."""
        c = LLMClient(provider="openai", base_url="https://api.deepseek.com")
        assert c.base_url == "https://api.deepseek.com"

    def test_base_url_none_by_default(self):
        r"""By default, base_url should be None."""
        c = LLMClient()
        assert c.base_url is None

    def test_api_key_env_stored(self):
        r"""Should store api_key_env passed to constructor."""
        c = LLMClient(provider="openai", api_key_env="DEEPSEEK_API_KEY")
        assert c.api_key_env == "DEEPSEEK_API_KEY"

    def test_api_key_env_none_by_default(self):
        r"""By default, api_key_env should be None."""
        c = LLMClient()
        assert c.api_key_env is None


# ---------------------------------------------------------------------------
# OpenAI base_url passthrough
# ---------------------------------------------------------------------------

class TestBaseUrl:
    r"""Tests to verify that base_url constructor parameter is passed to OpenAI client and not included when None."""
    @pytest.mark.asyncio
    async def test_base_url_passed_to_openai_client(self, monkeypatch):
        r"""When base_url is provided, it should be passed to the OpenAI client constructor."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        c = LLMClient(provider="openai", base_url="https://api.deepseek.com")

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="test"),
            finish_reason="stop",
        )]
        mock_response.model = "deepseek-chat"
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("openai.AsyncOpenAI", mock_client_cls):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], None, 256)

        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://api.deepseek.com"
        assert call_kwargs["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_no_base_url_when_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        c = LLMClient(provider="openai")

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="test"),
            finish_reason="stop",
        )]
        mock_response.model = "gpt-4o"
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("openai.AsyncOpenAI", mock_client_cls):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], None, 256)

        call_kwargs = mock_client_cls.call_args[1]
        assert "base_url" not in call_kwargs


# ---------------------------------------------------------------------------
# Custom API key env var
# ---------------------------------------------------------------------------

class TestApiKeyEnv:
    @pytest.mark.asyncio
    async def test_custom_env_var_used(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key-123")
        c = LLMClient(provider="openai", api_key_env="DEEPSEEK_API_KEY")

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="test"),
            finish_reason="stop",
        )]
        mock_response.model = "deepseek-chat"
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("openai.AsyncOpenAI", mock_client_cls):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], None, 256)

        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["api_key"] == "ds-key-123"

    @pytest.mark.asyncio
    async def test_missing_custom_env_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c = LLMClient(provider="openai", api_key_env="DEEPSEEK_API_KEY")

        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY not set"):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], None, 256)

    @pytest.mark.asyncio
    async def test_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        c = LLMClient(provider="openai")  # no api_key_env

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="test"),
            finish_reason="stop",
        )]
        mock_response.model = "gpt-4o"
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("openai.AsyncOpenAI", mock_client_cls):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], None, 256)

        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["api_key"] == "oai-key"


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

class TestUnknownProvider:
    @pytest.mark.asyncio
    async def test_raises_value_error(self):
        c = LLMClient(provider="deepseek_native")  # not a real provider
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            await c._call_impl("system", [{"role": "user", "content": "hi"}], None, 256)


# ---------------------------------------------------------------------------
# Anthropic path
# ---------------------------------------------------------------------------

class TestCallAnthropic:
    @pytest.mark.asyncio
    async def test_happy_path_text_response(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        c = LLMClient(provider="anthropic")

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "hello world"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-sonnet-4-6"

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.messages.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("anthropic.AsyncAnthropic", mock_client_cls):
            result = await c._call_anthropic("system", [{"role": "user", "content": "hi"}], None, 256)

        assert result["content"] == [{"type": "text", "text": "hello world"}]
        assert result["stop_reason"] == "end_turn"
        assert result["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        c = LLMClient(provider="anthropic")

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
            await c._call_anthropic("system", [{"role": "user", "content": "hi"}], None, 256)

    @pytest.mark.asyncio
    async def test_tool_use_blocks_normalized(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        c = LLMClient(provider="anthropic")

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'll call the tool."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_123"
        tool_block.name = "dispatch_stage"
        tool_block.input = {"stage": "hpo"}

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.model = "claude-sonnet-4-6"

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.messages.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        with patch("anthropic.AsyncAnthropic", mock_client_cls):
            result = await c._call_anthropic("system", [{"role": "user", "content": "hi"}], [{"name": "dispatch_stage", "description": "d", "input_schema": {}}], 256)

        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "I'll call the tool."}
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "dispatch_stage"
        assert result["content"][1]["input"] == {"stage": "hpo"}


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    @pytest.mark.asyncio
    async def test_slow_call_raises_timeout_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        c = LLMClient(provider="anthropic", timeout=0.05)

        async def slow_create(**kwargs):
            await asyncio.sleep(1.0)

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.messages.create = slow_create
        mock_client_cls.return_value = mock_instance

        with patch("anthropic.AsyncAnthropic", mock_client_cls):
            with pytest.raises(asyncio.TimeoutError):
                await c.call("system", [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# OpenAI tool_calls in response
# ---------------------------------------------------------------------------

class TestOpenAIToolCalls:
    @pytest.mark.asyncio
    async def test_tool_calls_normalized_to_anthropic_format(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        c = LLMClient(provider="openai")

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc"
        mock_tool_call.function.name = "dispatch_stage"
        mock_tool_call.function.arguments = json.dumps({"stage": "hpo"})

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        tools = [{"name": "dispatch_stage", "description": "d", "input_schema": {"type": "object"}}]

        with patch("openai.AsyncOpenAI", mock_client_cls):
            result = await c._call_openai("system", [{"role": "user", "content": "hi"}], tools, 256)

        assert len(result["content"]) == 1
        tc = result["content"][0]
        assert tc["type"] == "tool_use"
        assert tc["id"] == "call_abc"
        assert tc["name"] == "dispatch_stage"
        assert tc["input"] == {"stage": "hpo"}

    @pytest.mark.asyncio
    async def test_tool_schema_converted_to_function_format(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        c = LLMClient(provider="openai")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content="ok", tool_calls=None),
            finish_reason="stop",
        )]
        mock_response.model = "gpt-4o"

        mock_client_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_instance

        tools = [{
            "name": "dispatch_stage",
            "description": "Dispatch a pipeline stage",
            "input_schema": {"type": "object", "properties": {"stage": {"type": "string"}}},
        }]

        with patch("openai.AsyncOpenAI", mock_client_cls):
            await c._call_openai("system", [{"role": "user", "content": "hi"}], tools, 256)

        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        oai_tools = call_kwargs["tools"]
        assert len(oai_tools) == 1
        assert oai_tools[0]["type"] == "function"
        assert oai_tools[0]["function"]["name"] == "dispatch_stage"
        assert oai_tools[0]["function"]["description"] == "Dispatch a pipeline stage"
        assert oai_tools[0]["function"]["parameters"] == {"type": "object", "properties": {"stage": {"type": "string"}}}


# ---------------------------------------------------------------------------
# _block_to_dict
# ---------------------------------------------------------------------------

class TestBlockToDict:
    def test_text_block(self):
        block = MagicMock()
        block.type = "text"
        block.text = "hello"
        result = LLMClient._block_to_dict(block)
        assert result == {"text": "hello"}

    def test_tool_use_block(self):
        block = MagicMock()
        block.type = "tool_use"
        block.id = "call_1"
        block.name = "my_tool"
        block.input = {"key": "val"}
        result = LLMClient._block_to_dict(block)
        assert result == {"id": "call_1", "name": "my_tool", "input": {"key": "val"}}

    def test_unknown_block_type_warns_and_returns_raw(self, caplog):
        block = MagicMock()
        block.type = "thinking"
        block.__str__ = lambda self: "ThinkingBlock(content='...')"

        with caplog.at_level(logging.WARNING):
            result = LLMClient._block_to_dict(block)

        assert "raw" in result
        assert "Unknown Anthropic content block type" in caplog.text


# ---------------------------------------------------------------------------
# Client caching
# ---------------------------------------------------------------------------

class TestClientCaching:
    def test_openai_client_cached(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        c = LLMClient(provider="openai")

        mock_client_cls = MagicMock()
        with patch("openai.AsyncOpenAI", mock_client_cls):
            client1 = c._get_openai_client()
            client2 = c._get_openai_client()

        assert client1 is client2
        mock_client_cls.assert_called_once()

    def test_anthropic_client_cached(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        c = LLMClient(provider="anthropic")

        mock_client_cls = MagicMock()
        with patch("anthropic.AsyncAnthropic", mock_client_cls):
            client1 = c._get_anthropic_client()
            client2 = c._get_anthropic_client()

        assert client1 is client2
        mock_client_cls.assert_called_once()
