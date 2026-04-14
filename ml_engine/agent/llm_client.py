"""
LLM client wrapper for the agentic layer.

Supports Anthropic and OpenAI. Configured via environment variables.
All calls have a hard 30-second timeout; LLM unavailability never blocks
a pipeline -- callers fall back to SimpleMutator or human escalation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 30


class LLMClient:
    """
    Thin async wrapper around Anthropic / OpenAI.

    Model is selected per-agent at construction time.
    Timeout is enforced with asyncio.wait_for -- if exceeded, raises
    asyncio.TimeoutError which callers must handle (fallback to SimpleMutator
    for HPO decisions, or human escalation for planning decisions).
    """

    def __init__(
        self,
        provider: str = "anthropic",   # "anthropic" | "openai"
        model: Optional[str] = None,
        timeout: float = LLM_TIMEOUT_SECONDS,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.timeout = timeout
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._anthropic_client: Any = None
        self._openai_client: Any = None

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return os.environ.get("OPENAI_MODEL", "gpt-4o")

    async def call(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Make one LLM call.

        Returns the raw response dict (Anthropic or OpenAI format).
        Raises asyncio.TimeoutError if the call exceeds self.timeout seconds.
        Raises RuntimeError if no API key is configured.
        """
        coro = self._call_impl(system, messages, tools, max_tokens)
        return await asyncio.wait_for(coro, timeout=self.timeout)

    async def _call_impl(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        if self.provider == "anthropic":
            return await self._call_anthropic(system, messages, tools, max_tokens)
        if self.provider == "openai":
            return await self._call_openai(system, messages, tools, max_tokens)
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _get_anthropic_client(self) -> Any:
        """Return cached AsyncAnthropic client, creating on first call."""
        if self._anthropic_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def _call_anthropic(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        client = self._get_anthropic_client()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)
        return {
            "content": [
                {"type": block.type, **self._block_to_dict(block)}
                for block in response.content
            ],
            "stop_reason": response.stop_reason,
            "model": response.model,
        }

    def _get_openai_client(self) -> Any:
        """Return cached AsyncOpenAI client, creating on first call."""
        if self._openai_client is None:
            env_var = self.api_key_env or "OPENAI_API_KEY"
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(f"{env_var} not set")
            from openai import AsyncOpenAI
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._openai_client = AsyncOpenAI(**client_kwargs)
        return self._openai_client

    async def _call_openai(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        client = self._get_openai_client()

        oai_messages = [{"role": "system", "content": system}] + messages
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            # Convert Anthropic tool schema to OpenAI function format
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ]

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Normalize response to match Anthropic content block format
        content_blocks: List[Dict[str, Any]] = []
        if choice.message.content:
            content_blocks.append({"type": "text", "text": choice.message.content})
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
        # Always include at least one text block for callers that expect it
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        return {
            "content": content_blocks,
            "stop_reason": choice.finish_reason,
            "model": response.model,
        }

    @staticmethod
    def _block_to_dict(block: Any) -> Dict[str, Any]:
        """Extract payload from an Anthropic content block (text or tool_use)."""
        if block.type == "text":
            return {"text": block.text}
        if block.type == "tool_use":
            return {"id": block.id, "name": block.name, "input": block.input}
        logger.warning("Unknown Anthropic content block type: %s", block.type)
        return {"raw": str(block)}
