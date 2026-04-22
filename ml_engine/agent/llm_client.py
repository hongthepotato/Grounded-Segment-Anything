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

    @staticmethod
    def _to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Anthropic-style messages to OpenAI chat-completion format.

        Claude is LLMClient's canonical internal format. Callers pass messages
        with typed content blocks (text / tool_use / tool_result). OpenAI's API
        expects tool invocations on `assistant.tool_calls[]` and tool outputs
        as separate `role=tool` messages. Without this conversion, turn 2+ of
        any tool-using conversation fails with
        `unknown variant 'tool_use', expected 'text'`.
        """
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Plain-string content passes through unchanged.
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                out.append(msg)
                continue

            if role == "user":
                # tool_result blocks become separate role=tool messages;
                # text blocks fold into a single role=user message.
                text_parts: List[str] = []
                for block in content:
                    btype = block.get("type")
                    if btype == "tool_result":
                        tc = block.get("content")
                        if isinstance(tc, list):
                            tc = "".join(
                                b.get("text", "")
                                for b in tc
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        out.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": tc if isinstance(tc, str) else json.dumps(tc),
                        })
                    elif btype == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    out.append({"role": "user", "content": "".join(text_parts)})

            elif role == "assistant":
                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                oai_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                out.append(oai_msg)

            else:
                out.append(msg)
        return out

    async def _call_openai(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        client = self._get_openai_client()

        oai_messages = [{"role": "system", "content": system}] + self._to_openai_messages(messages)
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
