"""
LLMProposeFn -- LLM-guided propose_fn for ExperimentLoop.

Stage 4 integration: the Executor agent replaces SimpleMutator with this
when `use_llm_propose=True` is set in the experiment job config.

ExperimentLoop.run() is synchronous (runs in a subprocess). This class
provides a synchronous propose() interface that calls the async LLM client
via asyncio.run() -- safe because the subprocess has no running event loop.

Fallback: if the LLM call fails or times out, SimpleMutator.propose() is
used instead. The pipeline never stalls waiting for LLM availability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from ml_engine.agent.llm_client import LLMClient
from ml_engine.agent.skills import SkillLoader
from ml_engine.experiment.trial_log import TrialLog
from ml_engine.experiment.mutators import SimpleMutator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are guiding hyperparameter optimization for a GroundingDINO + SAM LoRA \
fine-tuning experiment on a manufacturing quality inspection dataset.

Your job: given the trial history and mutable key constraints, propose the \
single most promising config override to try next.

Rules:
- Return ONLY a valid JSON object: {"key": value}
- One key at a time -- the loop handles one mutation per trial
- The key must exist in the mutable_keys list
- The value must respect the type and range constraints shown
- Use the trial history to reason: if a direction improves, continue it;
  if stuck, try a different axis; if oscillating, try the middle
- Never propose values outside the declared min/max bounds
- For log_scale=true keys (learning rates), think in log space

Respond with ONLY the JSON object. No explanation. No markdown fences.
Example: {"models.grounding_dino.lora.r": 64}
"""


class LLMProposeFn:
    """
    LLM-guided propose function for ExperimentLoop.

    Implements the ProposeFn signature: Callable[[TrialLog], Dict[str, Any]].
    Call propose() as you would SimpleMutator.propose().

    Usage::

        from ml_engine.experiment import SimpleMutator
        from ml_engine.experiment.llm_propose import LLMProposeFn

        fallback = SimpleMutator(mutable_keys=mutable_keys)
        propose_fn = LLMProposeFn(mutable_keys=mutable_keys, fallback=fallback)

        result = loop.run(..., propose_fn=propose_fn.propose)
    """

    def __init__(
        self,
        mutable_keys: Dict[str, Dict[str, Any]],
        fallback: Optional[SimpleMutator] = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        skill_name: str = "hpo_propose",
    ):
        self._mutable_keys = mutable_keys
        self._fallback = fallback or SimpleMutator(mutable_keys=mutable_keys)
        self._llm = LLMClient(
            provider=provider,
            model=model,
            timeout=timeout,
            base_url=base_url,
            api_key_env=api_key_env,
        )
        self._system_prompt = self._load_system_prompt(skill_name)
        self._consecutive_failures = 0
        self._MAX_CONSECUTIVE_FAILURES = 3

    @staticmethod
    def _load_system_prompt(skill_name: str) -> str:
        """Load system prompt from skill file, fall back to hardcoded default."""
        try:
            loader = SkillLoader()
            skill = loader.load(skill_name)
            return skill.to_system_prompt()
        except FileNotFoundError:
            logger.debug("Skill %r not found, using default _SYSTEM_PROMPT", skill_name)
            return _SYSTEM_PROMPT

    def propose(self, trial_log: TrialLog) -> Dict[str, Any]:
        """
        Propose next config overrides given trial history.

        Calls LLM with trial context and mutable key constraints.
        Falls back to SimpleMutator on LLM error, timeout, or bad response.

        After 3 consecutive LLM failures, logs a warning and always falls back
        until the next successful call resets the counter.
        """
        if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "LLMProposeFn: %d consecutive failures, using SimpleMutator fallback",
                self._consecutive_failures,
            )
            return self._fallback.propose(trial_log)

        try:
            result = asyncio.run(self._propose_async(trial_log))
            self._consecutive_failures = 0
            return result
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                "LLMProposeFn: LLM call failed (%d/%d): %s -- using SimpleMutator",
                self._consecutive_failures, self._MAX_CONSECUTIVE_FAILURES, e,
            )
            return self._fallback.propose(trial_log)

    async def _propose_async(self, trial_log: TrialLog) -> Dict[str, Any]:
        """Async LLM call with timeout. Raises on failure."""
        user_content = self._build_user_message(trial_log)
        response = await self._llm.call(
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_content}],
            tools=None,
            max_tokens=256,
        )

        # Extract text from response
        content_blocks = response.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                break

        overrides = self._parse_overrides(text)
        logger.info("LLMProposeFn proposed: %s", overrides)
        return overrides

    def _build_user_message(self, trial_log: TrialLog) -> str:
        """Build the user prompt: trial history + mutable key constraints."""
        lines = [trial_log.to_llm_context(), "", "Mutable keys and constraints:"]
        for key, schema in self._mutable_keys.items():
            vtype = schema.get("type", "any")
            if vtype in ("int", "float"):
                lo = schema.get("min", "?")
                hi = schema.get("max", "?")
                log_s = " [log_scale]" if schema.get("log_scale") else ""
                lines.append(f"  {key}: {vtype} [{lo}, {hi}]{log_s}")
            elif vtype == "choice":
                choices = schema.get("choices", [])
                lines.append(f"  {key}: choice {choices}")
            elif vtype == "bool":
                lines.append(f"  {key}: bool")
            else:
                lines.append(f"  {key}: {vtype}")
        lines.append("")
        lines.append("Propose one override as JSON:")
        return "\n".join(lines)

    def _parse_overrides(self, text: str) -> Dict[str, Any]:
        """
        Parse the LLM response text as a JSON overrides dict.

        Validates:
        - Is a non-empty dict
        - Key exists in mutable_keys
        - Value type matches schema

        Raises ValueError on parse or validation failure. The caller
        catches exceptions and increments consecutive_failures, falling
        back to SimpleMutator.
        """
        text = text.strip()

        # Strip markdown fences if the LLM ignored the instruction
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("LLMProposeFn: JSON parse failed: %s | text=%r", e, text[:100])
            raise ValueError(f"JSON parse failed: {e}") from e

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f"Expected non-empty dict, got: {type(parsed).__name__}")

        # Validate first key/value pair
        key = next(iter(parsed))
        value = parsed[key]

        if key not in self._mutable_keys:
            raise ValueError(f"Key {key!r} not in mutable_keys")

        schema = self._mutable_keys[key]
        vtype = schema.get("type", "any")

        if vtype == "int" and not isinstance(value, int):
            # LLM might return 64.0 instead of 64
            if isinstance(value, float) and value.is_integer():
                parsed[key] = int(value)
            else:
                raise ValueError(f"Key {key!r} expects int, got {type(value).__name__}")

        if vtype == "float" and not isinstance(value, (int, float)):
            raise ValueError(f"Key {key!r} expects float, got {type(value).__name__}")

        if vtype in ("int", "float"):
            lo = schema.get("min")
            hi = schema.get("max")
            v = parsed[key]
            if lo is not None and v < lo:
                raise ValueError(f"Key {key!r} value {v} below min {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"Key {key!r} value {v} above max {hi}")

        if vtype == "choice":
            choices = schema.get("choices", [])
            if value not in choices:
                raise ValueError(f"Key {key!r} value {value!r} not in choices {choices}")

        if vtype == "bool" and not isinstance(value, bool):
            raise ValueError(f"Key {key!r} expects bool, got {type(value).__name__}")

        # Return only the first key (one mutation per trial)
        return {key: parsed[key]}
