"""
Unit tests for ml_engine.experiment.llm_propose.LLMProposeFn.

The LLMClient is mocked so no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml_engine.experiment.llm_propose import LLMProposeFn
from ml_engine.experiment.mutators import SimpleMutator
from ml_engine.experiment.trial_log import TrialLog, TrialRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUTABLE_KEYS = {
    "batch_size": {"type": "int", "min": 1, "max": 32},
    "models.grounding_dino.lora.r": {"type": "int", "min": 2, "max": 128},
    "models.grounding_dino.learning_rate": {"type": "float", "min": 1e-6, "max": 1e-2, "log_scale": True},
    "optimizer": {"type": "choice", "choices": ["AdamW", "SGD"]},
    "training_dynamics.mixed_precision.enabled": {"type": "bool"},
}


def make_trial_log(num_trials: int = 2, tmp_path=None) -> TrialLog:
    import tempfile

    out_dir = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    log = TrialLog(run_id="test-run", output_dir=out_dir, budget_summary={"metric_mode": "max"})
    for i in range(num_trials):
        log.append(
            TrialRecord(
                trial_id=f"trial-{i}",
                overrides={"batch_size": 8 + i},
                primary_metric=0.4 + i * 0.05,
                all_metrics={"val_mAP50": 0.4 + i * 0.05},
                status="keep",
                description=f"trial {i}",
            )
        )
    return log


def llm_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def make_proposer(**kwargs) -> LLMProposeFn:
    return LLMProposeFn(mutable_keys=MUTABLE_KEYS, **kwargs)


# ---------------------------------------------------------------------------
# _parse_overrides
# ---------------------------------------------------------------------------


class TestParseOverrides:
    r"""_parse_overrides: parses the LLM text response into a dict of overrides."""

    def test_valid_int_key(self):
        r"""Valid int key should be parsed correctly."""
        p = make_proposer()
        result = p._parse_overrides('{"batch_size": 16}')
        assert result == {"batch_size": 16}

    def test_valid_float_key(self):
        r"""Valid float key should be parsed correctly."""
        p = make_proposer()
        result = p._parse_overrides('{"models.grounding_dino.learning_rate": 0.001}')
        assert result["models.grounding_dino.learning_rate"] == pytest.approx(0.001)

    def test_valid_choice_key(self):
        r"""Valid choice key should be parsed correctly."""
        p = make_proposer()
        result = p._parse_overrides('{"optimizer": "AdamW"}')
        assert result == {"optimizer": "AdamW"}

    def test_valid_bool_key(self):
        r"""Valid bool key should be parsed correctly."""
        p = make_proposer()
        result = p._parse_overrides('{"training_dynamics.mixed_precision.enabled": true}')
        assert result == {"training_dynamics.mixed_precision.enabled": True}

    def test_strips_markdown_fences(self):
        r"""LLM may return JSON wrapped in markdown code fences, which should be stripped."""
        p = make_proposer()
        text = '```json\n{"batch_size": 4}\n```'
        result = p._parse_overrides(text)
        assert result == {"batch_size": 4}

    def test_strips_plain_code_fence(self):
        r"""LLM may return JSON wrapped in plain code fences, which should be stripped."""
        p = make_proposer()
        text = '```\n{"batch_size": 8}\n```'
        result = p._parse_overrides(text)
        assert result == {"batch_size": 8}

    def test_invalid_json_raises(self):
        r"""Invalid JSON should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="JSON parse failed"):
            p._parse_overrides("not json")

    def test_empty_dict_raises(self):
        r"""Empty dict should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="non-empty dict"):
            p._parse_overrides("{}")

    def test_list_raises(self):
        r"""List should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="non-empty dict"):
            p._parse_overrides("[1, 2, 3]")

    def test_unknown_key_raises(self):
        r"""Unknown key should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="not in mutable_keys"):
            p._parse_overrides('{"unknown_key_xyz": 5}')

    def test_int_key_float_value_coerced(self):
        r"""LLM returns 64.0 for an int key -- should be coerced to 64."""
        p = make_proposer()
        result = p._parse_overrides('{"batch_size": 16.0}')
        assert result == {"batch_size": 16}
        assert isinstance(result["batch_size"], int)

    def test_int_key_non_integer_float_raises(self):
        r"""LLM returns 16.5 for an int key -- should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="expects int"):
            p._parse_overrides('{"batch_size": 16.5}')

    def test_int_below_min_raises(self):
        r"""Int value below min should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="below min"):
            p._parse_overrides('{"batch_size": 0}')  # min is 1

    def test_int_above_max_raises(self):
        r"""Int value above max should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="above max"):
            p._parse_overrides('{"batch_size": 100}')  # max is 32

    def test_float_below_min_raises(self):
        r"""Float value below min should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="below min"):
            p._parse_overrides('{"models.grounding_dino.learning_rate": 1e-9}')

    def test_choice_not_in_choices_raises(self):
        r"""Choice value not in choices should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="not in choices"):
            p._parse_overrides('{"optimizer": "RMSProp"}')

    def test_bool_key_string_raises(self):
        r"""LLM returns "true" for a bool key -- should raise ValueError."""
        p = make_proposer()
        with pytest.raises(ValueError, match="expects bool"):
            p._parse_overrides('{"training_dynamics.mixed_precision.enabled": "true"}')

    def test_only_first_key_returned(self):
        r"""Multi-key response -- only first key should be returned."""
        p = make_proposer()
        result = p._parse_overrides('{"batch_size": 4, "optimizer": "SGD"}')
        assert len(result) == 1
        assert "batch_size" in result


# ---------------------------------------------------------------------------
# propose() -- fallback behavior
# ---------------------------------------------------------------------------


class TestProposeFallback:
    r"""Tests for the propose() method's fallback behavior when _parse_overrides raises exceptions."""

    def test_falls_back_after_3_consecutive_failures(self):
        r"""After 3 consecutive failures, should call fallback.propose() and return its result."""
        fallback = MagicMock(spec=SimpleMutator)
        fallback.propose.return_value = {"batch_size": 8}

        p = LLMProposeFn(
            mutable_keys=MUTABLE_KEYS,
            fallback=fallback,
        )
        p._consecutive_failures = 3  # at threshold

        trial_log = make_trial_log()
        result = p.propose(trial_log)

        fallback.propose.assert_called_once_with(trial_log)
        assert result == {"batch_size": 8}

    def test_fallback_created_automatically_when_not_provided(self):
        r"""Fallback should be created automatically when not provided."""
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        assert p._fallback is not None
        assert isinstance(p._fallback, SimpleMutator)

    def test_consecutive_failures_reset_on_success(self):
        r"""Consecutive failures counter should be reset on successful proposal."""
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        p._consecutive_failures = 2

        async def mock_propose_async(trial_log):
            return {"batch_size": 16}

        with patch.object(p, "_propose_async", new=mock_propose_async):
            import asyncio

            # Patch asyncio.run to call the coroutine directly
            with patch("ml_engine.experiment.llm_propose.asyncio") as mock_asyncio:
                mock_asyncio.run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
                p.propose(make_trial_log())  # side effect: resets _consecutive_failures

        assert p._consecutive_failures == 0

    def test_failure_increments_counter(self):
        r"""Exception in _propose_async should increment consecutive_failures and call fallback."""
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        p._consecutive_failures = 0

        with patch("ml_engine.experiment.llm_propose.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = RuntimeError("LLM down")
            result = p.propose(make_trial_log())

        # Counter incremented
        assert p._consecutive_failures == 1
        # Fallback result returned (not None, not raised)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_includes_mutable_key_names(self):
        p = make_proposer()
        msg = p._build_user_message(make_trial_log())
        assert "batch_size" in msg
        assert "optimizer" in msg

    def test_includes_type_and_range(self):
        p = make_proposer()
        msg = p._build_user_message(make_trial_log())
        assert "int" in msg
        assert "[1, 32]" in msg  # batch_size range

    def test_includes_log_scale_annotation(self):
        p = make_proposer()
        msg = p._build_user_message(make_trial_log())
        assert "[log_scale]" in msg

    def test_includes_choice_list(self):
        p = make_proposer()
        msg = p._build_user_message(make_trial_log())
        assert "AdamW" in msg
        assert "SGD" in msg

    def test_includes_trial_history(self):
        p = make_proposer()
        trial_log = make_trial_log(2)
        msg = p._build_user_message(trial_log)
        # TrialLog.to_llm_context() should be embedded
        assert len(msg) > 100  # non-trivial


# ---------------------------------------------------------------------------
# Skill file loading
# ---------------------------------------------------------------------------


class TestSkillLoading:
    def test_loads_hpo_propose_skill(self):
        """Should load the hpo_propose.md skill file and use it as system prompt."""
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        # The skill file exists, so system prompt should contain skill content
        assert (
            "hyperparameter optimization" in p._system_prompt.lower()
            or "hpo" in p._system_prompt.lower()
            or "propose" in p._system_prompt.lower()
        )

    def test_falls_back_to_default_on_missing_skill(self):
        """If skill file doesn't exist, should use the hardcoded _SYSTEM_PROMPT."""
        from ml_engine.experiment.llm_propose import _SYSTEM_PROMPT

        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS, skill_name="nonexistent_skill_xyz")
        assert p._system_prompt == _SYSTEM_PROMPT

    def test_skill_prompt_contains_json_instruction(self):
        """The skill prompt must instruct JSON-only output."""
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        assert "JSON" in p._system_prompt


# ---------------------------------------------------------------------------
# LLMClient created once (not per call)
# ---------------------------------------------------------------------------


class TestLLMClientReuse:
    def test_llm_client_is_instance_attribute(self):
        p = LLMProposeFn(mutable_keys=MUTABLE_KEYS)
        from ml_engine.agent.llm_client import LLMClient

        assert isinstance(p._llm, LLMClient)

    def test_base_url_threaded_to_client(self):
        p = LLMProposeFn(
            mutable_keys=MUTABLE_KEYS,
            provider="openai",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        )
        assert p._llm.base_url == "https://api.deepseek.com"
        assert p._llm.api_key_env == "DEEPSEEK_API_KEY"
        assert p._llm.provider == "openai"
