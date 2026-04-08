"""
ConfigGuard -- boundary enforcement for HPO config mutations.

Validates that proposed config changes stay within declared mutable ranges
and don't touch immutable keys. Used by ExperimentLoop before each trial.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    passed: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


class ConfigGuard:
    """
    Validates config mutations before each experiment trial.

    mutable_keys schema::

        {
            "lora.r": {"type": "int", "min": 2, "max": 128},
            "training.optimizer": {"type": "choice", "choices": ["AdamW", "SGD"]},
            "augmentation.horizontal_flip": {"type": "bool"},
        }

    immutable_keys: list of dotted paths that must not change between trials.
    """

    def __init__(
        self,
        mutable_keys: Dict[str, Dict[str, Any]],
        immutable_keys: List[str],
    ):
        self._mutable = mutable_keys
        self._immutable = set(immutable_keys)

    def validate(self, proposed_overrides: Dict[str, Any]) -> GuardResult:
        """
        Check that proposed_overrides are within declared boundaries.

        Validates the flat dotted-key override dict produced by SimpleMutator
        (or an LLM-guided proposer at Stage 4). Does not need the base config
        because all constraints are absolute (type, range, choice), not relative.

        Args:
            proposed_overrides: Flat dict of dotted_key -> value to apply,
                e.g. {"lora.r": 32, "learning_rate": 1e-4}.

        Returns:
            GuardResult with passed=True if all checks pass.
        """
        errors: List[str] = []

        for key, value in proposed_overrides.items():
            # Immutable check
            if key in self._immutable:
                errors.append(f"'{key}' is immutable and cannot be changed \
                              by the experiment loop")
                continue

            # Must be in mutable_keys
            if key not in self._mutable:
                errors.append(f"'{key}' is not in mutable_keys — \
                              add it explicitly to allow HPO mutations")
                continue

            schema = self._mutable[key]
            vtype = schema.get("type", "any")

            # Type + range checks
            if vtype == "int":
                if not isinstance(value, int):
                    errors.append(f"'{key}' must be int, got {type(value).__name__}")
                    continue
                if "min" in schema and value < schema["min"]:
                    errors.append(f"'{key}'={value} below min={schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"'{key}'={value} above max={schema['max']}")

            elif vtype == "float":
                if not isinstance(value, (int, float)):
                    errors.append(f"'{key}' must be float, got {type(value).__name__}")
                    continue
                if "min" in schema and value < schema["min"]:
                    errors.append(f"'{key}'={value} below min={schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"'{key}'={value} above max={schema['max']}")

            elif vtype == "bool":
                if not isinstance(value, bool):
                    errors.append(f"'{key}' must be bool, got {type(value).__name__}")

            elif vtype == "choice":
                choices = schema.get("choices", [])
                if value not in choices:
                    errors.append(f"'{key}'={value!r} not in allowed choices {choices}")

            elif vtype == "list":
                if not isinstance(value, list):
                    errors.append(f"'{key}' must be list, got {type(value).__name__}")

        passed = len(errors) == 0
        if not passed:
            logger.warning("ConfigGuard rejected overrides: %s", errors)
        return GuardResult(passed=passed, errors=errors)
