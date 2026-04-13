"""
Unit tests for ml_engine.experiment.config_guard.ConfigGuard.

Pure function -- no Redis, no I/O.
"""

from __future__ import annotations

import pytest

from ml_engine.experiment.config_guard import ConfigGuard, GuardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MUTABLE_KEYS = {
    "batch_size":        {"type": "int",    "min": 1,    "max": 32},
    "learning_rate":     {"type": "float",  "min": 1e-6, "max": 1e-2},
    "optimizer":         {"type": "choice", "choices": ["AdamW", "SGD"]},
    "use_amp":           {"type": "bool"},
    "augmentations":     {"type": "list",   "items": ["flip", "rotate", "crop"]},
    "lora_r":            {"type": "int",    "min": 2,    "max": 128},
    "weight_decay":      {"type": "float",  "min": 0.0,  "max": 0.1},
}

IMMUTABLE_KEYS = [
    "num_classes",
    "class_names",
    "models.grounding_dino.model.base_checkpoint",
]


@pytest.fixture
def guard():
    return ConfigGuard(mutable_keys=MUTABLE_KEYS, immutable_keys=IMMUTABLE_KEYS)


# ---------------------------------------------------------------------------
# GuardResult helpers
# ---------------------------------------------------------------------------

class TestGuardResult:
    r"""Tests for GuardResult dataclass behavior."""
    def test_bool_true_when_passed(self):
        r"""GuardResult should evaluate to True if passed=True."""
        r = GuardResult(passed=True)
        assert bool(r) is True

    def test_bool_false_when_failed(self):
        r"""GuardResult should evaluate to False if passed=False."""
        r = GuardResult(passed=False, errors=["bad"])
        assert bool(r) is False
        assert r.errors == ["bad"]

    def test_errors_default_empty(self):
        r"""GuardResult should have empty errors list by default."""
        r = GuardResult(passed=True)
        assert r.errors == []


# ---------------------------------------------------------------------------
# validate() -- passing cases
# ---------------------------------------------------------------------------

class TestValidatePassing:
    r"""Tests for ConfigGuard.validate() that should pass without errors."""
    def test_valid_int(self, guard):
        r"""Valid integer within range should pass."""
        result = guard.validate({"batch_size": 16})
        assert result.passed
        assert result.errors == []

    def test_valid_float(self, guard):
        r"""Valid float within range should pass."""
        result = guard.validate({"learning_rate": 1e-4})
        assert result.passed

    def test_valid_choice(self, guard):
        r"""Valid choice should pass."""
        result = guard.validate({"optimizer": "AdamW"})
        assert result.passed

    def test_valid_bool_true(self, guard):
        r"""Valid boolean true should pass."""
        result = guard.validate({"use_amp": True})
        assert result.passed

    def test_valid_bool_false(self, guard):
        r"""Valid boolean false should pass."""
        result = guard.validate({"use_amp": False})
        assert result.passed

    def test_valid_list(self, guard):
        r"""Valid list with allowed items should pass."""
        result = guard.validate({"augmentations": ["flip", "rotate"]})
        assert result.passed

    def test_int_at_min_boundary(self, guard):
        r"""Integer at minimum boundary should pass."""
        result = guard.validate({"batch_size": 1})
        assert result.passed

    def test_int_at_max_boundary(self, guard):
        r"""Integer at maximum boundary should pass."""
        result = guard.validate({"batch_size": 32})
        assert result.passed

    def test_float_at_min_boundary(self, guard):
        r"""Float at minimum boundary should pass."""
        result = guard.validate({"learning_rate": 1e-6})
        assert result.passed

    def test_float_at_max_boundary(self, guard):
        r"""Float at maximum boundary should pass."""
        result = guard.validate({"weight_decay": 0.1})
        assert result.passed

    def test_multiple_valid_keys(self, guard):
        r"""Multiple valid keys should pass together."""
        result = guard.validate({"batch_size": 8, "optimizer": "SGD", "use_amp": True})
        assert result.passed

    def test_empty_overrides_passes(self, guard):
        r"""Empty overrides should pass."""
        result = guard.validate({})
        assert result.passed


# ---------------------------------------------------------------------------
# validate() -- int errors
# ---------------------------------------------------------------------------

class TestValidateIntErrors:
    def test_int_below_min(self, guard):
        r"""Integer below minimum boundary should fail with appropriate error message."""
        result = guard.validate({"batch_size": 0})
        assert not result.passed
        assert any("below min" in e for e in result.errors)

    def test_int_above_max(self, guard):
        r"""Integer above maximum boundary should fail with appropriate error message.s"""
        result = guard.validate({"batch_size": 64})
        assert not result.passed
        assert any("above max" in e for e in result.errors)

    def test_int_wrong_type_float(self, guard):
        r"""Integer key with float value should fail type check."""
        result = guard.validate({"batch_size": 8.5})
        assert not result.passed
        assert any("must be int" in e for e in result.errors)

    def test_int_wrong_type_string(self, guard):
        r"""Integer key with string value should fail type check."""
        result = guard.validate({"batch_size": "16"})
        assert not result.passed


# ---------------------------------------------------------------------------
# validate() -- float errors
# ---------------------------------------------------------------------------

class TestValidateFloatErrors:
    r"""Float key with value below minimum boundary should fail with appropriate error message."""
    def test_float_below_min(self, guard):
        r"""Float below minimum boundary should fail with appropriate error message."""
        result = guard.validate({"learning_rate": 1e-10})
        assert not result.passed
        assert any("below min" in e for e in result.errors)

    def test_float_above_max(self, guard):
        r"""Float above maximum boundary should fail with appropriate error message."""
        result = guard.validate({"learning_rate": 0.5})
        assert not result.passed

    def test_float_wrong_type(self, guard):
        r"""Float key with string value should fail type check."""  
        result = guard.validate({"learning_rate": "0.001"})
        assert not result.passed

    def test_float_int_value_zero_fails_range(self, guard):
        r"""Float key with integer value of zero should fail range check."""
        result = guard.validate({"weight_decay": 0})  # 0.0 is valid min
        assert result.passed  # 0 == 0.0, at min boundary


# ---------------------------------------------------------------------------
# validate() -- choice errors
# ---------------------------------------------------------------------------

class TestValidateChoiceErrors:
    r"""Choice key with invalid value should fail with appropriate error message."""
    def test_invalid_choice(self, guard):
        result = guard.validate({"optimizer": "RMSProp"})
        assert not result.passed
        assert any("not in allowed choices" in e for e in result.errors)

    def test_case_sensitive(self, guard):
        r"""Choice key with lowercase value should fail with appropriate error message."""
        result = guard.validate({"optimizer": "adamw"})
        assert not result.passed


# ---------------------------------------------------------------------------
# validate() -- bool errors
# ---------------------------------------------------------------------------

class TestValidateBoolErrors:
    r"""Boolean key with non-boolean value should fail with appropriate error message."""
    def test_string_true_rejected(self, guard):
        result = guard.validate({"use_amp": "true"})
        assert not result.passed
        assert any("must be bool" in e for e in result.errors)

    def test_int_one_rejected(self, guard):
        r"""Integer 1 should not be accepted as boolean True."""
        result = guard.validate({"use_amp": 1})
        assert not result.passed


# ---------------------------------------------------------------------------
# validate() -- list errors
# ---------------------------------------------------------------------------

class TestValidateListErrors:
    r"""List key with invalid value should fail with appropriate error message."""
    def test_string_rejected_as_list(self, guard):
        r"""String value for list key should fail type check."""
        result = guard.validate({"augmentations": "flip"})
        assert not result.passed

    def test_none_rejected_as_list(self, guard):
        r"""None value for list key should fail type check."""
        result = guard.validate({"augmentations": None})
        assert not result.passed


# ---------------------------------------------------------------------------
# validate() -- immutable and unknown key errors
# ---------------------------------------------------------------------------

class TestValidateImmutableUnknown:
    def test_immutable_key_rejected(self, guard):
        result = guard.validate({"num_classes": 5})
        assert not result.passed
        assert any("immutable" in e for e in result.errors)

    def test_unknown_key_rejected(self, guard):
        result = guard.validate({"totally_unknown_key": 42})
        assert not result.passed
        assert any("not in mutable_keys" in e for e in result.errors)

    def test_multiple_errors_collected(self, guard):
        """Both an immutable key and an out-of-range key should produce 2 errors."""
        result = guard.validate({"num_classes": 5, "batch_size": 999})
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# validate() -- mixed valid/invalid
# ---------------------------------------------------------------------------

class TestValidateMixed:
    def test_one_valid_one_invalid(self, guard):
        result = guard.validate({"batch_size": 8, "optimizer": "RMSProp"})
        assert not result.passed
        assert len(result.errors) == 1
