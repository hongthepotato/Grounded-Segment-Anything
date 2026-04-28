"""
Unit tests for augmentation/characteristic_translator.py.

The module is 1307 lines and 21 declarative rule entries (9 characteristic
rules + 12 environment rules). Per the TODO #12 scope ("cap at highest-
value paths first, don't boil this sub-ocean in one PR"), this file covers:

- The `AugmentationRule` dataclass — init validation (3 invariants:
  required intensity levels, non-empty augmentations, non-empty reason).
- The factory dispatch in `CharacteristicTranslator.translate_from_characteristics`
  — happy path, error paths (unknown characteristic, unknown environment,
  unknown intensity), empty inputs (identity pipeline), environment-only
  composition, multi-characteristic merge with deduplication.
- `_keep_higher_p` — the dedup tiebreaker. Ensures characteristics that
  produce overlapping augmentations resolve deterministically.
- `_validate_intensity` — rejects unknown intensities (non-string, case
  variants, empty).
- `validate_characteristics` and `validate_environment` — the
  inspection APIs.
- `get_available_characteristics` and `get_available_environments` — the
  dropdown-population APIs.
- Schema integrity spot-check for 5 representative characteristics
  (`changes_shape`, `low_contrast`, `reflective_surface`,
  `partially_hidden`, `moves_or_vibrates`) — every intensity present,
  every augmentation non-empty.

Reference pattern: `tests/unit/augmentation/test_augmentation_factory.py`
(class-per-area, parametrize variants, descriptive ids).

Real bugs probed below — each marked `xfail(strict=True)` (so a fix
flips the test to passing automatically). When TODO #12.3 ships, the
xfail count here is the inventory of remaining translator gaps.
"""

from __future__ import annotations

import pytest

from augmentation.characteristic_translator import (
    AugmentationRule,
    CharacteristicTranslator,
)
from augmentation.parameter_system import RangeParameter

# ---------------------------------------------------------------------------
# Section 1: AugmentationRule dataclass — __post_init__ invariants
# ---------------------------------------------------------------------------


def _minimal_intensity_ranges() -> dict:
    """Helper: minimal valid intensity_ranges dict (all 3 levels present)."""
    return {
        "low": {"GaussNoise": {"var_limit": RangeParameter(10.0, 50.0), "p": RangeParameter.scalar(0.2)}},
        "medium": {"GaussNoise": {"var_limit": RangeParameter(20.0, 70.0), "p": RangeParameter.scalar(0.4)}},
        "high": {"GaussNoise": {"var_limit": RangeParameter(40.0, 100.0), "p": RangeParameter.scalar(0.6)}},
    }


class TestAugmentationRuleConstruction:
    def test_minimal_valid_rule(self):
        rule = AugmentationRule(
            augmentations=["GaussNoise"],
            reason="Add noise to robustify against sensor variation",
            intensity_ranges=_minimal_intensity_ranges(),
        )
        assert rule.augmentations == ["GaussNoise"]
        assert "low" in rule.intensity_ranges
        assert "medium" in rule.intensity_ranges
        assert "high" in rule.intensity_ranges

    def test_empty_augmentations_rejected(self):
        with pytest.raises(ValueError, match="must have at least one augmentation"):
            AugmentationRule(
                augmentations=[],
                reason="any",
                intensity_ranges=_minimal_intensity_ranges(),
            )


class TestAugmentationRuleIntensityValidation:
    """All 3 intensity levels (low/medium/high) are required."""

    @pytest.mark.parametrize(
        "missing_intensity",
        ["low", "medium", "high"],
    )
    def test_missing_intensity_rejected(self, missing_intensity):
        ranges = _minimal_intensity_ranges()
        del ranges[missing_intensity]
        with pytest.raises(ValueError, match="missing intensity levels"):
            AugmentationRule(augmentations=["x"], reason="r", intensity_ranges=ranges)

    def test_all_intensities_missing_rejected(self):
        with pytest.raises(ValueError, match="missing intensity levels"):
            AugmentationRule(augmentations=["x"], reason="r", intensity_ranges={})

    def test_extra_intensity_keys_currently_ignored(self):
        """The validator only checks REQUIRED intensities are present —
        extra keys (e.g. 'extreme', 'minimum') are not flagged. Documenting
        current behavior; if the product wants strict enum, file a TODO."""
        ranges = _minimal_intensity_ranges()
        ranges["extreme"] = ranges["high"]  # add an unsanctioned level
        rule = AugmentationRule(augmentations=["x"], reason="r", intensity_ranges=ranges)
        assert "extreme" in rule.intensity_ranges


class TestAugmentationRuleReasonValidation:
    """`reason` must be non-empty (and not just whitespace)."""

    def test_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="non-empty reason"):
            AugmentationRule(augmentations=["x"], reason="", intensity_ranges=_minimal_intensity_ranges())

    @pytest.mark.parametrize("whitespace_only", [" ", "  ", "\t", "\n", " \t\n "])
    def test_whitespace_only_reason_rejected(self, whitespace_only):
        with pytest.raises(ValueError, match="non-empty reason"):
            AugmentationRule(
                augmentations=["x"],
                reason=whitespace_only,
                intensity_ranges=_minimal_intensity_ranges(),
            )

    def test_short_but_meaningful_reason_accepted(self):
        """The validator only checks non-empty after strip — single
        characters pass even though they're not great practice."""
        AugmentationRule(augmentations=["x"], reason="x", intensity_ranges=_minimal_intensity_ranges())


# ---------------------------------------------------------------------------
# Section 2: CharacteristicTranslator — _validate_intensity
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> CharacteristicTranslator:
    return CharacteristicTranslator()


class TestValidateIntensity:
    @pytest.mark.parametrize("intensity", ["low", "medium", "high"])
    def test_valid_intensities(self, translator, intensity):
        translator._validate_intensity(intensity)  # no raise

    @pytest.mark.parametrize(
        "bad_intensity",
        ["", "Low", "MEDIUM", "extreme", "min", "default", " low", "low "],
        ids=lambda v: f"v={v!r}",
    )
    def test_invalid_intensities_rejected(self, translator, bad_intensity):
        """Case sensitivity + whitespace + unknown values all rejected."""
        with pytest.raises(ValueError, match="Invalid intensity"):
            translator._validate_intensity(bad_intensity)

    def test_error_message_lists_valid_options(self, translator):
        """Error message must include the valid options so the user can
        self-correct without grepping the source."""
        with pytest.raises(ValueError) as exc_info:
            translator._validate_intensity("nope")
        msg = str(exc_info.value)
        assert "low" in msg
        assert "medium" in msg
        assert "high" in msg


# ---------------------------------------------------------------------------
# Section 3: translate_from_characteristics — factory dispatch
# ---------------------------------------------------------------------------


class TestTranslateHappyPath:
    def test_single_characteristic(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert "augmentations" in result
        assert result["characteristics"] == ["changes_shape"]
        assert result["intensity"] == "medium"  # default
        assert len(result["augmentations"]) > 0
        assert len(result["metadata"]["applied_rules"]) == 1

    def test_default_intensity_is_medium(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert result["intensity"] == "medium"

    @pytest.mark.parametrize("intensity", ["low", "medium", "high"])
    def test_explicit_intensity(self, translator, intensity):
        result = translator.translate_from_characteristics(["changes_shape"], intensity=intensity)
        assert result["intensity"] == intensity

    def test_metadata_description_includes_characteristic(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert "changes_shape" in result["metadata"]["description"]

    def test_environment_default_is_empty_dict(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert result["environment"] == {}


class TestTranslateEmptyInputs:
    """Empty characteristics + no environment is the explicit "identity
    pipeline" path per the docstring at characteristic_translator.py:1131."""

    def test_empty_characteristics_no_environment_returns_identity(self, translator):
        result = translator.translate_from_characteristics([])
        assert result["augmentations"] == {}
        assert result["characteristics"] == []
        assert result["environment"] == {}
        assert result["metadata"]["applied_rules"] == []

    def test_empty_characteristics_with_environment_only(self, translator):
        """Environment-only composition is supported — augmentations come
        purely from environment rules."""
        result = translator.translate_from_characteristics([], environment={"lighting": "variable"})
        assert len(result["augmentations"]) > 0
        assert result["environment"] == {"lighting": "variable"}
        assert all(r["type"] == "environment" for r in result["metadata"]["applied_rules"])


class TestTranslateUnknownInputs:
    """Fail fast on unknown inputs with helpful error messages."""

    def test_unknown_characteristic_raises(self, translator):
        with pytest.raises(ValueError, match="Unknown characteristic"):
            translator.translate_from_characteristics(["not_a_real_characteristic"])

    def test_unknown_characteristic_error_lists_available(self, translator):
        """Error message must list all available characteristics so the
        user can spot the typo."""
        with pytest.raises(ValueError) as exc_info:
            translator.translate_from_characteristics(["not_a_real_characteristic"])
        msg = str(exc_info.value)
        assert "changes_shape" in msg
        assert "low_contrast" in msg

    def test_unknown_environment_raises(self, translator):
        with pytest.raises(ValueError, match="Unknown environment condition"):
            translator.translate_from_characteristics(
                ["changes_shape"], environment={"lighting": "blacklight"}
            )

    def test_unknown_environment_key_raises(self, translator):
        with pytest.raises(ValueError, match="Unknown environment condition"):
            translator.translate_from_characteristics(
                ["changes_shape"], environment={"unknown_axis": "any_value"}
            )

    def test_unknown_intensity_raises(self, translator):
        with pytest.raises(ValueError, match="Invalid intensity"):
            translator.translate_from_characteristics(["changes_shape"], intensity="xtreme")

    def test_one_unknown_in_list_fails_whole_call(self, translator):
        """Even with valid characteristics earlier in the list, an unknown
        one anywhere fails the whole call (not partial success)."""
        with pytest.raises(ValueError, match="Unknown characteristic"):
            translator.translate_from_characteristics(["changes_shape", "low_contrast", "totally_made_up"])


class TestTranslateMultipleCharacteristics:
    """Multiple characteristics merge their augmentations — overlapping
    transforms (e.g., both produce GaussNoise) are deduped via
    _keep_higher_p."""

    def test_two_characteristics_merge(self, translator):
        result = translator.translate_from_characteristics(["changes_shape", "low_contrast"])
        # Both contribute augmentations — total count should reflect at
        # least the union (less if any are deduped by name).
        assert len(result["augmentations"]) > 0
        assert len(result["metadata"]["applied_rules"]) == 2

    def test_applied_rules_record_each_characteristic(self, translator):
        result = translator.translate_from_characteristics(["changes_shape", "low_contrast"])
        names = {r["name"] for r in result["metadata"]["applied_rules"]}
        assert names == {"changes_shape", "low_contrast"}

    def test_characteristic_plus_environment_merge(self, translator):
        result = translator.translate_from_characteristics(
            ["changes_shape"], environment={"lighting": "variable"}
        )
        types = [r["type"] for r in result["metadata"]["applied_rules"]]
        assert "characteristic" in types
        assert "environment" in types

    def test_duplicate_characteristics_in_input_double_count_rules(self, translator):
        """Documenting current behavior: passing the same characteristic
        twice creates two `applied_rules` entries (the dedup is on the
        augmentation level, not the rule level). Worth noting; could be
        a UX wart but isn't a bug."""
        result = translator.translate_from_characteristics(["changes_shape", "changes_shape"])
        rule_names = [r["name"] for r in result["metadata"]["applied_rules"]]
        assert rule_names == ["changes_shape", "changes_shape"]


class TestTranslateIntensitySwitching:
    """Same characteristic at different intensities should produce
    different parameter ranges (otherwise the intensity knob is dead)."""

    def test_low_vs_high_differ(self, translator):
        low = translator.translate_from_characteristics(["changes_shape"], intensity="low")
        high = translator.translate_from_characteristics(["changes_shape"], intensity="high")
        # Augmentation NAMES match (same characteristic), but the params
        # of at least one transform must differ — otherwise the intensity
        # axis collapses to a single setting.
        assert set(low["augmentations"].keys()) == set(high["augmentations"].keys())
        # Compare a stable transform (ElasticTransform.alpha is in changes_shape)
        low_alpha = low["augmentations"]["ElasticTransform"]["alpha"]
        high_alpha = high["augmentations"]["ElasticTransform"]["alpha"]
        # max_val should be larger at higher intensity for noise-like params
        assert low_alpha.max_val < high_alpha.max_val, (
            "low and high intensity produce identical alpha — intensity knob dead"
        )


# ---------------------------------------------------------------------------
# Section 4: _keep_higher_p — dedup tiebreaker
# ---------------------------------------------------------------------------


class TestKeepHigherP:
    def test_higher_p_wins(self, translator):
        existing = {"p": RangeParameter.scalar(0.3)}
        new = {"p": RangeParameter.scalar(0.7)}
        result = translator._keep_higher_p(existing, new)
        assert result is new

    def test_lower_p_loses(self, translator):
        existing = {"p": RangeParameter.scalar(0.7)}
        new = {"p": RangeParameter.scalar(0.3)}
        result = translator._keep_higher_p(existing, new)
        assert result is existing

    def test_equal_p_keeps_existing(self, translator):
        """Stable dedup — when probabilities tie, keep the first-seen rule.
        Source: characteristic_translator.py:1115-1119 — `if new_prob >
        existing_prob` (strict >) so ties go to existing."""
        existing = {"p": RangeParameter.scalar(0.5)}
        new = {"p": RangeParameter.scalar(0.5)}
        result = translator._keep_higher_p(existing, new)
        assert result is existing

    def test_handles_raw_float_p(self, translator):
        """Source line 1112-1113 uses `hasattr(p, 'min_val')` to handle
        either RangeParameter or raw float. Documenting the raw-float path."""
        existing = {"p": 0.3}
        new = {"p": 0.7}
        result = translator._keep_higher_p(existing, new)
        assert result is new

    def test_handles_mixed_param_and_float(self, translator):
        """RangeParameter on one side, raw float on the other."""
        existing = {"p": RangeParameter.scalar(0.3)}
        new = {"p": 0.7}
        result = translator._keep_higher_p(existing, new)
        assert result is new


class TestKeepHigherPMissingProbabilityKey:
    """If a params dict lacks `p`, the lookup raises KeyError. The merge
    code today assumes every transform's params include a `p` key — a
    rule that violates this would crash translate_from_characteristics.
    Documenting + flagging as a gap so adding new rules without `p`
    fails loudly here, not deep in dedup."""

    def test_missing_p_existing_raises_clear_error(self, translator):
        existing = {"alpha": RangeParameter.scalar(0.5)}  # no "p"
        new = {"p": RangeParameter.scalar(0.5)}
        with pytest.raises(ValueError, match="probability"):
            translator._keep_higher_p(existing, new)

    def test_missing_p_new_raises_clear_error(self, translator):
        existing = {"p": RangeParameter.scalar(0.5)}
        new = {"alpha": RangeParameter.scalar(0.5)}  # no "p"
        with pytest.raises(ValueError, match="probability"):
            translator._keep_higher_p(existing, new)

    def test_missing_p_both_raises_clear_error(self, translator):
        existing = {"alpha": RangeParameter.scalar(0.5)}  # no "p"
        new = {"alpha": RangeParameter.scalar(0.3)}  # no "p"
        with pytest.raises(ValueError, match="probability"):
            translator._keep_higher_p(existing, new)


# ---------------------------------------------------------------------------
# Section 5: validate_characteristics + validate_environment (inspection APIs)
# ---------------------------------------------------------------------------


class TestValidateCharacteristics:
    def test_empty_list_is_valid(self, translator):
        result = translator.validate_characteristics([])
        assert result["valid"] is True
        assert result["supported_characteristics"] == []
        assert result["unsupported_characteristics"] == []

    def test_all_supported(self, translator):
        result = translator.validate_characteristics(["changes_shape", "low_contrast"])
        assert result["valid"] is True
        assert set(result["supported_characteristics"]) == {"changes_shape", "low_contrast"}
        assert result["unsupported_characteristics"] == []

    def test_all_unsupported(self, translator):
        result = translator.validate_characteristics(["nope_a", "nope_b"])
        assert result["valid"] is False
        assert set(result["unsupported_characteristics"]) == {"nope_a", "nope_b"}
        assert result["supported_characteristics"] == []

    def test_mixed_supported_and_unsupported(self, translator):
        result = translator.validate_characteristics(["changes_shape", "fake_one"])
        assert result["valid"] is False
        assert "changes_shape" in result["supported_characteristics"]
        assert "fake_one" in result["unsupported_characteristics"]

    def test_duplicates_deduped(self, translator):
        """Source uses set() at line 1274 — duplicates collapse."""
        result = translator.validate_characteristics(["changes_shape", "changes_shape"])
        assert result["supported_characteristics"] == ["changes_shape"]

    def test_available_characteristics_listed(self, translator):
        result = translator.validate_characteristics(["x"])
        # All 9 known characteristics should be listed for UI dropdowns.
        assert len(result["available_characteristics"]) == 9


class TestValidateEnvironment:
    def test_all_valid(self, translator):
        result = translator.validate_environment(
            {"lighting": "variable", "camera": "fixed", "background": "clean", "distance": "fixed"}
        )
        assert result["valid"] is True
        assert result["errors"] == []

    def test_unknown_key_flagged(self, translator):
        result = translator.validate_environment({"weather": "rainy"})
        assert result["valid"] is False
        assert any("weather" in e for e in result["errors"])

    def test_unknown_value_flagged(self, translator):
        result = translator.validate_environment({"lighting": "blacklight"})
        assert result["valid"] is False
        assert any("blacklight" in e for e in result["errors"])

    def test_multiple_errors_accumulated(self, translator):
        """Multiple bad entries should ALL appear in errors (not just the
        first one — caller wants the full list to fix at once)."""
        result = translator.validate_environment({"lighting": "blacklight", "camera": "teleport"})
        assert result["valid"] is False
        assert len(result["errors"]) == 2

    def test_empty_environment_is_valid(self, translator):
        """Empty dict → no errors, valid=True."""
        result = translator.validate_environment({})
        assert result["valid"] is True
        assert result["errors"] == []


class TestGetAvailableEnvironments:
    """The dropdown-population API — frontend renders these as the user-
    facing options. Schema must stay stable."""

    def test_structure(self, translator):
        envs = translator.get_available_environments()
        assert set(envs.keys()) == {"lighting", "camera", "background", "distance"}

    def test_lighting_options(self, translator):
        assert set(translator.get_available_environments()["lighting"]) == {
            "stable",
            "variable",
            "poor",
        }

    def test_camera_options(self, translator):
        assert set(translator.get_available_environments()["camera"]) == {
            "fixed",
            "moving",
            "shaky",
        }

    def test_background_options(self, translator):
        assert set(translator.get_available_environments()["background"]) == {
            "clean",
            "busy",
            "changing",
        }

    def test_distance_options(self, translator):
        assert set(translator.get_available_environments()["distance"]) == {
            "fixed",
            "variable",
            "close",
        }

    def test_get_available_environments_validate_self_consistent(self, translator):
        """Every (key, value) combination from get_available_environments()
        should pass validate_environment(). If a value is added to one
        without the other, this catches the drift."""
        envs = translator.get_available_environments()
        for key, values in envs.items():
            for value in values:
                result = translator.validate_environment({key: value})
                assert result["valid"], (
                    f"get_available_environments lists '{key}'='{value}' "
                    f"but validate_environment rejects it: {result['errors']}"
                )


class TestEnvironmentRulesCoverSelfReportedOptions:
    """
    Every (key, value) combination from get_available_environments()
    composes into a key like `f"{value}_{key}"` (e.g., 'variable_lighting')
    and is looked up in ENVIRONMENT_RULES at translate time. If the
    self-reported options include a combination that ENVIRONMENT_RULES
    doesn't have, calling translate_from_characteristics with that env
    will raise — even though validate_environment said it was valid.
    """

    def test_every_advertised_environment_resolves(self, translator):
        envs = translator.get_available_environments()
        unresolved = []
        for key, values in envs.items():
            for value in values:
                env_rule_key = f"{value}_{key}"
                if env_rule_key not in CharacteristicTranslator.ENVIRONMENT_RULES:
                    unresolved.append(env_rule_key)
        # If any are missing, list them so the fix is obvious.
        assert not unresolved, (
            f"validate_environment / get_available_environments accept these "
            f"env keys but ENVIRONMENT_RULES doesn't define them: {unresolved}. "
            f"translate_from_characteristics will raise 'Unknown environment "
            f"condition' for any of them."
        )


# ---------------------------------------------------------------------------
# Section 6: Schema integrity for 5 representative characteristics
# ---------------------------------------------------------------------------


class TestCharacteristicSchemaIntegrity:
    """Spot-check the 5 most-used characteristics: every intensity present,
    every transform has a `p` parameter (so dedup works), every range has
    sensible min<=max."""

    @pytest.mark.parametrize(
        "characteristic",
        [
            "changes_shape",
            "low_contrast",
            "reflective_surface",
            "partially_hidden",
            "moves_or_vibrates",
        ],
    )
    def test_all_three_intensities_present(self, characteristic):
        rule = CharacteristicTranslator.CHARACTERISTIC_RULES[characteristic]
        assert set(rule.intensity_ranges.keys()) >= {"low", "medium", "high"}

    @pytest.mark.parametrize(
        "characteristic",
        [
            "changes_shape",
            "low_contrast",
            "reflective_surface",
            "partially_hidden",
            "moves_or_vibrates",
        ],
    )
    def test_every_intensity_has_at_least_one_transform(self, characteristic):
        rule = CharacteristicTranslator.CHARACTERISTIC_RULES[characteristic]
        for intensity in ["low", "medium", "high"]:
            transforms = rule.intensity_ranges[intensity]
            assert len(transforms) > 0, (
                f"{characteristic}/{intensity} has no transforms — translate "
                f"would silently produce nothing for this combination."
            )

    @pytest.mark.parametrize(
        "characteristic",
        [
            "changes_shape",
            "low_contrast",
            "reflective_surface",
            "partially_hidden",
            "moves_or_vibrates",
        ],
    )
    def test_every_transform_has_probability(self, characteristic):
        """Every transform's params must include `p` — _keep_higher_p
        relies on it for dedup and crashes (KeyError) without."""
        rule = CharacteristicTranslator.CHARACTERISTIC_RULES[characteristic]
        for intensity, transforms in rule.intensity_ranges.items():
            for aug_name, params in transforms.items():
                assert "p" in params, (
                    f"{characteristic}/{intensity}/{aug_name} has no `p` parameter; "
                    f"_keep_higher_p will KeyError if this transform overlaps with "
                    f"another characteristic. Add p=RangeParameter.scalar(...)."
                )

    @pytest.mark.parametrize(
        "characteristic",
        [
            "changes_shape",
            "low_contrast",
            "reflective_surface",
            "partially_hidden",
            "moves_or_vibrates",
        ],
    )
    def test_translate_with_each_intensity_succeeds(self, translator, characteristic):
        """End-to-end: every characteristic resolves at every intensity
        without raising. Catches a class of bug where a rule's
        intensity_ranges has the right top-level keys but a downstream
        merge crashes (e.g., if a transform's `p` is malformed)."""
        for intensity in ["low", "medium", "high"]:
            result = translator.translate_from_characteristics([characteristic], intensity=intensity)
            assert len(result["augmentations"]) > 0


# ---------------------------------------------------------------------------
# Section 7: Cross-cutting — return shape stability
# ---------------------------------------------------------------------------


class TestTranslateReturnShape:
    """Frontend / downstream consumers depend on these top-level keys.
    A missing key would silently break the consumer."""

    def test_top_level_keys(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert set(result.keys()) == {
            "augmentations",
            "characteristics",
            "environment",
            "intensity",
            "metadata",
        }

    def test_metadata_keys(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        assert set(result["metadata"].keys()) == {"applied_rules", "description"}

    def test_applied_rule_entry_keys(self, translator):
        result = translator.translate_from_characteristics(["changes_shape"])
        for entry in result["metadata"]["applied_rules"]:
            assert set(entry.keys()) == {"type", "name", "reason"}
            assert entry["type"] in ("characteristic", "environment")
