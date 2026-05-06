"""
Integration tests: CharacteristicTranslator → ConfigurableAugmentationPipeline.

These tests complete the pipeline that unit tests in
tests/unit/augmentation/test_characteristic_translator.py cannot exercise:
translate_from_characteristics() only assembles dicts of RangeParameter objects;
it never calls albumentations. This file calls ConfigurableAugmentationPipeline,
which runs the full chain through TransformParameterBuilder and into albumentations
constructors. Invalid parameter values (wrong ranges, misspelled keys) that pass
unit tests will fail here.

Bugs documented by xfail markers in this suite:
- TODO #31: CLAHE clip_limit=0.8 (< 1.0 min) — albumentations ValidationError
- TODO #31: ColorJitter hue=(-0.7, 0.7) (outside [-0.5, 0.5]) — albumentations ValidationError
- TODO #32: translate_from_characteristics aliases into CHARACTERISTIC_RULES — mutable alias
- TODO #33: RandomSunFlare src_radius sampled as float, albumentations requires int
- TODO #34: SafeRotate has no specific builder (method name mismatch: safe_rotate vs
  safe_rotation); generic builder passes p as a float tuple instead of a scalar float
- TODO #35: RandomSizedBBoxSafeCrop height/width — to_albumentations_format() returns float
  tuple (1024.0, 1024.0) but albumentations requires an integer
"""

from __future__ import annotations

import logging
from typing import Dict

import pytest

from augmentation.augmentation_factory import ConfigurableAugmentationPipeline
from augmentation.characteristic_translator import CharacteristicTranslator

INTENSITIES = ["low", "medium", "high"]

_CLAHE_LOW = pytest.mark.xfail(
    strict=True,
    reason="TODO #31: CLAHE clip_limit=0.8 is rejected by albumentations (requires >= 1.0); "
    "transform is silently dropped from the pipeline",
)
_CLAHE_LOW_ENV = _CLAHE_LOW

_SAFE_ROTATE = pytest.mark.xfail(
    strict=True,
    reason="TODO #34: SafeRotate has no specific builder (method is named build_safe_rotation_params "
    "but get_builder_method looks for build_safe_rotate_params); falls back to build_generic_params "
    "which passes p as a tuple (0.4, 0.4) instead of a scalar float; albumentations rejects it",
)

_BBOX_SAFE_CROP = pytest.mark.xfail(
    strict=True,
    reason="TODO #35: RandomSizedBBoxSafeCrop height/width — RangeParameter.scalar(1024) passed "
    "through to_albumentations_format() returns float tuple (1024.0, 1024.0); albumentations "
    "requires a plain integer",
)

_SUN_FLARE = pytest.mark.xfail(
    strict=True,
    reason="TODO #33: RandomSunFlare src_radius — build_random_sun_flare_params uses .sample() "
    "which returns a float; albumentations requires an integer for src_radius",
)

_SUN_FLARE_AND_COLOR_JITTER = pytest.mark.xfail(
    strict=True,
    reason="TODO #33 + TODO #31: reflective_surface/high has two simultaneous failures — "
    "RandomSunFlare src_radius float (TODO #33) and ColorJitter hue=(-0.7, 0.7) (TODO #31)",
)


def _char_param(characteristic: str, intensity: str, mark=None):
    """Build a pytest.param for the characteristic × intensity matrix."""
    p = pytest.param(characteristic, intensity, id=f"{intensity}-{characteristic}")
    return (
        p
        if mark is None
        else pytest.param(characteristic, intensity, id=f"{intensity}-{characteristic}", marks=mark)
    )


_CHARACTERISTIC_PARAMS = [
    # --- changes_shape: passes at all intensities ---
    _char_param("changes_shape", "low"),
    _char_param("changes_shape", "medium"),
    _char_param("changes_shape", "high"),
    # --- low_contrast: CLAHE clip_limit=0.8 at low ---
    _char_param("low_contrast", "low", _CLAHE_LOW),
    _char_param("low_contrast", "medium"),
    _char_param("low_contrast", "high"),
    # --- reflective_surface: RandomSunFlare float src_radius at all; ColorJitter hue at high ---
    _char_param("reflective_surface", "low", _SUN_FLARE),
    _char_param("reflective_surface", "medium", _SUN_FLARE),
    _char_param("reflective_surface", "high", _SUN_FLARE_AND_COLOR_JITTER),
    # --- partially_hidden: passes at all intensities ---
    _char_param("partially_hidden", "low"),
    _char_param("partially_hidden", "medium"),
    _char_param("partially_hidden", "high"),
    # --- moves_or_vibrates: SafeRotate routing at all intensities ---
    _char_param("moves_or_vibrates", "low", _SAFE_ROTATE),
    _char_param("moves_or_vibrates", "medium", _SAFE_ROTATE),
    _char_param("moves_or_vibrates", "high", _SAFE_ROTATE),
    # --- changes_size: RandomSizedBBoxSafeCrop float at all intensities ---
    _char_param("changes_size", "low", _BBOX_SAFE_CROP),
    _char_param("changes_size", "medium", _BBOX_SAFE_CROP),
    _char_param("changes_size", "high", _BBOX_SAFE_CROP),
    # --- semi_transparent: passes at all intensities ---
    _char_param("semi_transparent", "low"),
    _char_param("semi_transparent", "medium"),
    _char_param("semi_transparent", "high"),
    # --- similar_to_background: CLAHE clip_limit=0.8 at low ---
    _char_param("similar_to_background", "low", _CLAHE_LOW),
    _char_param("similar_to_background", "medium"),
    _char_param("similar_to_background", "high"),
    # --- multiple_objects: RandomSizedBBoxSafeCrop float at all intensities ---
    _char_param("multiple_objects", "low", _BBOX_SAFE_CROP),
    _char_param("multiple_objects", "medium", _BBOX_SAFE_CROP),
    _char_param("multiple_objects", "high", _BBOX_SAFE_CROP),
]


def _env_param(env_key: str, env_val: str, intensity: str, mark=None):
    pid = f"{intensity}-{env_key}={env_val}"
    p = pytest.param(
        {env_key: env_val},
        intensity,
        id=pid,
    )
    return (
        p
        if mark is None
        else pytest.param(
            {env_key: env_val},
            intensity,
            id=pid,
            marks=mark,
        )
    )


_ENVIRONMENT_PARAMS = [
    _env_param("lighting", "variable", "low"),
    _env_param("lighting", "variable", "medium"),
    _env_param("lighting", "variable", "high"),
    # poor lighting: CLAHE clip_limit=0.8 at low only (medium/high use >= 1.0)
    _env_param("lighting", "poor", "low", _CLAHE_LOW_ENV),
    _env_param("lighting", "poor", "medium"),
    _env_param("lighting", "poor", "high"),
    _env_param("camera", "moving", "low"),
    _env_param("camera", "moving", "medium"),
    _env_param("camera", "moving", "high"),
    # shaky camera: SafeRotate routing at all intensities
    _env_param("camera", "shaky", "low", _SAFE_ROTATE),
    _env_param("camera", "shaky", "medium", _SAFE_ROTATE),
    _env_param("camera", "shaky", "high", _SAFE_ROTATE),
    _env_param("background", "busy", "low"),
    _env_param("background", "busy", "medium"),
    _env_param("background", "busy", "high"),
    _env_param("background", "changing", "low"),
    _env_param("background", "changing", "medium"),
    _env_param("background", "changing", "high"),
    _env_param("distance", "variable", "low"),
    _env_param("distance", "variable", "medium"),
    _env_param("distance", "variable", "high"),
    # close distance: RandomSizedBBoxSafeCrop float at all intensities
    _env_param("distance", "close", "low", _BBOX_SAFE_CROP),
    _env_param("distance", "close", "medium", _BBOX_SAFE_CROP),
    _env_param("distance", "close", "high", _BBOX_SAFE_CROP),
]


@pytest.fixture(scope="module")
def translator() -> CharacteristicTranslator:
    return CharacteristicTranslator()


# ---------------------------------------------------------------------------
# Core: every characteristic × every intensity builds without dropping transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("characteristic,intensity", _CHARACTERISTIC_PARAMS)
def test_pipeline_builds_without_dropping_transforms(
    translator: CharacteristicTranslator,
    characteristic: str,
    intensity: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Every characteristic at every intensity must produce a pipeline where all
    declared transforms are successfully instantiated — none silently skipped.

    _instantiate_transform swallows ValueError/TypeError and returns None when
    albumentations rejects a parameter value. A dropped transform only appears
    in the error log; the pipeline builds with fewer transforms than declared.
    This test makes that silent failure visible and auditable.

    xfail markers on specific params document the known broken combinations
    (TODOs #31–#35). When a bug is fixed, remove its marker.
    """
    result = translator.translate_from_characteristics([characteristic], intensity=intensity)
    augmentations = result["augmentations"]
    expected = len(augmentations)

    with caplog.at_level(logging.ERROR, logger="augmentation.augmentation_factory"):
        pipeline = ConfigurableAugmentationPipeline(augmentations)

    actual = len(pipeline.pipeline.transforms)
    error_lines = [r.message for r in caplog.records if r.levelno >= logging.ERROR]

    assert actual == expected, (
        f"{characteristic!r} at {intensity!r}: expected {expected} transform(s) in pipeline "
        f"but got {actual}. Silently skipped transforms indicate invalid parameter values "
        f"rejected by albumentations.\nFactory errors logged:\n"
        + "\n".join(f"  - {msg}" for msg in error_lines)
    )


# ---------------------------------------------------------------------------
# Core: every environment rule × every intensity builds without dropping transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("environment,intensity", _ENVIRONMENT_PARAMS)
def test_environment_pipeline_builds_without_dropping_transforms(
    translator: CharacteristicTranslator,
    environment: Dict[str, str],
    intensity: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same no-silent-skip contract for environment rules."""
    result = translator.translate_from_characteristics([], environment=environment, intensity=intensity)
    augmentations = result["augmentations"]
    if not augmentations:
        pytest.skip(f"No augmentations for environment={environment} at {intensity!r}")

    expected = len(augmentations)

    with caplog.at_level(logging.ERROR, logger="augmentation.augmentation_factory"):
        pipeline = ConfigurableAugmentationPipeline(augmentations)

    actual = len(pipeline.pipeline.transforms)
    error_lines = [r.message for r in caplog.records if r.levelno >= logging.ERROR]

    assert actual == expected, (
        f"environment={environment} at {intensity!r}: expected {expected} transform(s) "
        f"but got {actual}.\nFactory errors logged:\n" + "\n".join(f"  - {msg}" for msg in error_lines)
    )


# ---------------------------------------------------------------------------
# Mutable alias regression (TODO #32)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="TODO #32: translate_from_characteristics returns direct references into "
    "CHARACTERISTIC_RULES; mutating result['augmentations'] corrupts class-level state",
)
def test_translate_result_does_not_alias_class_level_rule_dict(
    translator: CharacteristicTranslator,
) -> None:
    """
    translate_from_characteristics must return copies of the per-transform param dicts,
    not direct references into CHARACTERISTIC_RULES. Any caller that mutates the returned
    result["augmentations"] dict permanently corrupts global rule state for all subsequent
    calls in the same process (TODO #32).
    """
    result = translator.translate_from_characteristics(["changes_shape"], intensity="medium")
    augmentations = result["augmentations"]

    first_aug_type = next(iter(augmentations))
    original_keys = set(
        CharacteristicTranslator.CHARACTERISTIC_RULES["changes_shape"]
        .intensity_ranges["medium"][first_aug_type]
        .keys()
    )

    # Mutate the returned dict
    augmentations[first_aug_type]["__test_mutation__"] = object()  # type: ignore[index]

    live_keys = set(
        CharacteristicTranslator.CHARACTERISTIC_RULES["changes_shape"]
        .intensity_ranges["medium"][first_aug_type]
        .keys()
    )

    assert live_keys == original_keys, (
        "Mutating result['augmentations'] corrupted CHARACTERISTIC_RULES — "
        "translate_from_characteristics must return dict copies, not aliases. (TODO #32)"
    )
