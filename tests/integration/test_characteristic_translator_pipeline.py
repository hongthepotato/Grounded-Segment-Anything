"""
Integration tests: CharacteristicTranslator → ConfigurableAugmentationPipeline.

These tests complete the pipeline that unit tests in
tests/unit/augmentation/test_characteristic_translator.py cannot exercise:
translate_from_characteristics() only assembles dicts of RangeParameter objects;
it never calls albumentations. This file calls ConfigurableAugmentationPipeline,
which runs the full chain through TransformParameterBuilder and into albumentations
constructors. Invalid parameter values (wrong ranges, misspelled keys) that pass
unit tests will fail here.
"""

from __future__ import annotations

import logging
from typing import Dict

import pytest

from augmentation.augmentation_factory import ConfigurableAugmentationPipeline
from augmentation.characteristic_translator import CharacteristicTranslator

INTENSITIES = ["low", "medium", "high"]


def _char_param(characteristic: str, intensity: str):
    """Build a pytest.param for the characteristic × intensity matrix."""
    return pytest.param(characteristic, intensity, id=f"{intensity}-{characteristic}")


_CHARACTERISTIC_PARAMS = [
    _char_param("changes_shape", "low"),
    _char_param("changes_shape", "medium"),
    _char_param("changes_shape", "high"),
    _char_param("low_contrast", "low"),
    _char_param("low_contrast", "medium"),
    _char_param("low_contrast", "high"),
    _char_param("reflective_surface", "low"),
    _char_param("reflective_surface", "medium"),
    _char_param("reflective_surface", "high"),
    _char_param("partially_hidden", "low"),
    _char_param("partially_hidden", "medium"),
    _char_param("partially_hidden", "high"),
    _char_param("moves_or_vibrates", "low"),
    _char_param("moves_or_vibrates", "medium"),
    _char_param("moves_or_vibrates", "high"),
    _char_param("changes_size", "low"),
    _char_param("changes_size", "medium"),
    _char_param("changes_size", "high"),
    _char_param("semi_transparent", "low"),
    _char_param("semi_transparent", "medium"),
    _char_param("semi_transparent", "high"),
    _char_param("similar_to_background", "low"),
    _char_param("similar_to_background", "medium"),
    _char_param("similar_to_background", "high"),
    _char_param("multiple_objects", "low"),
    _char_param("multiple_objects", "medium"),
    _char_param("multiple_objects", "high"),
]


def _env_param(env_key: str, env_val: str, intensity: str):
    pid = f"{intensity}-{env_key}={env_val}"
    return pytest.param({env_key: env_val}, intensity, id=pid)


_ENVIRONMENT_PARAMS = [
    _env_param("lighting", "variable", "low"),
    _env_param("lighting", "variable", "medium"),
    _env_param("lighting", "variable", "high"),
    _env_param("lighting", "poor", "low"),
    _env_param("lighting", "poor", "medium"),
    _env_param("lighting", "poor", "high"),
    _env_param("lighting", "stable", "low"),
    _env_param("lighting", "stable", "medium"),
    _env_param("lighting", "stable", "high"),
    _env_param("camera", "moving", "low"),
    _env_param("camera", "moving", "medium"),
    _env_param("camera", "moving", "high"),
    _env_param("camera", "shaky", "low"),
    _env_param("camera", "shaky", "medium"),
    _env_param("camera", "shaky", "high"),
    _env_param("camera", "fixed", "low"),
    _env_param("camera", "fixed", "medium"),
    _env_param("camera", "fixed", "high"),
    _env_param("background", "busy", "low"),
    _env_param("background", "busy", "medium"),
    _env_param("background", "busy", "high"),
    _env_param("background", "changing", "low"),
    _env_param("background", "changing", "medium"),
    _env_param("background", "changing", "high"),
    _env_param("background", "clean", "low"),
    _env_param("background", "clean", "medium"),
    _env_param("background", "clean", "high"),
    _env_param("distance", "variable", "low"),
    _env_param("distance", "variable", "medium"),
    _env_param("distance", "variable", "high"),
    _env_param("distance", "close", "low"),
    _env_param("distance", "close", "medium"),
    _env_param("distance", "close", "high"),
    _env_param("distance", "fixed", "low"),
    _env_param("distance", "fixed", "medium"),
    _env_param("distance", "fixed", "high"),
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
# Mutable alias regression (was TODO #32)
# ---------------------------------------------------------------------------


def test_translate_result_does_not_alias_class_level_rule_dict(
    translator: CharacteristicTranslator,
) -> None:
    """
    translate_from_characteristics must return copies of the per-transform param dicts,
    not direct references into CHARACTERISTIC_RULES. Mutating the returned
    result["augmentations"] dict must not corrupt global rule state for subsequent calls.
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
        "translate_from_characteristics must return dict copies, not aliases."
    )
