"""
Integration tests for ConfigurableAugmentationPipeline — pixel-apply path.

Extends test_characteristic_translator_pipeline.py, which covers only pipeline
construction. This file adds __call__ coverage: real numpy images + bboxes
flowing through the full albumentations pipeline.

Coverage:
  Scenario 1 — pixel-apply across all characteristics × intensities (30 parametrized)
  Scenario 2 — COCO bbox: coordinate space correctness, edge cases, crash pins
  Scenario 3 — stacking: dedup count, no input mutation, p-value ordering
  Scenario 4 — structural invariants: identity pipeline, empty bboxes, no orphan keys

Implementation notes:
  - RandomSizedBBoxSafeCrop (changes_size, multiple_objects) raises KeyError without
    bboxes; all pixel-apply tests supply a sample bbox.
  - RandomSizedBBoxSafeCrop resizes output to 1024×1024 — spatial shape assertions
    only check ndim and channel count, never exact H×W.
  - PiecewiseAffine UserWarning ("very slow") is suppressed — expected behaviour.
  - min_visibility=0.3 in _create_pipeline may drop bboxes with <30% post-transform
    visibility; bbox-count assertions allow reduction, never increase.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from augmentation.augmentation_factory import ConfigurableAugmentationPipeline
from augmentation.characteristic_translator import CharacteristicTranslator

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

_TRANSLATOR = CharacteristicTranslator()
_ALL_CHARACTERISTICS = list(CharacteristicTranslator.CHARACTERISTIC_RULES.keys())
_INTENSITIES = ["low", "medium", "high"]

# Characteristics containing RandomSizedBBoxSafeCrop — crash without bboxes,
# output shape is 1024×1024 when the transform fires.
_CROP_CHARACTERISTICS = {"changes_size", "multiple_objects"}


def _char_params() -> list:
    return [pytest.param(c, i, id=f"{i}-{c}") for c in _ALL_CHARACTERISTICS for i in _INTENSITIES]


@pytest.fixture(scope="module")
def rgb_image() -> np.ndarray:
    """128×128 uint8 RGB image, seeded for reproducibility."""
    return np.random.default_rng(seed=42).integers(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def sample_bbox() -> List[List[int]]:
    """One COCO [x, y, w, h] bbox safe for a 128×128 image."""
    return [[10, 10, 50, 50]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_image(image: Any) -> None:
    assert isinstance(image, np.ndarray), "image must be ndarray"
    assert image.dtype == np.uint8, f"expected uint8, got {image.dtype}"
    assert image.ndim == 3, f"expected 3D (H,W,C), got ndim={image.ndim}"
    assert image.shape[2] == 3, f"expected 3 channels, got {image.shape[2]}"
    assert int(image.min()) >= 0 and int(image.max()) <= 255


def _assert_valid_coco_bbox(bbox: Any, img_h: int, img_w: int) -> None:
    assert len(bbox) == 4, f"bbox must have 4 elements, got {len(bbox)}"
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    assert x >= 0 and y >= 0, f"top-left must be non-negative, got ({x}, {y})"
    assert w > 0 and h > 0, f"dims must be positive, got w={w}, h={h}"
    assert x + w <= img_w + 1e-6, f"bbox right edge {x + w:.2f} > image width {img_w}"
    assert y + h <= img_h + 1e-6, f"bbox bottom edge {y + h:.2f} > image height {img_h}"


def _pipeline_for(characteristic: str, intensity: str) -> ConfigurableAugmentationPipeline:
    result = _TRANSLATOR.translate_from_characteristics([characteristic], intensity=intensity)
    return ConfigurableAugmentationPipeline(result["augmentations"])


# ---------------------------------------------------------------------------
# Scenario 1 — Pixel-apply across all characteristics × intensities
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("characteristic,intensity", _char_params())
def test_pixel_apply_all_characteristics(
    characteristic: str,
    intensity: str,
    rgb_image: np.ndarray,
    sample_bbox: List[List[int]],
) -> None:
    """
    Every characteristic at every intensity must apply without error and return
    a valid uint8 3-channel image. Always passes sample_bbox to handle
    RandomSizedBBoxSafeCrop.

    Also asserts: output always carries all four keys — no orphan keys added,
    no expected keys dropped.
    """
    pipeline = _pipeline_for(characteristic, intensity)
    output = pipeline(image=rgb_image, bboxes=sample_bbox)

    _assert_valid_image(output["image"])
    assert set(output.keys()) == {"image", "masks", "keypoints", "bboxes"}, (
        f"unexpected output keys: {set(output.keys())}"
    )


# ---------------------------------------------------------------------------
# Scenario 2 — COCO bbox: coordinate space, edge cases, crash pins
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_crop_characteristics_crash_without_bboxes(rgb_image: np.ndarray) -> None:
    """
    Pins known behavior: RandomSizedBBoxSafeCrop raises KeyError when no bboxes
    are passed. This documents a hard API constraint — callers of changes_size or
    multiple_objects MUST provide bboxes.

    The transform runs with p=0.8 at high intensity, so we loop up to 10 times
    to guarantee it fires at least once (P(all misses) = 0.2^10 < 1e-6).

    If this test starts FAILING (no exception raised), the underlying behavior
    changed and the 'always pass bbox' constraint in Scenario 1 needs re-evaluation.
    """
    for characteristic in _CROP_CHARACTERISTICS:
        pipeline = _pipeline_for(characteristic, "high")
        raised = False
        for _ in range(10):
            try:
                pipeline(image=rgb_image)
            except KeyError:
                raised = True
                break
        assert raised, (
            f"{characteristic}: RandomSizedBBoxSafeCrop never raised KeyError in 10 "
            "calls without bboxes — API constraint may have changed"
        )


@pytest.mark.integration
def test_crop_output_bboxes_in_1024_coordinate_space(rgb_image: np.ndarray) -> None:
    """
    RandomSizedBBoxSafeCrop resizes output to 1024×1024. Output bboxes must be in
    the NEW coordinate space (max 1024), NOT the original 128×128 space.

    Catches a bug where bbox coordinates are returned in the pre-resize space while
    the image has been resized — coordinates would appear valid against 128×128 but
    map to wrong positions on the 1024×1024 output.

    Runs 5 times at high intensity (p=0.8) to ensure the crop fires at least once.
    """
    pipeline = _pipeline_for("changes_size", "high")
    found_crop = False

    for _ in range(5):
        output = pipeline(image=rgb_image, bboxes=[[10, 10, 50, 50]])
        h, w = output["image"].shape[:2]
        if h == 1024 and w == 1024:
            found_crop = True
            for bbox in output["bboxes"]:
                _assert_valid_coco_bbox(bbox, img_h=1024, img_w=1024)
            break

    assert found_crop, (
        "RandomSizedBBoxSafeCrop never fired in 5 runs at high intensity (p=0.8) — "
        "check if the transform is being silently dropped"
    )


@pytest.mark.integration
def test_bbox_count_never_increases(rgb_image: np.ndarray) -> None:
    """
    Output bbox count must never exceed input count. albumentations can clip and
    drop bboxes (min_visibility=0.3) but must not duplicate them.

    Tests both a shape-preserving characteristic (moves_or_vibrates) and a
    shape-changing one (changes_size) with 3 input bboxes.
    """
    for characteristic in ("moves_or_vibrates", "changes_size"):
        pipeline = _pipeline_for(characteristic, "medium")
        bboxes_in = [[5, 5, 20, 20], [40, 40, 20, 20], [80, 80, 10, 10]]

        output = pipeline(image=rgb_image, bboxes=bboxes_in)

        assert len(output["bboxes"]) <= len(bboxes_in), (
            f"{characteristic}: output has MORE bboxes than input — duplication bug"
        )


@pytest.mark.integration
def test_bbox_passthrough_coordinate_validity_after_spatial_transform(
    rgb_image: np.ndarray,
) -> None:
    """
    After ElasticTransform + PiecewiseAffine (changes_shape), surviving bboxes must
    fit within the unchanged 128×128 output image. Catches off-by-one or scale errors
    in bbox coordinate transform that would place bboxes outside the image.
    """
    pipeline = _pipeline_for("changes_shape", "high")
    h, w = rgb_image.shape[:2]

    output = pipeline(image=rgb_image, bboxes=[[10, 20, 40, 50]])

    assert output["image"].shape == (h, w, 3), "changes_shape must not resize image"
    for bbox in output["bboxes"]:
        _assert_valid_coco_bbox(bbox, h, w)


@pytest.mark.integration
def test_empty_bbox_list_does_not_crash_and_returns_empty(rgb_image: np.ndarray) -> None:
    """Passing bboxes=[] must not crash; result['bboxes'] must be empty list."""
    pipeline = _pipeline_for("low_contrast", "medium")

    output = pipeline(image=rgb_image, bboxes=[])

    _assert_valid_image(output["image"])
    assert output["bboxes"] == []


@pytest.mark.integration
def test_single_pixel_bbox_no_division_by_zero(rgb_image: np.ndarray) -> None:
    """1×1 pixel bbox must not cause division-by-zero anywhere in the pipeline."""
    pipeline = _pipeline_for("partially_hidden", "high")
    h, w = rgb_image.shape[:2]

    output = pipeline(image=rgb_image, bboxes=[[10, 10, 1, 1]])

    _assert_valid_image(output["image"])
    for bbox in output["bboxes"]:
        _assert_valid_coco_bbox(bbox, h, w)


# ---------------------------------------------------------------------------
# Scenario 3 — Stacking: dedup count, p-ordering, no input mutation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stacked_dedup_reduces_overlapping_transforms() -> None:
    """
    low_contrast and similar_to_background share CLAHE and Sharpen.
    Stacking both must produce 4 unique transforms (CLAHE + RandomBrightnessContrast +
    Sharpen + RandomGamma), NOT 6 (no duplicates).

    Catches: wrong merge approach (manual dict merge with ** would overwrite instead
    of dedup, silently discarding one set of params).
    """
    result = _TRANSLATOR.translate_from_characteristics(
        ["low_contrast", "similar_to_background"],
        intensity="medium",
    )
    aug_types = set(result["augmentations"].keys())

    assert aug_types == {"CLAHE", "RandomBrightnessContrast", "Sharpen", "RandomGamma"}, (
        f"expected 4 unique transforms after dedup, got {aug_types}"
    )
    assert len(result["augmentations"]) == 4, (
        f"dedup must produce exactly 4 entries, got {len(result['augmentations'])}"
    )


@pytest.mark.integration
def test_high_intensity_p_strictly_greater_than_low_intensity() -> None:
    """
    For low_contrast: CLAHE p=0.2 at low, p=0.6 at high (from CHARACTERISTIC_RULES).
    The built pipeline transform must reflect this — high intensity must have a
    strictly higher p than low.

    Catches: parameter builder discarding the p field and defaulting to a constant.
    """
    import albumentations as A

    def get_clahe_p(intensity: str) -> float:
        result = _TRANSLATOR.translate_from_characteristics(["low_contrast"], intensity=intensity)
        pipeline = ConfigurableAugmentationPipeline(result["augmentations"])
        for t in pipeline.pipeline.transforms:
            if isinstance(t, A.CLAHE):
                return t.p
        raise AssertionError(f"CLAHE not found in low_contrast pipeline at intensity={intensity!r}")

    p_low = get_clahe_p("low")
    p_high = get_clahe_p("high")

    assert p_high > p_low, (
        f"high intensity must have higher CLAHE p than low: got p_low={p_low}, p_high={p_high}"
    )


@pytest.mark.integration
def test_input_image_not_mutated_by_pipeline(rgb_image: np.ndarray, sample_bbox: List[List[int]]) -> None:
    """
    Calling pipeline(image=rgb_image) must not modify the original array in-place.
    Catches pipelines that call np operations without copy, mutating the caller's data.
    """
    original = rgb_image.copy()
    pipeline = _pipeline_for("reflective_surface", "high")

    pipeline(image=rgb_image, bboxes=sample_bbox)

    assert np.array_equal(rgb_image, original), "pipeline must not mutate the input image array in-place"


@pytest.mark.integration
def test_all_nine_characteristics_stacked(
    rgb_image: np.ndarray,
    sample_bbox: List[List[int]],
) -> None:
    """
    All 9 characteristics stacked at medium intensity. Single translate call — NOT
    nine separate calls merged manually (which would bypass _keep_higher_p dedup).

    Asserts: no Compose conflict, output is valid, total unique transforms is less
    than the sum of per-characteristic counts (dedup fired).
    """
    per_char_counts = [
        len(_TRANSLATOR.translate_from_characteristics([c], intensity="medium")["augmentations"])
        for c in _ALL_CHARACTERISTICS
    ]
    sum_naive = sum(per_char_counts)

    result = _TRANSLATOR.translate_from_characteristics(_ALL_CHARACTERISTICS, intensity="medium")
    stacked_count = len(result["augmentations"])

    assert stacked_count < sum_naive, (
        f"stacking all 9 characteristics should reduce transform count via dedup: "
        f"naive sum={sum_naive}, stacked={stacked_count}"
    )

    pipeline = ConfigurableAugmentationPipeline(result["augmentations"])
    output = pipeline(image=rgb_image, bboxes=sample_bbox)
    _assert_valid_image(output["image"])


@pytest.mark.integration
def test_pipeline_stable_across_repeated_calls(
    rgb_image: np.ndarray,
    sample_bbox: List[List[int]],
) -> None:
    """
    The same pipeline instance called 5 times must return valid uint8 RGB each time.
    Catches: mutable state inside the pipeline that accumulates across calls and
    corrupts dtype or channel count on subsequent invocations.
    """
    pipeline = _pipeline_for("semi_transparent", "medium")

    for _ in range(5):
        output = pipeline(image=rgb_image, bboxes=sample_bbox)
        _assert_valid_image(output["image"])


# ---------------------------------------------------------------------------
# Scenario 4 — Structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_identity_pipeline_returns_pixel_identical_image(rgb_image: np.ndarray) -> None:
    """
    Empty augmentations dict builds an identity (no-op) pipeline. Output image
    must be pixel-identical to the input.

    Catches: Compose wrapper accidentally modifying the image even with no transforms,
    or dtype conversion on the round-trip.
    """
    pipeline = ConfigurableAugmentationPipeline({})

    output = pipeline(image=rgb_image)

    assert np.array_equal(output["image"], rgb_image), (
        "identity pipeline (no augmentations) must return pixel-identical image"
    )
    assert output["bboxes"] == [] and output["masks"] == [] and output["keypoints"] == []


@pytest.mark.integration
def test_all_characteristics_produce_at_least_one_transform() -> None:
    """
    Every characteristic at every intensity must produce a non-empty augmentations
    dict. An empty dict means the translator silently dropped the characteristic —
    the pipeline becomes a no-op when it should be active.
    """
    for characteristic in _ALL_CHARACTERISTICS:
        for intensity in _INTENSITIES:
            result = _TRANSLATOR.translate_from_characteristics([characteristic], intensity=intensity)
            assert len(result["augmentations"]) > 0, (
                f"{characteristic!r} at {intensity!r} produced empty augmentations dict"
            )


@pytest.mark.integration
def test_masks_passed_through_pipeline(rgb_image: np.ndarray) -> None:
    """
    Passing a real binary mask exercises the has_masks=True branch in __call__:
    stack → augment → unstack. Output must contain one mask whose shape matches
    the output image. Exercises the ndarray-unstack path at __call__:508-512.
    """
    pipeline = _pipeline_for("low_contrast", "medium")
    h, w = rgb_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    output = pipeline(image=rgb_image, masks=[mask])

    assert isinstance(output["masks"], list), "output['masks'] must be a list"
    assert len(output["masks"]) == 1, "one mask in → one mask out"
    returned_mask = output["masks"][0]
    assert isinstance(returned_mask, np.ndarray), "mask must be ndarray"
    assert returned_mask.ndim == 2, f"mask must be 2D, got ndim={returned_mask.ndim}"
    out_h, out_w = output["image"].shape[:2]
    assert returned_mask.shape == (out_h, out_w), (
        f"mask {returned_mask.shape} must match image {(out_h, out_w)}"
    )


@pytest.mark.integration
def test_keypoints_passed_through_pipeline(rgb_image: np.ndarray) -> None:
    """
    Passing keypoints in [[x, y], ...] format exercises the has_keypoints=True
    branch in __call__. low_contrast is color-only so coordinates are unchanged;
    remove_invisible=False means no keypoints are dropped.
    """
    pipeline = _pipeline_for("low_contrast", "medium")
    keypoints_in = [[20, 20], [50, 50], [80, 30]]

    output = pipeline(image=rgb_image, keypoints=keypoints_in)

    assert isinstance(output["keypoints"], list), "output['keypoints'] must be a list"
    assert len(output["keypoints"]) == len(keypoints_in), (
        f"keypoint count must not change: in={len(keypoints_in)}, out={len(output['keypoints'])}"
    )


@pytest.mark.integration
def test_masks_returned_empty_when_not_provided(
    rgb_image: np.ndarray,
    sample_bbox: List[List[int]],
) -> None:
    """
    When masks and keypoints are not passed, output['masks'] and output['keypoints']
    must be empty lists — not None, not missing, not a stale value from a prior call.
    Catches: output dict built from input references rather than fresh empty lists.
    """
    pipeline = _pipeline_for("low_contrast", "medium")

    output = pipeline(image=rgb_image, bboxes=sample_bbox)

    assert output["masks"] == [], f"expected [], got {output['masks']!r}"
    assert output["keypoints"] == [], f"expected [], got {output['keypoints']!r}"
