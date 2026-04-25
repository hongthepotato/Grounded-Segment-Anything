"""
Error-path coverage for ConfigurableAugmentationPipeline._validate_bboxes.

Context: the validator was rewritten in the ci-and-tests PR (commit 5f501b8)
to use COCO format [x, y, w, h] after a pascal-voc misinterpretation bug
was discovered. Happy-path validation is exercised by
tests/unit/test_augmentation.py::test_pipeline_application. This file covers
every error branch of the validator itself.

Error contract (from augmentation/augmentation_factory.py::_validate_bboxes):
- TypeError for structural / type errors: wrong container type, wrong
  element types, non-integer coordinates (incl. string-floats).
- ValueError for value errors: wrong tuple length, non-positive dimensions,
  out-of-bounds coordinates, bbox extending beyond the image.

Validator is called statically with self=None. The method body uses no
instance attributes; `self` is only there because it's an instance method.
Skipping a full pipeline construction keeps these tests fast and isolated.
"""

import numpy as np
import pytest

from augmentation.augmentation_factory import ConfigurableAugmentationPipeline

# Image dims for bounds tests. Chosen to be realistic, not round, so bugs
# that accidentally hardcode dimensions surface clearly.
IMG_H = 480
IMG_W = 640

# Alias: _validate_bboxes as a static callable.
_validate = ConfigurableAugmentationPipeline._validate_bboxes


# ---------------------------------------------------------------------------
# Happy-path variants — these must NOT raise
# ---------------------------------------------------------------------------


class TestValidInputs:
    def test_none_is_valid(self):
        """`None` is the sentinel for 'no bboxes', per docstring."""
        _validate(None, None, IMG_H, IMG_W)

    def test_empty_list_is_valid(self):
        """Empty list is the other 'no bboxes' representation."""
        _validate(None, [], IMG_H, IMG_W)

    def test_single_bbox_is_valid(self):
        _validate(None, [[100, 100, 50, 50]], IMG_H, IMG_W)

    def test_multiple_bboxes_are_valid(self):
        _validate(None, [[0, 0, 10, 10], [100, 100, 50, 50]], IMG_H, IMG_W)

    def test_bbox_spanning_full_image(self):
        """x+w == image_w and y+h == image_h are both allowed (inclusive)."""
        _validate(None, [[0, 0, IMG_W, IMG_H]], IMG_H, IMG_W)

    @pytest.mark.parametrize(
        "coord_example",
        [
            pytest.param(["100", "100", "50", "50"], id="string-integers"),
            pytest.param(
                [np.int32(100), np.int32(100), np.int32(50), np.int32(50)],
                id="numpy-int32",
            ),
            pytest.param(
                [np.int64(100), np.int64(100), np.int64(50), np.int64(50)],
                id="numpy-int64",
            ),
        ],
    )
    def test_accepted_coord_types(self, coord_example):
        """String-integers and numpy integer types are explicitly coerced to int."""
        _validate(None, [coord_example], IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Container-level type errors (bboxes itself)
# ---------------------------------------------------------------------------


class TestInvalidContainer:
    @pytest.mark.parametrize(
        "non_list",
        [
            pytest.param("not a list", id="string"),
            pytest.param(42, id="int"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param((1, 2, 3), id="tuple"),
            pytest.param(np.array([[1, 2, 3, 4]]), id="ndarray"),
        ],
    )
    def test_non_list_raises_typeerror(self, non_list):
        with pytest.raises(TypeError, match="bboxes must be a list"):
            _validate(None, non_list, IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Per-bbox structure errors
# ---------------------------------------------------------------------------


class TestInvalidBboxElement:
    @pytest.mark.parametrize(
        "non_list_bbox",
        [
            pytest.param((1, 2, 3, 4), id="tuple"),
            pytest.param("1,2,3,4", id="string"),
            pytest.param(None, id="None"),
            pytest.param(np.array([1, 2, 3, 4]), id="ndarray"),
        ],
    )
    def test_non_list_bbox_element_raises_typeerror(self, non_list_bbox):
        with pytest.raises(TypeError, match="bbox 1 must be a list"):
            _validate(None, [non_list_bbox], IMG_H, IMG_W)

    def test_empty_bbox_raises_valueerror(self):
        """`[[]]` is malformed per docstring contract."""
        with pytest.raises(ValueError, match="bbox 1 must have exactly 4 elements, got 0"):
            _validate(None, [[]], IMG_H, IMG_W)

    @pytest.mark.parametrize(
        "wrong_len_bbox,expected_count",
        [
            pytest.param([1, 2], 2, id="two-elements"),
            pytest.param([1, 2, 3], 3, id="three-elements"),
            pytest.param([1, 2, 3, 4, 5], 5, id="five-elements"),
        ],
    )
    def test_wrong_length_bbox_raises_valueerror(self, wrong_len_bbox, expected_count):
        with pytest.raises(
            ValueError,
            match=rf"bbox 1 must have exactly 4 elements, got {expected_count}",
        ):
            _validate(None, [wrong_len_bbox], IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Per-coordinate type errors
# ---------------------------------------------------------------------------


class TestInvalidCoordinateTypes:
    @pytest.mark.parametrize(
        "non_int_coord",
        [
            pytest.param(1.5, id="float"),
            pytest.param([1], id="list"),
            pytest.param({"x": 1}, id="dict"),
            pytest.param(None, id="None"),
        ],
    )
    def test_non_int_coord_raises_typeerror(self, non_int_coord):
        """Non-integer-family coordinates (floats, containers, None) rejected."""
        with pytest.raises(TypeError, match="must be integers"):
            _validate(None, [[non_int_coord, 10, 10, 10]], IMG_H, IMG_W)

    @pytest.mark.parametrize(
        "string_float",
        [
            pytest.param("1.0", id="plain-decimal"),
            pytest.param("1.5e3", id="decimal-scientific"),
            pytest.param("1e10", id="pure-scientific"),
        ],
    )
    def test_string_float_coord_raises_typeerror(self, string_float):
        """Strings containing '.' or 'e' are rejected as string-floats."""
        with pytest.raises(TypeError, match="string-float"):
            _validate(None, [[string_float, 10, 10, 10]], IMG_H, IMG_W)

    def test_non_numeric_string_raises_typeerror(self):
        """
        A string with no '.' or 'e' attempts int() conversion and fails,
        surfacing as TypeError via the except-re-raise path.
        """
        with pytest.raises(TypeError, match="must be integers or string-integers"):
            _validate(None, [["abc", 10, 10, 10]], IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Width/height value errors
# ---------------------------------------------------------------------------


class TestInvalidDimensions:
    @pytest.mark.parametrize("w", [0, -5, -100])
    def test_non_positive_width_raises_valueerror(self, w):
        with pytest.raises(ValueError, match=r"width \(.+\) must be > 0"):
            _validate(None, [[10, 10, w, 10]], IMG_H, IMG_W)

    @pytest.mark.parametrize("h", [0, -5, -100])
    def test_non_positive_height_raises_valueerror(self, h):
        with pytest.raises(ValueError, match=r"height \(.+\) must be > 0"):
            _validate(None, [[10, 10, 10, h]], IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Out-of-bounds coordinate errors
# ---------------------------------------------------------------------------


class TestOutOfBounds:
    @pytest.mark.parametrize(
        "x,y",
        [
            pytest.param(-1, 10, id="negative-x"),
            pytest.param(10, -1, id="negative-y"),
            pytest.param(IMG_W, 10, id="x-equals-image-width"),
            pytest.param(10, IMG_H, id="y-equals-image-height"),
            pytest.param(IMG_W + 10, 10, id="x-beyond-width"),
            pytest.param(10, IMG_H + 10, id="y-beyond-height"),
        ],
    )
    def test_top_left_out_of_bounds_raises_valueerror(self, x, y):
        """Top-left corner must satisfy 0 <= x < image_w AND 0 <= y < image_h."""
        with pytest.raises(ValueError, match="top-left .+ out of bounds"):
            _validate(None, [[x, y, 10, 10]], IMG_H, IMG_W)

    def test_bbox_extends_beyond_image_width(self):
        """x+w > image_w is rejected."""
        with pytest.raises(ValueError, match="extends beyond image"):
            _validate(None, [[100, 100, IMG_W, 10]], IMG_H, IMG_W)

    def test_bbox_extends_beyond_image_height(self):
        """y+h > image_h is rejected."""
        with pytest.raises(ValueError, match="extends beyond image"):
            _validate(None, [[100, 100, 10, IMG_H]], IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Index reporting for multi-bbox inputs
# ---------------------------------------------------------------------------


class TestErrorIndexing:
    def test_error_reports_1indexed_position_of_invalid_bbox(self):
        """First invalid bbox surfaces with its 1-indexed position in the error."""
        bboxes = [
            [0, 0, 10, 10],
            [10, 10, 10, 10],
            [20, 20, 10, 10],
            [30, 30, 0, 10],  # 4th bbox: width = 0 → should raise
        ]
        with pytest.raises(ValueError, match=r"bbox 4 invalid: width"):
            _validate(None, bboxes, IMG_H, IMG_W)
