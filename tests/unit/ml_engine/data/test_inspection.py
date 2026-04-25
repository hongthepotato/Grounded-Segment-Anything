"""
Unit tests for ml_engine.data.inspection module.

Tests:
- inspect_dataset: Analyze COCO data structure and annotations
- detect_annotation_mode: Detect original annotation type
- get_required_models_from_mode: Map mode to required models
"""

import pytest

from core.constants import (
    GROUNDING_DINO,
    MODE_COMBINED,
    MODE_DETECTION,
    MODE_SEGMENTATION,
    SAM,
)
from ml_engine.data.inspection import (
    detect_annotation_mode,
    get_required_models_from_mode,
    inspect_dataset,
)


class TestInspectDataset:
    """Tests for inspect_dataset function."""

    def test_returns_expected_keys(self, valid_coco_data_combined):
        """Inspection result contains all expected keys."""
        info = inspect_dataset(valid_coco_data_combined)

        expected_keys = [
            "has_boxes",
            "has_masks",
            "num_classes",
            "class_mapping",
            "category_id_to_index",
            "index_to_category_id",
            "num_images",
            "num_annotations",
            "annotation_mode",
            "class_counts",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_combined_data_detection(self, valid_coco_data_combined):
        """Detects both boxes and masks in combined data."""
        info = inspect_dataset(valid_coco_data_combined)

        assert info["has_boxes"] is True
        assert info["has_masks"] is True

    def test_boxes_only_detection(self, valid_coco_data_boxes_only):
        """Detects only boxes when masks are absent."""
        info = inspect_dataset(valid_coco_data_boxes_only)

        assert info["has_boxes"] is True
        assert info["has_masks"] is False

    def test_masks_only_detection(self, valid_coco_data_masks_only):
        """Detects only masks when boxes are absent."""
        info = inspect_dataset(valid_coco_data_masks_only)

        assert info["has_boxes"] is False
        assert info["has_masks"] is True

    def test_num_classes(self, valid_coco_data_combined):
        """Correctly counts number of classes."""
        info = inspect_dataset(valid_coco_data_combined)

        expected_classes = len(valid_coco_data_combined["categories"])
        assert info["num_classes"] == expected_classes

    def test_class_mapping(self, valid_coco_data_combined):
        """Creates correct class_mapping (id -> name)."""
        info = inspect_dataset(valid_coco_data_combined)

        # Should map category_id -> category_name
        for cat in valid_coco_data_combined["categories"]:
            assert info["class_mapping"][cat["id"]] == cat["name"]

    def test_category_id_to_index(self, valid_coco_data_sparse_ids):
        """Creates correct category_id_to_index mapping."""
        info = inspect_dataset(valid_coco_data_sparse_ids)

        # Sparse IDs (1, 90) should map to sequential indices (0, 1)
        assert len(info["category_id_to_index"]) == 2
        assert set(info["category_id_to_index"].values()) == {0, 1}
        assert set(info["category_id_to_index"].keys()) == {1, 90}

    def test_index_to_category_id(self, valid_coco_data_sparse_ids):
        """Creates correct index_to_category_id mapping."""
        info = inspect_dataset(valid_coco_data_sparse_ids)

        # Sequential indices should map back to original IDs
        assert len(info["index_to_category_id"]) == 2
        assert 0 in info["index_to_category_id"]
        assert 1 in info["index_to_category_id"]
        assert info["index_to_category_id"][0] == 1
        assert info["index_to_category_id"][1] == 90

    def test_num_images(self, valid_coco_data_combined):
        """Correctly counts images."""
        info = inspect_dataset(valid_coco_data_combined)

        assert info["num_images"] == len(valid_coco_data_combined["images"])

    def test_num_annotations(self, valid_coco_data_combined):
        """Correctly counts annotations."""
        info = inspect_dataset(valid_coco_data_combined)

        assert info["num_annotations"] == len(valid_coco_data_combined["annotations"])

    def test_class_counts(self, valid_coco_data_combined):
        """Correctly counts annotations per class."""
        info = inspect_dataset(valid_coco_data_combined)

        # Verify counts match manual counting
        expected_counts = {0: 2, 1: 2}

        assert info["class_counts"] == expected_counts

    def test_annotation_mode_combined(self, valid_coco_data_combined):
        """Annotation mode is 'combined' when both present."""
        info = inspect_dataset(valid_coco_data_combined)
        # Note: The constant value may differ from 'combined'
        assert info["annotation_mode"] == "DETECTION_AND_SEGMENTATION"

    def test_no_annotations_raises(self):
        """Raises error when no valid annotations found."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0}  # No bbox or segmentation
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        with pytest.raises(KeyError, match="No valid annotations found in dataset"):
            inspect_dataset(data)


class TestDetectAnnotationMode:
    """Tests for detect_annotation_mode function."""

    def test_detection_mode(self, valid_coco_data_boxes_only):
        """Detects 'detection' mode for boxes-only data."""
        mode = detect_annotation_mode(valid_coco_data_boxes_only)
        assert mode == MODE_DETECTION

    def test_segmentation_mode(self, valid_coco_data_masks_only):
        """Detects 'segmentation' mode for masks-only data."""
        mode = detect_annotation_mode(valid_coco_data_masks_only)
        assert mode == MODE_SEGMENTATION

    def test_combined_mode(self, valid_coco_data_combined):
        """Detects 'combined' mode for both boxes and masks."""
        mode = detect_annotation_mode(valid_coco_data_combined)
        assert mode == MODE_COMBINED

    def test_no_annotations_raises(self):
        """Raises ValueError when no valid annotations."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0}  # Empty
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        with pytest.raises(ValueError, match="No valid annotations found in dataset"):
            detect_annotation_mode(data)

    def test_none_bbox_not_counted(self):
        """Annotations with None bbox are not counted as having boxes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": None,
                    "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        assert detect_annotation_mode(data) == MODE_SEGMENTATION

    def test_empty_bbox_not_counted(self):
        """Annotations with empty bbox are not counted as having boxes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": [],
                    "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        assert detect_annotation_mode(data) == MODE_SEGMENTATION

    def test_none_segmentation_not_counted(self):
        """Annotations with None segmentation are not counted as having masks."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "segmentation": None}
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        assert detect_annotation_mode(data) == MODE_DETECTION

    def test_empty_segmentation_not_counted(self):
        """Annotations with empty segmentation are not counted as having masks."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "segmentation": []}
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        assert detect_annotation_mode(data) == MODE_DETECTION

    def test_single_bbox_counted_as_detection(self):
        """Multiple annotations but only one has valid bbox."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]},
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": None},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        assert detect_annotation_mode(data) == MODE_DETECTION

    def test_single_segmentation_counted_as_segmentation(self):
        """Multiple annotations but only one has valid segmentation."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[10, 10, 20, 10, 20, 20]]},
                {"id": 1, "image_id": 0, "category_id": 0, "segmentation": None},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        assert detect_annotation_mode(data) == MODE_SEGMENTATION


class TestGetRequiredModelsFromMode:
    """Tests for get_required_models_from_mode function."""

    def test_detection_mode_returns_dino(self):
        """Detection mode requires Grounding DINO."""
        models = get_required_models_from_mode(MODE_DETECTION)

        assert GROUNDING_DINO in models
        assert SAM not in models

    def test_segmentation_mode_returns_both_models(self):
        """Segmentation mode requires SAM for masks and GroundingDINO for bbox prompts.

        GroundingDINO is co-loaded because SAM needs box prompts to generate masks,
        and the normalization step auto-generates boxes from mask contours so DINO
        has valid training targets. See inspection.py::get_required_models_from_mode
        docstring for the rationale.
        """
        models = get_required_models_from_mode(MODE_SEGMENTATION)

        assert SAM in models
        assert GROUNDING_DINO in models
        assert len(models) == 2

    def test_combined_mode_returns_both(self):
        """Combined mode requires both DINO and SAM."""
        models = get_required_models_from_mode(MODE_COMBINED)

        assert GROUNDING_DINO in models
        assert SAM in models
        assert len(models) == 2

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown annotation mode: unknown_mode"):
            get_required_models_from_mode("unknown_mode")


class TestInspectDatasetEdgeCases:
    """Hard edge cases for inspect_dataset function."""

    def test_large_category_ids(self):
        """Handles very large category IDs (like COCO-80 format)."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 9999, "bbox": [10, 10, 20, 20]},
                {"id": 1, "image_id": 0, "category_id": 123456, "bbox": [30, 30, 20, 20]},
            ],
            "categories": [{"id": 9999, "name": "class_a"}, {"id": 123456, "name": "class_b"}],
        }

        info = inspect_dataset(data)
        assert info["num_classes"] == 2
        assert info["category_id_to_index"][9999] == 0
        assert info["category_id_to_index"][123456] == 1
        assert info["index_to_category_id"][0] == 9999
        assert info["index_to_category_id"][1] == 123456

    def test_unicode_category_names(self):
        """Category names with unicode characters should be handled."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]},
                {"id": 1, "image_id": 0, "category_id": 1, "bbox": [30, 30, 20, 20]},
            ],
            "categories": [
                {"id": 0, "name": "耳朵"},
                {"id": 1, "name": "дефект"},
            ],
        }

        info = inspect_dataset(data)
        assert info["class_mapping"][0] == "耳朵"
        assert info["class_mapping"][1] == "дефект"

    def test_empty_string_category_name(self):
        """Category with empty string name should still work."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]}],
            "categories": [{"id": 0, "name": ""}],
        }

        info = inspect_dataset(data)
        assert info["class_mapping"][0] == ""

    def test_single_image_no_annotations(self):
        """Image without any annotations should still be counted."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 0, "name": "cat"}],
        }

        with pytest.raises(KeyError, match="No valid annotations found in dataset"):
            inspect_dataset(data)

    def test_very_long_category_names(self):
        """Category names that are extremely long should be handled."""
        long_name = "a" * 10000
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]}],
            "categories": [{"id": 0, "name": long_name}],
        }

        info = inspect_dataset(data)
        assert info["class_mapping"][0] == long_name


class TestDetectAnnotationModeEdgeCases:
    """Hard edge cases for detect_annotation_mode function."""

    def test_rle_segmentation_format(self):
        """RLE format segmentation (COCO crowd format) should be detected."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": {"counts": [10, 20, 30], "size": [100, 100]},  # RLE format
                    "iscrowd": 1,
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }

        mode = detect_annotation_mode(data)
        assert mode == MODE_SEGMENTATION


class TestWeirdButValidCOCOFormats:
    """Tests for weird but technically valid COCO data formats."""

    def test_minimal_fields_only(self, weird_coco_minimal_fields):
        """COCO with only required fields (no optional fields)."""
        info = inspect_dataset(weird_coco_minimal_fields)

        assert info["has_boxes"] is True
        assert info["num_classes"] == 1
        assert info["num_images"] == 1
        assert info["annotation_mode"] == "DETECTION_ONLY"
        assert info["class_mapping"][0] == "cat"
        assert info["num_annotations"] == 1

    def test_extra_custom_fields(self, weird_coco_extra_fields):
        """COCO with extra custom fields should be ignored gracefully."""
        info = inspect_dataset(weird_coco_extra_fields)

        assert info["has_boxes"] is True
        assert info["has_masks"] is False
        assert info["class_mapping"][0] == "cat"
        assert info["num_annotations"] == 1
        assert info["num_classes"] == 1
        assert info["annotation_mode"] == "DETECTION_ONLY"

    def test_unordered_ids(self, weird_coco_unordered):
        """COCO with non-sequential, unordered IDs."""
        info = inspect_dataset(weird_coco_unordered)

        assert info["num_images"] == 3
        assert info["num_annotations"] == 3
        assert info["num_classes"] == 2

        assert 1 in info["category_id_to_index"]
        assert 2 in info["category_id_to_index"]

    def test_special_characters_in_names(self, weird_coco_special_char_names):
        """Category names with special characters."""
        info = inspect_dataset(weird_coco_special_char_names)

        assert info["class_mapping"][0] == "cat/dog"
        assert info["class_mapping"][1] == "item #1"
        assert info["class_mapping"][2] == "type (a)"

    def test_very_large_coordinates(self, weird_coco_huge_coordinates):
        """Very large coordinate values (8K images)."""
        info = inspect_dataset(weird_coco_huge_coordinates)

        assert info["has_boxes"] is True
        assert info["num_images"] == 1

    def test_multi_polygon_segmentation(self, weird_coco_mixed_segmentation_formats):
        """Complex multi-polygon segmentations."""
        info = inspect_dataset(weird_coco_mixed_segmentation_formats)

        assert info["has_masks"] is True
        assert info["num_annotations"] == 2

    def test_degenerate_bboxes(self, weird_coco_single_point_bbox):
        """Degenerate bboxes (zero width/height - points and lines)."""
        info = inspect_dataset(weird_coco_single_point_bbox)

        assert info["has_boxes"] is True
        assert info["num_annotations"] == 3

    def test_whitespace_in_names(self, weird_coco_whitespace_in_names):
        """Whitespace (spaces, tabs, newlines) in names."""
        info = inspect_dataset(weird_coco_whitespace_in_names)

        assert "  cat  " in info["class_mapping"].values()
        assert "dog\n" in info["class_mapping"].values()

    def test_scientific_notation_values(self, weird_coco_scientific_notation):
        """Scientific notation in numeric fields."""
        info = inspect_dataset(weird_coco_scientific_notation)

        assert info["num_images"] == 1
        assert info["has_boxes"] is True

    def test_explicit_null_values(self, weird_coco_null_fields):
        """Explicit None/null values in optional fields."""
        info = inspect_dataset(weird_coco_null_fields)

        assert info["has_boxes"] is True
        assert info["num_classes"] == 1


class TestWeirdFormatConsistency:
    """Test consistency across functions with weird formats."""

    def test_unordered_data_consistency(self, weird_coco_unordered):
        """Unordered IDs produce consistent results."""
        info1 = inspect_dataset(weird_coco_unordered)
        info2 = inspect_dataset(weird_coco_unordered)

        assert info1["num_classes"] == info2["num_classes"]
        assert info1["class_mapping"] == info2["class_mapping"]
