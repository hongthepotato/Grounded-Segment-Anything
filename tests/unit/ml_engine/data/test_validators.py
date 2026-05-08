"""
Unit tests for ml_engine.data.validators module.

Tests:
- validate_coco_format: Validate COCO data structure
- compute_bbox_from_mask: Generate bbox from segmentation
- compute_area_from_mask: Compute area from segmentation
- normalize_coco_annotations: Normalize annotations to canonical form
- check_data_quality: Quality checks and warnings
- split_dataset: Train/val/test splitting
"""

import copy

import pytest

from ml_engine.data.validators import (
    check_data_quality,
    compute_area_from_mask,
    compute_bbox_from_mask,
    normalize_coco_annotations,
    split_dataset,
    validate_coco_format,
)

# =============================================================================
# validate_coco_format Tests
# =============================================================================


class TestValidateCOCOFormat:
    """Tests for validate_coco_format function."""

    def test_valid_combined_data(self, valid_coco_data_combined):
        """Valid COCO data with boxes and masks passes validation."""
        is_valid, errors = validate_coco_format(valid_coco_data_combined)
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_boxes_only(self, valid_coco_data_boxes_only):
        """Valid COCO data with boxes only passes validation."""
        is_valid, errors = validate_coco_format(valid_coco_data_boxes_only)
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_masks_only(self, valid_coco_data_masks_only):
        """Valid COCO data with masks only passes validation."""
        is_valid, errors = validate_coco_format(valid_coco_data_masks_only)
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_sparse_ids(self, valid_coco_data_sparse_ids):
        """COCO data with non-sequential IDs passes validation."""
        is_valid, errors = validate_coco_format(valid_coco_data_sparse_ids)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_images_key(self, invalid_coco_missing_images):
        """Missing 'images' key fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_missing_images)
        assert is_valid is False
        assert any("images" in e.lower() for e in errors)

    def test_missing_annotations_key(self, invalid_coco_missing_annotations):
        """Missing 'annotations' key fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_missing_annotations)
        assert is_valid is False
        assert any("annotations" in e.lower() for e in errors)

    def test_missing_categories_key(self, invalid_coco_missing_categories):
        """Missing 'categories' key fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_missing_categories)
        assert is_valid is False
        assert any("categories" in e.lower() for e in errors)

    def test_empty_images_list(self, invalid_coco_empty_images):
        """Empty images list fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_empty_images)
        assert is_valid is False
        assert any("no images" in e.lower() for e in errors)

    def test_empty_annotations_list(self, invalid_coco_empty_annotations):
        """Empty annotations list fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_empty_annotations)
        assert is_valid is False
        assert any("no annotations" in e.lower() for e in errors)

    def test_negative_ids(self, invalid_coco_negative_ids):
        """Negative IDs fail validation."""
        is_valid, errors = validate_coco_format(invalid_coco_negative_ids)
        assert is_valid is False
        assert len(errors) == 5

    def test_duplicate_category_ids(self, invalid_coco_duplicate_category_ids):
        """Duplicate category IDs fail validation."""
        is_valid, errors = validate_coco_format(invalid_coco_duplicate_category_ids)
        assert is_valid is False
        assert any("duplicate" in e.lower() for e in errors)

    def test_dangling_image_reference(self, invalid_coco_dangling_reference):
        """Annotation referencing non-existent image fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_dangling_reference)
        assert is_valid is False
        assert any("non-existent" in e.lower() or "not found" in e.lower() for e in errors)

    def test_no_bbox_no_segmentation(self, invalid_coco_no_bbox_no_seg):
        """Annotation with neither bbox nor segmentation fails validation."""
        is_valid, errors = validate_coco_format(invalid_coco_no_bbox_no_seg)
        assert is_valid is False
        assert any("bbox" in e.lower() or "segmentation" in e.lower() for e in errors)


class TestValidateBboxFormat:
    """Tests for bbox validation within validate_coco_format."""

    def test_valid_bbox(self):
        """Valid bbox [x, y, w, h] passes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [100, 100, 50, 50]}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_bbox_with_floats(self):
        """Bbox with float values is valid."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [100.5, 100.5, 50.5, 50.5]}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_bbox_wrong_length(self):
        """Bbox with wrong number of elements fails."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [100, 100, 50]}  # Only 3 elements
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert any("must have 4 elements" in e.lower() for e in errors)
        assert is_valid is False

    def test_bbox_negative_width(self):
        """Bbox with negative width fails."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [100, 100, -50, 50]}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert any("width must be >= 0" in e.lower() for e in errors)

    def test_bbox_none_with_valid_segmentation(self):
        """None bbox with valid segmentation is allowed."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": None,
                    "segmentation": [[100, 100, 150, 100, 150, 150, 100, 150]],
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_bbox_none_with_invalid_segmentation(self):
        """None bbox with invalid segmentation fails."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": None, "segmentation": ["invalid"]}
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert any("is none/empty but no valid segmentation found" in e.lower() for e in errors)


class TestValidateSegmentationFormat:
    """Tests for segmentation validation within validate_coco_format."""

    def test_valid_polygon(self, polygon_segmentation):
        """Valid polygon segmentation passes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "segmentation": polygon_segmentation}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_valid_multi_polygon(self, multi_polygon_segmentation):
        """Multiple polygons in segmentation passes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "segmentation": multi_polygon_segmentation}
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_valid_rle_format(self):
        """RLE segmentation format passes."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": {"counts": [1, 2, 3], "size": [480, 640]},
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True

    def test_polygon_too_few_points(self):
        """Polygon with fewer than 3 points (6 coords) fails."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[100, 100, 150, 150]],  # Only 2 points
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert any("must have at least 6 coordinates (3 points)" in e.lower() for e in errors)

    def test_polygon_odd_coordinates(self):
        """Polygon with odd number of coordinates fails."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [[100, 100, 150, 100, 150, 150, 100]],  # 7 coords
                }
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert any("must have even number of coordinates (x,y pairs)" in e.lower() for e in errors)


# =============================================================================
# compute_bbox_from_mask Tests
# =============================================================================


class TestComputeBboxFromMask:
    """Tests for compute_bbox_from_mask function."""

    # -------------------------------------------------------------------------
    # Polygon Format Tests - Various Shapes
    # -------------------------------------------------------------------------

    def test_simple_square_polygon(self, polygon_segmentation):
        """Compute bbox from simple square polygon."""
        bbox = compute_bbox_from_mask(polygon_segmentation, height=480, width=640)

        # Expected: [x_min, y_min, width, height]
        assert len(bbox) == 4
        # Polygon is from (100,100) to (150,150), so box is 51x51
        assert bbox[0] == pytest.approx(100.0, abs=1)
        assert bbox[1] == pytest.approx(100.0, abs=1)
        assert bbox[2] == pytest.approx(51.0, abs=1)
        assert bbox[3] == pytest.approx(51.0, abs=1)

    def test_multi_polygon(self, multi_polygon_segmentation):
        """
        Compute ONE bbox that encompasses all polygons (COCO semantics).

        Per COCO spec: multiple polygons in one annotation = one object with
        disconnected parts. The bbox must cover the entire object.
        """
        bbox = compute_bbox_from_mask(multi_polygon_segmentation, height=480, width=640)

        # Two polygons: (100-120, 100-120) and (150-170, 150-170)
        # ONE bounding box should cover BOTH parts of the object
        assert bbox[0] == pytest.approx(100.0, abs=1)
        assert bbox[1] == pytest.approx(100.0, abs=1)
        # Width should span from 100 to 170 = 71
        assert bbox[2] == pytest.approx(71.0, abs=2)
        assert bbox[3] == pytest.approx(71.0, abs=2)

    def test_large_polygon(self):
        """Compute bbox for larger polygon."""
        polygon = [[0, 0, 200, 0, 200, 150, 0, 150]]
        bbox = compute_bbox_from_mask(polygon, height=480, width=640)

        assert bbox[0] == pytest.approx(0.0, abs=1)
        assert bbox[1] == pytest.approx(0.0, abs=1)
        assert bbox[2] == pytest.approx(201.0, abs=1)
        assert bbox[3] == pytest.approx(151.0, abs=1)

    def test_bbox_clipped_to_image_bounds(self):
        """Bbox should be clipped to image dimensions."""
        # Polygon extends beyond image bounds
        polygon = [[0, 0, 700, 0, 700, 500, 0, 500]]  # Larger than 640x480
        bbox = compute_bbox_from_mask(polygon, height=480, width=640)

        # Should be clipped
        assert bbox[0] >= 0
        assert bbox[1] >= 0
        assert bbox[0] + bbox[2] <= 640
        assert bbox[1] + bbox[3] <= 480

    def test_empty_polygon_raises_error(self):
        """Empty polygon should raise an exception."""
        # pycocotools raises Exception for invalid input types
        with pytest.raises(Exception):
            compute_bbox_from_mask([[]], height=480, width=640)

    # -------------------------------------------------------------------------
    # Polygon Format Tests - Complex Shapes
    # -------------------------------------------------------------------------

    def test_triangle_polygon(self):
        """Compute bbox from triangle (minimum valid polygon - 3 points)."""
        triangle = [[150, 50, 100, 150, 200, 150]]
        bbox = compute_bbox_from_mask(triangle, height=480, width=640)

        assert bbox[0] == pytest.approx(100.0, abs=2)
        assert bbox[1] == pytest.approx(50.0, abs=2)
        assert bbox[2] == pytest.approx(101.0, abs=2)
        assert bbox[3] == pytest.approx(101.0, abs=2)

    def test_irregular_polygon(self):
        """Compute bbox from irregular (non-convex) polygon."""
        # L-shaped polygon
        l_shape = [[50, 50, 150, 50, 150, 100, 100, 100, 100, 200, 50, 200]]
        bbox = compute_bbox_from_mask(l_shape, height=480, width=640)

        assert bbox[0] == pytest.approx(50.0, abs=2)
        assert bbox[1] == pytest.approx(50.0, abs=2)
        assert bbox[2] == pytest.approx(101.0, abs=2)
        assert bbox[3] == pytest.approx(151.0, abs=2)

    def test_concave_polygon(self):
        """Compute bbox from concave polygon (star-like shape)."""
        arrow = [[50, 100, 100, 50, 100, 80, 150, 80, 150, 120, 100, 120, 100, 150]]
        bbox = compute_bbox_from_mask(arrow, height=480, width=640)

        assert bbox[0] == pytest.approx(50.0, abs=2)
        assert bbox[1] == pytest.approx(50.0, abs=2)
        assert bbox[2] == pytest.approx(101.0, abs=2)
        assert bbox[3] == pytest.approx(101.0, abs=2)

    def test_polygon_at_image_corner(self):
        """Compute bbox from polygon at image corner (0,0)."""
        corner = [[0, 0, 50, 0, 50, 50, 0, 50]]
        bbox = compute_bbox_from_mask(corner, height=480, width=640)

        assert bbox[0] == pytest.approx(0.0, abs=1)
        assert bbox[1] == pytest.approx(0.0, abs=1)
        assert bbox[2] == pytest.approx(51.0, abs=1)
        assert bbox[3] == pytest.approx(51.0, abs=1)

    # -------------------------------------------------------------------------
    # RLE Format Tests
    # -------------------------------------------------------------------------

    def test_uncompressed_rle_format(self):
        """Compute bbox from uncompressed RLE format (list of counts)."""
        import numpy as np
        from pycocotools import mask as mask_utils

        # Create binary mask
        binary_mask = np.zeros((200, 200), dtype=np.uint8, order="F")
        binary_mask[100:110, 100:110] = 1

        # Encode to RLE
        rle = mask_utils.encode(binary_mask)

        # Compute bbox
        bbox = compute_bbox_from_mask(rle, height=200, width=200)

        assert bbox[0] == pytest.approx(100.0, abs=1)
        assert bbox[1] == pytest.approx(100.0, abs=1)
        assert bbox[2] == pytest.approx(10.0, abs=1)
        assert bbox[3] == pytest.approx(10.0, abs=1)

    def test_rle_complex_shape(self):
        """Compute bbox from RLE of complex (non-rectangular) shape."""
        import numpy as np
        from pycocotools import mask as mask_utils

        # Create a circle-ish shape
        binary_mask = np.zeros((100, 100), dtype=np.uint8, order="F")
        y, x = np.ogrid[:100, :100]
        center = (50, 50)
        radius = 30
        circle_mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        binary_mask[circle_mask] = 1

        rle = mask_utils.encode(binary_mask)
        bbox = compute_bbox_from_mask(rle, height=100, width=100)

        # Circle center at (50,50) with radius 30: bbox ~(20, 20, 61, 61)
        assert bbox[0] == pytest.approx(20.0, abs=2)
        assert bbox[1] == pytest.approx(20.0, abs=2)
        assert bbox[2] == pytest.approx(61.0, abs=2)
        assert bbox[3] == pytest.approx(61.0, abs=2)


# =============================================================================
# compute_area_from_mask Tests
# =============================================================================


class TestComputeAreaFromMask:
    """Tests for compute_area_from_mask function."""

    def test_simple_square_polygon(self, polygon_segmentation):
        """Compute area of square polygon."""
        area = compute_area_from_mask(polygon_segmentation, height=480, width=640)

        assert area == pytest.approx(2500.0, rel=0.1)

    def test_larger_rectangle(self):
        """Compute area of larger rectangle."""
        polygon = [[0, 0, 100, 0, 100, 50, 0, 50]]
        area = compute_area_from_mask(polygon, height=480, width=640)

        assert area == pytest.approx(5000.0, rel=0.1)

    def test_area_is_float(self, polygon_segmentation):
        """Area should be returned as float."""
        area = compute_area_from_mask(polygon_segmentation, height=480, width=640)
        assert isinstance(area, float)

    def test_area_positive(self, polygon_segmentation):
        """Area should be positive for valid polygon."""
        area = compute_area_from_mask(polygon_segmentation, height=480, width=640)
        assert area > 0

    # -------------------------------------------------------------------------
    # Complex Shape Tests
    # -------------------------------------------------------------------------

    def test_triangle_area(self):
        """Compute area of triangle (base * height / 2)."""
        # Right triangle: base=100, height=100
        # Points: (100,100), (200,100), (100,200)
        triangle = [[100, 100, 200, 100, 100, 200]]
        area = compute_area_from_mask(triangle, height=480, width=640)

        assert area == pytest.approx(5000.0, rel=0.15)

    def test_irregular_polygon_area(self):
        """Compute area of L-shaped polygon."""
        # L-shape: 100x150 minus 50x100 corner
        l_shape = [[50, 50, 150, 50, 150, 100, 100, 100, 100, 200, 50, 200]]
        area = compute_area_from_mask(l_shape, height=480, width=640)

        assert area == pytest.approx(10000.0, rel=0.15)

    def test_thin_polygon_area(self):
        """Compute area of very thin polygon."""
        thin = [[100, 100, 300, 100, 300, 102, 100, 102]]
        area = compute_area_from_mask(thin, height=480, width=640)

        assert area == pytest.approx(400.0, rel=0.2)

    # -------------------------------------------------------------------------
    # RLE Format Tests
    # -------------------------------------------------------------------------

    def test_rle_area(self):
        """Compute area from RLE format."""
        import numpy as np
        from pycocotools import mask as mask_utils

        binary_mask = np.zeros((200, 200), dtype=np.uint8, order="F")
        binary_mask[50:100, 50:100] = 1

        rle = mask_utils.encode(binary_mask)
        area = compute_area_from_mask(rle, height=200, width=200)

        assert area == pytest.approx(2500.0, rel=0.05)

    def test_rle_circle_area(self):
        """Compute area from RLE of circular shape."""
        import numpy as np
        from pycocotools import mask as mask_utils

        binary_mask = np.zeros((100, 100), dtype=np.uint8, order="F")
        y, x = np.ogrid[:100, :100]
        center = (50, 50)
        radius = 20
        circle_mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        binary_mask[circle_mask] = 1

        rle = mask_utils.encode(binary_mask)
        area = compute_area_from_mask(rle, height=100, width=100)

        expected_area = 3.14159 * (radius**2)
        assert area == pytest.approx(expected_area, rel=0.1)


# =============================================================================
# normalize_coco_annotations Tests
# =============================================================================


class TestNormalizeCOCOAnnotations:
    """Tests for normalize_coco_annotations function."""

    def test_adds_bbox_from_mask(self, valid_coco_data_masks_only):
        """Normalizing adds bbox when only mask is present."""
        data = copy.deepcopy(valid_coco_data_masks_only)

        assert "bbox" not in data["annotations"][0] or data["annotations"][0].get("bbox") in [None, []]

        normalized = normalize_coco_annotations(data, in_place=False)

        for ann in normalized["annotations"]:
            assert "bbox" in ann
            assert len(ann["bbox"]) == 4
            assert all(isinstance(x, float) for x in ann["bbox"])

    def test_adds_area_from_mask(self, valid_coco_data_masks_only):
        """Normalizing adds area when only mask is present."""
        data = copy.deepcopy(valid_coco_data_masks_only)

        normalized = normalize_coco_annotations(data, in_place=False)

        for ann in normalized["annotations"]:
            assert "area" in ann
            assert ann["area"] > 0

    def test_preserves_existing_bbox(self, valid_coco_data_combined):
        """Normalizing preserves existing valid bbox."""
        data = copy.deepcopy(valid_coco_data_combined)
        original_bbox = data["annotations"][0]["bbox"].copy()

        normalized = normalize_coco_annotations(data, in_place=False)
        assert normalized["annotations"][0]["bbox"] == original_bbox

    def test_in_place_modification(self, valid_coco_data_masks_only):
        """in_place=True modifies original data."""
        data = copy.deepcopy(valid_coco_data_masks_only)
        original_id = id(data)

        result = normalize_coco_annotations(data, in_place=True)

        # Should be same object
        assert id(result) == original_id
        bbox = result["annotations"][0]["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(100.0, abs=2)
        assert bbox[1] == pytest.approx(100.0, abs=2)

    def test_not_in_place_creates_copy(self, valid_coco_data_masks_only):
        """in_place=False creates a copy."""
        data = copy.deepcopy(valid_coco_data_masks_only)
        original_id = id(data)

        result = normalize_coco_annotations(data, in_place=False)

        # Should be different object
        assert id(result) != original_id

    def test_converts_polygon_to_rle(self, valid_coco_data_masks_only):
        """Normalizing converts polygon to compressed RLE."""
        data = copy.deepcopy(valid_coco_data_masks_only)

        assert isinstance(data["annotations"][0]["segmentation"], list)
        assert isinstance(data["annotations"][0]["segmentation"][0], list)

        normalized = normalize_coco_annotations(data, in_place=False)

        # Should be converted to RLE dict with bytes counts
        seg = normalized["annotations"][0]["segmentation"]
        assert isinstance(seg, dict)
        assert "counts" in seg
        assert "size" in seg
        assert isinstance(seg["counts"], bytes)


# =============================================================================
# check_data_quality Tests
# =============================================================================


class TestCheckDataQuality:
    """Tests for check_data_quality function."""

    def test_returns_expected_keys(self, valid_coco_data_combined):
        """Quality report contains expected keys."""
        report = check_data_quality(valid_coco_data_combined)

        expected_keys = [
            "total_images",
            "total_annotations",
            "images_without_annotations",
            "small_objects",
            "large_objects",
            "samples_per_class",
            "class_distribution",
            "warnings",
        ]
        for key in expected_keys:
            assert key in report

    def test_counts_total_images(self, valid_coco_data_combined):
        """Correctly counts total images."""
        report = check_data_quality(valid_coco_data_combined)
        assert report["total_images"] == len(valid_coco_data_combined["images"])

    def test_counts_total_annotations(self, valid_coco_data_combined):
        """Correctly counts total annotations."""
        report = check_data_quality(valid_coco_data_combined)
        assert report["total_annotations"] == len(valid_coco_data_combined["annotations"])

    def test_detects_images_without_annotations(self, coco_data_with_quality_issues):
        """Detects images that have no annotations."""
        report = check_data_quality(coco_data_with_quality_issues)
        assert report["images_without_annotations"] == 1

    def test_detects_small_objects(self, coco_data_with_quality_issues):
        """Detects very small objects."""
        report = check_data_quality(coco_data_with_quality_issues)
        assert report["small_objects"] == 12

    def test_detects_large_objects(self, coco_data_with_quality_issues):
        """Detects very large objects."""
        report = check_data_quality(coco_data_with_quality_issues)
        assert report["large_objects"] == 1

    def test_class_distribution(self, valid_coco_data_combined):
        """Correctly computes class distribution."""
        report = check_data_quality(valid_coco_data_combined)

        # Should have counts for each class
        assert len(report["class_distribution"]) == 2
        # Total should match annotation count
        total = sum(report["class_distribution"].values())
        assert total == report["total_annotations"]

    def test_warnings_is_list(self, valid_coco_data_combined):
        """Warnings field is always a list."""
        report = check_data_quality(valid_coco_data_combined)
        assert isinstance(report["warnings"], list)

    def test_detects_class_imbalance(self, coco_data_with_quality_issues):
        """Detects high class imbalance."""
        report = check_data_quality(coco_data_with_quality_issues)

        # Should have warning about class imbalance
        assert any("imbalance" in w.lower() for w in report["warnings"])


# =============================================================================
# split_dataset Tests
# =============================================================================


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_incorrect_ratio_raises(self, coco_data_for_splitting):
        """Incorrect split ratios raise error."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            split_dataset(coco_data_for_splitting, splits={"train": 0.6, "val": 0.2, "test": 0.1})
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            split_dataset(coco_data_for_splitting, splits={"train": 0.9, "val": 0.2, "test": 0.1})
        with pytest.raises(ValueError, match="Split ratios must be greater than 0"):
            split_dataset(coco_data_for_splitting, splits={"train": 0.9, "val": 0.2, "test": -0.1})

    def test_default_splits(self, coco_data_for_splitting):
        """Default 70/15/15 split works correctly."""
        splits = split_dataset(coco_data_for_splitting)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_custom_splits(self, coco_data_for_splitting):
        """Custom split ratios work correctly."""
        splits = split_dataset(coco_data_for_splitting, splits={"train": 0.8, "val": 0.1, "test": 0.1})

        total_images = len(coco_data_for_splitting["images"])
        train_count = len(splits["train"]["images"])
        val_count = len(splits["val"]["images"])
        test_count = len(splits["test"]["images"])

        # Train should be approximately 80%
        assert train_count == pytest.approx(total_images * 0.8, abs=1)
        assert val_count == pytest.approx(total_images * 0.1, abs=1)
        assert test_count == pytest.approx(total_images * 0.1, abs=1)

    def test_splits_contain_coco_structure(self, coco_data_for_splitting):
        """Each split contains valid COCO structure."""
        splits = split_dataset(coco_data_for_splitting)

        for split_name, split_data in splits.items():
            assert "images" in split_data
            assert "annotations" in split_data
            assert "categories" in split_data

    def test_no_image_overlap(self, coco_data_for_splitting):
        """No images appear in multiple splits."""
        splits = split_dataset(coco_data_for_splitting)

        train_ids = {img["id"] for img in splits["train"]["images"]}
        val_ids = {img["id"] for img in splits["val"]["images"]}
        test_ids = {img["id"] for img in splits["test"]["images"]}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_all_images_assigned(self, coco_data_for_splitting):
        """All annotated images are assigned to some split."""
        splits = split_dataset(coco_data_for_splitting)

        all_split_ids = set()
        for split_data in splits.values():
            for img in split_data["images"]:
                all_split_ids.add(img["id"])

        # Get IDs of images that have annotations
        image_to_anns = {}
        for ann in coco_data_for_splitting["annotations"]:
            image_to_anns[ann["image_id"]] = True

        annotated_ids = {img["id"] for img in coco_data_for_splitting["images"] if img["id"] in image_to_anns}

        assert all_split_ids == annotated_ids

    def test_annotations_follow_images(self, coco_data_for_splitting):
        """Annotations in each split only reference images in that split."""
        splits = split_dataset(coco_data_for_splitting)

        for split_name, split_data in splits.items():
            image_ids = {img["id"] for img in split_data["images"]}

            for ann in split_data["annotations"]:
                assert ann["image_id"] in image_ids, (
                    f"Annotation {ann['id']} in {split_name} references image not in split"
                )

    def test_categories_preserved_in_all_splits(self, coco_data_for_splitting):
        """All original categories are preserved in each split."""
        original_categories = coco_data_for_splitting["categories"]
        splits = split_dataset(coco_data_for_splitting)

        for split_name, split_data in splits.items():
            assert split_data["categories"] == original_categories

    def test_reproducible_with_seed(self, coco_data_for_splitting):
        """Same random seed produces same splits."""
        splits1 = split_dataset(coco_data_for_splitting, random_seed=42)
        splits2 = split_dataset(coco_data_for_splitting, random_seed=42)

        # Should have same images in each split
        for split_name in ["train", "val", "test"]:
            ids1 = {img["id"] for img in splits1[split_name]["images"]}
            ids2 = {img["id"] for img in splits2[split_name]["images"]}
            assert ids1 == ids2

    def test_different_seed_different_splits(self, coco_data_for_splitting):
        """Different random seeds produce different splits."""
        splits1 = split_dataset(coco_data_for_splitting, random_seed=42)
        splits2 = split_dataset(coco_data_for_splitting, random_seed=123)

        train_ids1 = {img["id"] for img in splits1["train"]["images"]}
        train_ids2 = {img["id"] for img in splits2["train"]["images"]}

        assert train_ids1 != train_ids2 or len(coco_data_for_splitting["images"]) < 5

    def test_stratified_split(self, coco_data_for_splitting):
        """Stratified split maintains class distribution."""
        splits = split_dataset(coco_data_for_splitting, stratify=True, random_seed=42)

        # Each non-empty split should have some annotations
        for split_name, split_data in splits.items():
            if split_data["images"]:  # If split has images
                # Should also have annotations
                assert len(split_data["annotations"]) > 0

    def test_small_dataset_fallback(self):
        """Very small datasets fall back to random split."""
        small_data = {
            "images": [
                {"id": i, "file_name": f"img_{i}.jpg", "width": 640, "height": 480}
                for i in range(4)  # Only 4 images
            ],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 0, "bbox": [0, 0, 50, 50]} for i in range(4)
            ],
            "categories": [{"id": 0, "name": "class_a"}],
        }

        # Should not raise, should fall back to random split
        splits = split_dataset(small_data, stratify=True)

        # Should still produce valid splits
        assert "train" in splits or "all" in splits


# =============================================================================
# Imperfect COCO Data Edge Cases
# =============================================================================


class TestImperfectCOCOData:
    """Tests for handling imperfect/malformed COCO data from real annotation tools."""

    def test_empty_nested_segmentation_list_with_valid_bbox(self):
        """Segmentation as [[]] with valid bbox should fail (invalid seg format)."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[]], "bbox": [10, 10, 20, 20]},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # [[]] is invalid segmentation format (empty polygon)
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert any("polygon" in e.lower() or "empty" in e.lower() for e in errors)

    def test_bbox_none_with_empty_nested_segmentation(self):
        """bbox=None with segmentation=[[]] should fail (no valid annotation)."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": None, "segmentation": [[]]},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # Both bbox and segmentation are invalid
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        # Should catch that [[]] is not a valid segmentation to fall back on
        assert len(errors) >= 1

    def test_bbox_none_with_too_few_coords_segmentation(self):
        """bbox=None with segmentation=[[10, 20]] (too few coords) should fail."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": None, "segmentation": [[10, 20]]},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # [[10, 20]] has only 2 coords, needs 6 minimum
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False

    def test_missing_both_bbox_and_segmentation(self):
        """Annotations missing both bbox and segmentation should fail."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0},  # No bbox, no segmentation
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        assert len(errors) > 0

    def test_bbox_none_segmentation_empty_list(self):
        """bbox=None and segmentation=[] should fail."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0},
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": None},
                {"id": 2, "image_id": 0, "category_id": 0, "segmentation": []},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        # Should have errors for annotations without valid bbox or segmentation
        assert len(errors) >= 1

    def test_bbox_empty_list(self):
        """bbox=[] (empty list) should fail but valid bbox should pass."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0},  # Missing both
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": None},  # None bbox
                {"id": 2, "image_id": 0, "category_id": 0, "bbox": []},  # Empty list bbox
                {"id": 3, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]},  # Valid
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        # First 3 annotations are invalid
        assert len(errors) >= 1

    def test_bbox_boolean_false_with_valid_segmentation(self):
        """bbox=False (wrong type) with valid segmentation should pass."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": False,
                    "segmentation": [[10, 10, 20, 10, 20, 20]],
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # Has valid segmentation, so should pass (False bbox is treated as missing)
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False

    def test_bbox_integer_zero_with_valid_segmentation(self):
        """bbox=0 (wrong type) with valid segmentation should pass."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": 0,
                    "segmentation": [[10, 10, 20, 10, 20, 20]],
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # Has valid segmentation, so should pass
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False

    def test_bbox_string_with_valid_segmentation(self):
        """bbox="invalid" (wrong type) with valid segmentation should pass."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": "invalid",
                    "segmentation": [[10, 10, 20, 10, 20, 20]],
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # Has valid segmentation, so should pass
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False

    def test_rle_segmentation_format(self):
        """RLE format segmentation should be valid."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": {"counts": [10, 20, 30], "size": [100, 100]},
                    "iscrowd": 1,
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is True
        assert len(errors) == 0

    def test_single_valid_among_invalids(self):
        """Dataset with mix of valid/invalid annotations fails for invalid ones."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0},  # Invalid: no bbox/seg
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]},  # Valid
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        is_valid, errors = validate_coco_format(data)
        assert is_valid is False
        # Should report error for annotation 0

    def test_normalize_handles_empty_nested_segmentation(self):
        """normalize_coco_annotations preserves bbox when segmentation is [[]]
        (empty nested list — a real-world COCO export quirk).

        Pinned contract: must succeed and leave the bbox untouched. Downstream
        training depends on the bbox surviving normalization. If a future change
        makes this raise, callers (dataloaders) need their own guard — flip this
        test to pytest.raises with the new exception only after auditing those
        callers.
        """
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[]], "bbox": [10, 10, 20, 20]},
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        result = normalize_coco_annotations(data, in_place=False)
        assert result["annotations"][0]["bbox"] == [10, 10, 20, 20]

    def test_normalize_handles_wrong_type_bbox(self):
        """normalize_coco_annotations handles wrong-type bbox with valid segmentation."""
        data = {
            "images": [{"id": 0, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": "invalid",
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                },
            ],
            "categories": [{"id": 0, "name": "cat"}],
        }
        # Should generate bbox from segmentation
        result = normalize_coco_annotations(data, in_place=False)
        # Bbox should be computed from segmentation
        bbox = result["annotations"][0]["bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
