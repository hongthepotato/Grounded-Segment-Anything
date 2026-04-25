"""
Unit tests for ml_engine.data.preprocessing module.

Tests:
- BaseModelPreprocessor: Abstract base class
- SAMPreprocessor: SAM-specific preprocessing
- GroundingDINOPreprocessor: DINO-specific preprocessing
- YOLOPreprocessor: YOLO-specific preprocessing
- MultiModelPreprocessor: Multi-model orchestrator
- create_preprocessor_from_models: Factory function
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ml_engine.data.preprocessing import (
    BaseModelPreprocessor,
    GroundingDINOPreprocessor,
    MultiModelPreprocessor,
    SAMPreprocessor,
    create_preprocessor_from_models,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def preprocessing_config():
    """Complete preprocessing configuration for all models."""
    return {
        "grounding_dino": {
            "input_size": {"min_size": 800, "max_size": 1333},
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "sam": {
            "input_size": {"height": 1024, "width": 1024},
            "normalization": {
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
            },
            "padding_value": 0,
            "mask_output_size": 256,
        },
        "yolo": {
            "input_size": {"size": 640},
            "normalization": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
    }


@pytest.fixture
def temp_config_file(preprocessing_config):
    """Create temporary config file."""
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "preprocessing.yaml"

    config = {"preprocessing": preprocessing_config}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    yield str(config_path)

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Sample PIL Image for testing. (W, H)"""
    return Image.new("RGB", (640, 480), color="red")


@pytest.fixture
def sample_boxes():
    """Sample bounding boxes in COCO format [x, y, w, h]."""
    return np.array(
        [
            [100, 100, 50, 50],
            [200, 150, 80, 60],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_masks():
    """Sample binary masks."""
    masks = np.zeros((2, 480, 640), dtype=np.uint8)
    masks[0, 100:150, 100:150] = 1
    masks[1, 150:210, 200:280] = 1
    return masks


# =============================================================================
# SAMPreprocessor Tests
# =============================================================================


class TestSAMPreprocessor:
    """Tests for SAMPreprocessor class."""

    def test_initialization(self, preprocessing_config):
        """Initializes with correct config."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        assert preprocessor.model_name == "sam"
        assert preprocessor.config == config

    def test_preprocess_returns_dict(self, preprocessing_config, sample_image):
        """preprocess returns dict with required keys."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image)

        assert isinstance(result, dict)
        assert "image" in result
        assert "boxes" in result
        assert "masks" in result
        assert "metadata" in result

    def test_preprocess_returns_tensor(self, preprocessing_config, sample_image):
        """preprocess returns a torch Tensor for image."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)
        result = preprocessor.preprocess(sample_image)

        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 1024, 1024)

    def test_metadata_contains_required_keys(self, preprocessing_config, sample_image):
        """Metadata contains required keys."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image)
        metadata = result["metadata"]

        assert "original_size" in metadata
        assert "final_size" in metadata
        assert "model_name" in metadata

    def test_original_size_captured(self, preprocessing_config, sample_image):
        """Original size is correctly captured."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image)
        metadata = result["metadata"]

        # Original image is 640x480 (W x H), metadata stores as (H, W)
        assert metadata["original_size"] == (480, 640)
        assert metadata["final_size"] == (1024, 1024)
        assert metadata["model_name"] == "sam"

    def test_boxes_transformed_in_preprocess(self, preprocessing_config, sample_image, sample_boxes):
        """Boxes are transformed during preprocess."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image, boxes=sample_boxes)

        assert result["boxes"] is not None
        assert isinstance(result["boxes"], np.ndarray)
        assert result["boxes"].shape == sample_boxes.shape
        assert np.array_equal(result["boxes"][0], [160, 160, 240, 240])
        assert np.array_equal(result["boxes"][1], [320, 240, 448, 336])

    def test_boxes_in_xyxy_format(self, preprocessing_config, sample_image, sample_boxes):
        """SAM boxes are in xyxy format (x2 > x1, y2 > y1)."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image, boxes=sample_boxes)

        for box in result["boxes"]:
            assert box[2] > box[0]  # x2 > x1
            assert box[3] > box[1]  # y2 > y1

    def test_masks_transformed_in_preprocess(self, preprocessing_config, sample_image, sample_masks):
        """Masks are transformed during preprocess."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image, masks=sample_masks)

        # Should be resized to mask_output_size (256x256)
        assert result["masks"] is not None
        assert result["masks"].shape == (2, 256, 256)

    def test_masks_preserve_shape_with_padding(self, preprocessing_config, sample_image, sample_masks):
        """Transformed masks preserve shape characteristics accounting for padding."""
        from segment_anything.utils.transforms import ResizeLongestSide

        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        result = preprocessor.preprocess(sample_image, masks=sample_masks)
        transformed = result["masks"]
        metadata = result["metadata"]

        orig_h, orig_w = metadata["original_size"]
        target_size = preprocessor.mask_output_size
        new_h, new_w = ResizeLongestSide.get_preprocess_shape(orig_h, orig_w, target_size)

        assert new_h == 192
        assert new_w == 256

        for i, mask in enumerate(sample_masks):
            orig_mask = mask
            trans_mask = transformed[i]

            content_area = trans_mask[:new_h, :new_w]

            orig_ratio = np.sum(orig_mask > 0) / orig_mask.size
            content_ratio = np.sum(content_area > 0) / content_area.size
            assert abs(orig_ratio - content_ratio) < 0.02

            if new_h < target_size:
                assert np.sum(trans_mask[new_h:, :]) == 0
            if new_w < target_size:
                assert np.sum(trans_mask[:, new_w:]) == 0

            if np.sum(orig_mask) > 0:
                orig_y, orig_x = np.where(orig_mask > 0)
                orig_center_y = np.mean(orig_y) / orig_h
                orig_center_x = np.mean(orig_x) / orig_w

                trans_y, trans_x = np.where(content_area > 0)
                trans_center_y = np.mean(trans_y) / new_h
                trans_center_x = np.mean(trans_x) / new_w

                assert abs(orig_center_y - trans_center_y) < 0.02
                assert abs(orig_center_x - trans_center_x) < 0.02

            unique_vals = len(np.unique(trans_mask))
            assert unique_vals <= 2

    def test_empty_boxes_returns_none(self, preprocessing_config, sample_image):
        """Handles empty boxes array - returns None."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        result = preprocessor.preprocess(sample_image, boxes=empty_boxes)

        assert result["boxes"] is None

    def test_empty_masks_returns_none(self, preprocessing_config, sample_image):
        """Handles empty masks array - returns None."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        empty_masks = np.zeros((0, 480, 640), dtype=np.uint8)
        result = preprocessor.preprocess(sample_image, masks=empty_masks)

        assert result["masks"] is None

    def test_tall_image_handling(self, preprocessing_config):
        """Handles tall (portrait) images correctly."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        tall_image = Image.new("RGB", (480, 800), color="green")
        result = preprocessor.preprocess(tall_image)
        tensor = result["image"]

        assert tensor[:, :, 614:].abs().sum() == 0
        assert tensor[:, :, :614].abs().sum() > 0
        assert tensor.shape == (3, 1024, 1024)

    def test_square_image_handling(self, preprocessing_config):
        """Handles square images correctly."""
        config = preprocessing_config["sam"]
        preprocessor = SAMPreprocessor("sam", config)

        square_image = Image.new("RGB", (512, 512), color="blue")
        result = preprocessor.preprocess(square_image)

        assert result["image"].shape == (3, 1024, 1024)


# =============================================================================
# GroundingDINOPreprocessor Tests
# =============================================================================


class TestGroundingDINOPreprocessor:
    """Tests for GroundingDINOPreprocessor class."""

    # =========================================================================
    # Basic Initialization Tests
    # =========================================================================

    def test_initialization(self, preprocessing_config):
        """Initializes with correct config."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        assert preprocessor.model_name == "grounding_dino"
        assert preprocessor.min_size == config["input_size"]["min_size"]
        assert preprocessor.max_size == config["input_size"]["max_size"]

    def test_preprocess_returns_dict(self, preprocessing_config, sample_image):
        """preprocess returns dict with required keys."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image)

        assert isinstance(result, dict)
        assert "image" in result
        assert "boxes" in result
        assert "masks" in result
        assert "metadata" in result
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["metadata"], dict)

    # =========================================================================
    # Image Preprocessing Tests
    # =========================================================================

    def test_output_tensor_shape(self, preprocessing_config, sample_image):
        """Output tensor has correct shape (C, H, W) with 3 channels."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image)

        assert len(result["image"].shape) == 3
        assert result["image"].shape[0] == 3

    def test_respects_min_size(self, preprocessing_config):
        """Shortest side is resized to min_size (800)."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image = Image.new("RGB", (640, 480), color="red")
        result = preprocessor.preprocess(image)
        tensor = result["image"]

        h, w = tensor.shape[1], tensor.shape[2]
        assert min(h, w) == 800

    def test_respects_max_size(self, preprocessing_config):
        """Longest side does not exceed max_size (1333)."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        large_image = Image.new("RGB", (2000, 1500), color="red")
        result = preprocessor.preprocess(large_image)
        tensor = result["image"]

        h, w = tensor.shape[1], tensor.shape[2]
        assert max(h, w) <= config["input_size"]["max_size"]

    def test_aspect_ratio_preserved(self, preprocessing_config):
        """Aspect ratio is preserved after resize."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image = Image.new("RGB", (800, 400), color="red")  # W:H = 2:1
        result = preprocessor.preprocess(image)
        tensor = result["image"]

        h, w = tensor.shape[1], tensor.shape[2]
        output_ratio = w / h
        input_ratio = 800 / 400
        assert abs(output_ratio - input_ratio) < 0.01

    def test_image_normalized_with_imagenet_stats(self, preprocessing_config):
        """Image pixel values are normalized with ImageNet mean/std."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        # Solid gray image (128, 128, 128)
        image = Image.new("RGB", (800, 800), color=(128, 128, 128))
        result = preprocessor.preprocess(image)
        tensor = result["image"]

        r_mean = tensor[0].mean().item()
        g_mean = tensor[1].mean().item()
        b_mean = tensor[2].mean().item()

        assert abs(r_mean - 0.074) < 0.001
        assert abs(g_mean - 0.205) < 0.001
        assert abs(b_mean - 0.427) < 0.001

    # =========================================================================
    # Box Transformation Tests - CRITICAL
    # =========================================================================

    def test_boxes_converted_to_normalized_cxcywh(self, preprocessing_config, sample_image, sample_boxes):
        """Boxes are converted to normalized [cx, cy, w, h] in [0, 1] range."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image, boxes=sample_boxes)
        transformed = result["boxes"]

        assert transformed is not None
        assert transformed.shape == (2, 4)
        assert transformed.min() >= 0.0, f"Min value {transformed.min()} < 0"
        assert transformed.max() <= 1.0, f"Max value {transformed.max()} > 1"

    def test_boxes_have_positive_dimensions(self, preprocessing_config, sample_image, sample_boxes):
        """Transformed boxes have positive width and height."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image, boxes=sample_boxes)
        transformed = result["boxes"]

        assert (transformed[:, 2] > 0).all(), "Width must be positive"
        assert (transformed[:, 3] > 0).all(), "Height must be positive"

    def test_full_image_box_transforms_correctly(self, preprocessing_config):
        """A box covering the full image should transform to [0.5, 0.5, 1.0, 1.0]."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image = Image.new("RGB", (800, 800), color="red")
        boxes = np.array([[0, 0, 800, 800]], dtype=np.float32)

        result = preprocessor.preprocess(image, boxes=boxes)
        transformed = result["boxes"]

        np.testing.assert_array_almost_equal(transformed[0], [0.5, 0.5, 1.0, 1.0], decimal=2)

    def test_centered_box_transforms_correctly(self, preprocessing_config):
        """A box at image center should have cx=0.5, cy=0.5."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image = Image.new("RGB", (800, 800), color="red")
        boxes = np.array([[300, 300, 200, 200]], dtype=np.float32)

        result = preprocessor.preprocess(image, boxes=boxes)
        transformed = result["boxes"]

        assert abs(transformed[0, 0] - 0.5) < 0.02, f"cx={transformed[0, 0]}, expected 0.5"
        assert abs(transformed[0, 1] - 0.5) < 0.02, f"cy={transformed[0, 1]}, expected 0.5"
        assert abs(transformed[0, 2] - 0.25) < 0.02, f"w={transformed[0, 2]}, expected 0.25"
        assert abs(transformed[0, 3] - 0.25) < 0.02, f"h={transformed[0, 3]}, expected 0.25"

    def test_corner_box_transforms_correctly(self, preprocessing_config):
        """A box at top-left corner should have small cx, cy."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image = Image.new("RGB", (800, 800), color="red")
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)

        result = preprocessor.preprocess(image, boxes=boxes)
        transformed = result["boxes"]

        assert abs(transformed[0, 0] - 0.0625) < 0.02
        assert abs(transformed[0, 1] - 0.0625) < 0.02
        assert abs(transformed[0, 2] - 0.125) < 0.02
        assert abs(transformed[0, 3] - 0.125) < 0.02

    def test_boxes_scale_correctly_with_resize(self, preprocessing_config):
        """Box relative position preserved when image is resized."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        image1 = Image.new("RGB", (800, 800), color="red")
        boxes1 = np.array([[200, 200, 400, 400]], dtype=np.float32)

        image2 = Image.new("RGB", (1600, 1600), color="red")
        boxes2 = np.array([[400, 400, 800, 800]], dtype=np.float32)

        result1 = preprocessor.preprocess(image1, boxes=boxes1)
        result2 = preprocessor.preprocess(image2, boxes=boxes2)

        # Both should have same normalized coordinates
        np.testing.assert_array_almost_equal(result1["boxes"][0], result2["boxes"][0], decimal=2)

    # =========================================================================
    # DINO Does Not Process Masks (Detection-Only Model)
    # =========================================================================

    def test_dino_masks_always_none(self, preprocessing_config, sample_image, sample_masks):
        """DINO is detection-only, masks are always None."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image, masks=sample_masks)

        # DINO doesn't process masks
        assert result["masks"] is None

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_boxes_returns_none(self, preprocessing_config, sample_image):
        """Empty boxes array results in None."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        result = preprocessor.preprocess(sample_image, boxes=empty_boxes)

        assert result["boxes"] is None

    def test_single_box_handled(self, preprocessing_config, sample_image):
        """Single box is handled correctly."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        single_box = np.array([[100, 100, 50, 50]], dtype=np.float32)
        result = preprocessor.preprocess(sample_image, boxes=single_box)

        assert result["boxes"] is not None
        assert result["boxes"].shape == (1, 4)

    def test_no_annotations_still_works(self, preprocessing_config, sample_image):
        """Preprocessing works without any boxes or masks."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image)

        assert result["image"] is not None
        assert result["image"].shape[0] == 3
        assert result["boxes"] is None
        assert result["masks"] is None

    # =========================================================================
    # Metadata Correctness
    # =========================================================================

    def test_metadata_original_size_correct(self, preprocessing_config, sample_image):
        """Metadata contains correct original image size (H, W)."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image)

        assert result["metadata"]["original_size"] == (480, 640)

    def test_metadata_final_size_matches_tensor(self, preprocessing_config, sample_image):
        """Metadata final_size matches actual tensor dimensions."""
        config = preprocessing_config["grounding_dino"]
        preprocessor = GroundingDINOPreprocessor("grounding_dino", config)

        result = preprocessor.preprocess(sample_image)
        tensor = result["image"]
        metadata = result["metadata"]

        expected_h, expected_w = tensor.shape[1], tensor.shape[2]
        assert metadata["final_size"] == (expected_h, expected_w)


# =============================================================================
# MultiModelPreprocessor Tests
# =============================================================================


class TestMultiModelPreprocessor:
    """Tests for MultiModelPreprocessor class."""

    def test_initialization_single_model(self, temp_config_file):
        """Initializes with single model."""
        preprocessor = MultiModelPreprocessor(active_models=["sam"], config_path=temp_config_file)

        assert "sam" in preprocessor.preprocessors
        assert len(preprocessor.preprocessors) == 1

    def test_initialization_multiple_models(self, temp_config_file):
        """Initializes with multiple models."""
        preprocessor = MultiModelPreprocessor(
            active_models=["sam", "grounding_dino"], config_path=temp_config_file
        )

        assert "sam" in preprocessor.preprocessors
        assert "grounding_dino" in preprocessor.preprocessors
        assert len(preprocessor.preprocessors) == 2

    def test_unknown_model_raises(self, temp_config_file):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError):
            MultiModelPreprocessor(active_models=["unknown_model"], config_path=temp_config_file)

    def test_preprocess_batch(self, temp_config_file, sample_image):
        """preprocess_batch returns dict with all models."""
        preprocessor = MultiModelPreprocessor(
            active_models=["sam", "grounding_dino"], config_path=temp_config_file
        )

        results = preprocessor.preprocess_batch(sample_image)

        assert "sam" in results
        assert "grounding_dino" in results

    def test_preprocess_batch_returns_dicts(self, temp_config_file, sample_image):
        """Each model result is a dict with image, boxes, masks, metadata."""
        preprocessor = MultiModelPreprocessor(active_models=["sam"], config_path=temp_config_file)

        results = preprocessor.preprocess_batch(sample_image)

        assert isinstance(results["sam"], dict)
        assert "image" in results["sam"]
        assert "boxes" in results["sam"]
        assert "masks" in results["sam"]
        assert "metadata" in results["sam"]

    def test_preprocess_for_model(self, temp_config_file, sample_image):
        """preprocess_for_model returns single model result dict."""
        preprocessor = MultiModelPreprocessor(
            active_models=["sam", "grounding_dino"], config_path=temp_config_file
        )

        result = preprocessor.preprocess_for_model(sample_image, "sam")

        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["metadata"], dict)

    def test_preprocess_for_unloaded_model_raises(self, temp_config_file, sample_image):
        """Requesting unloaded model raises ValueError."""
        preprocessor = MultiModelPreprocessor(
            active_models=["sam"],  # Only SAM loaded
            config_path=temp_config_file,
        )

        with pytest.raises(ValueError):
            preprocessor.preprocess_for_model(sample_image, "grounding_dino")

    def test_register_preprocessor_classmethod(self, temp_config_file):
        """register_preprocessor adds new preprocessor class."""

        # Create a mock preprocessor class
        class MockPreprocessor(BaseModelPreprocessor):
            def preprocess(self, image, boxes=None, masks=None):
                return {"image": torch.zeros(3, 100, 100), "boxes": None, "masks": None, "metadata": {}}

        # Register it
        MultiModelPreprocessor.register_preprocessor("mock_model", MockPreprocessor)

        # Verify it's registered
        assert "mock_model" in MultiModelPreprocessor.PREPROCESSOR_REGISTRY

        # Cleanup
        del MultiModelPreprocessor.PREPROCESSOR_REGISTRY["mock_model"]

    def test_register_non_subclass_raises(self, temp_config_file):
        """Registering non-subclass raises TypeError."""

        class NotAPreprocessor:
            pass

        with pytest.raises(TypeError):
            MultiModelPreprocessor.register_preprocessor("bad", NotAPreprocessor)


# =============================================================================
# create_preprocessor_from_models Tests
# =============================================================================


class TestCreatePreprocessorFromModels:
    """Tests for create_preprocessor_from_models factory function."""

    def test_creates_preprocessor(self, temp_config_file):
        """Factory creates MultiModelPreprocessor."""
        preprocessor = create_preprocessor_from_models(model_names=["sam"], config_path=temp_config_file)

        assert isinstance(preprocessor, MultiModelPreprocessor)

    def test_multiple_models(self, temp_config_file):
        """Factory handles multiple models."""
        preprocessor = create_preprocessor_from_models(
            model_names=["sam", "grounding_dino"], config_path=temp_config_file
        )

        assert "sam" in preprocessor.preprocessors
        assert "grounding_dino" in preprocessor.preprocessors

    def test_uses_default_config_path(self):
        """Uses default config path when not specified."""
        # This test may fail if default config doesn't exist
        # Skip if default config not available
        from core.constants import DEFAULT_CONFIGS_DIR

        default_path = DEFAULT_CONFIGS_DIR / "preprocessing.yaml"

        if not default_path.exists():
            pytest.skip("Default preprocessing config not found")

        preprocessor = create_preprocessor_from_models(model_names=["sam"])

        assert isinstance(preprocessor, MultiModelPreprocessor)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_workflow(self, temp_config_file, sample_image, sample_boxes, sample_masks):
        """Test complete preprocessing workflow."""
        preprocessor = MultiModelPreprocessor(active_models=["sam"], config_path=temp_config_file)

        # Preprocess image + boxes + masks in one step
        results = preprocessor.preprocess_batch(sample_image, boxes=sample_boxes, masks=sample_masks)

        # Get SAM result - everything is already transformed
        sam_result = results["sam"]

        # Verify all outputs
        assert sam_result["image"].shape == (3, 1024, 1024)
        assert sam_result["boxes"].shape == (2, 4)
        assert sam_result["masks"].shape == (2, 256, 256)

    def test_different_image_sizes(self, temp_config_file):
        """Preprocessing handles various image sizes."""
        preprocessor = MultiModelPreprocessor(active_models=["sam"], config_path=temp_config_file)

        sizes = [(640, 480), (800, 600), (1920, 1080), (480, 640)]

        for width, height in sizes:
            image = Image.new("RGB", (width, height), color="red")
            result = preprocessor.preprocess_for_model(image, "sam")

            # All should produce 1024x1024 output
            assert result["image"].shape == (3, 1024, 1024)
            # Original size should be captured correctly
            assert result["metadata"]["original_size"] == (height, width)

    def test_normalization_applied(self, temp_config_file):
        """Verify normalization is applied."""
        preprocessor = MultiModelPreprocessor(active_models=["sam"], config_path=temp_config_file)

        # Solid red image
        red_image = Image.new("RGB", (100, 100), color="red")
        result = preprocessor.preprocess_for_model(red_image, "sam")
        tensor = result["image"]

        # After normalization, values should not be simple 0-255 or 0-1
        # Red channel should have different value than blue/green
        assert not torch.allclose(tensor[0], tensor[1])  # R != G after normalization
