"""
Unit tests for ml_engine.data.dataset_factory module.

Tests:
- DatasetFactory.create_dataset: Factory method for creating datasets
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from PIL import Image

from ml_engine.data.dataset_factory import DatasetFactory
from ml_engine.data.loaders import TeacherDataset
from ml_engine.data.validators import normalize_coco_annotations

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir_with_images(valid_coco_data_combined):
    """Create temporary directory with images and config files."""
    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir()

    # Create images
    for img_info in valid_coco_data_combined["images"]:
        img_path = images_dir / img_info["file_name"]
        img = Image.new("RGB", (img_info["width"], img_info["height"]), color="gray")
        img.save(img_path)

    # Create preprocessing config
    config_dir = Path(temp_dir) / "configs"
    config_dir.mkdir()
    preprocessing_config = {
        "preprocessing": {
            "sam": {
                "input_size": {"height": 1024, "width": 1024},
                "normalization": {
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                },
                "padding_value": 0,
                "mask_output_size": 256,
            },
            "grounding_dino": {
                "input_size": {"min_size": 800, "max_size": 1333},
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
        }
    }
    with open(config_dir / "preprocessing.yaml", "w", encoding="utf-8") as f:
        yaml.dump(preprocessing_config, f)

    yield {
        "temp_dir": temp_dir,
        "images_dir": images_dir,
        "config_dir": config_dir,
        "coco_data": valid_coco_data_combined,
    }

    shutil.rmtree(temp_dir)


@pytest.fixture
def normalized_coco_data(valid_coco_data_combined):
    """Normalized COCO data."""
    return normalize_coco_annotations(valid_coco_data_combined, in_place=False)


@pytest.fixture
def mock_path_resolver():
    """Mock path resolver that returns a dummy path."""

    def resolver(file_name):
        return f"/dummy/path/{file_name}"

    return resolver


@pytest.fixture
def dataset_info_combined():
    """Dataset info for combined (boxes + masks) data."""
    return {"has_boxes": True, "has_masks": True, "num_classes": 2, "class_mapping": {0: "ear", 1: "defect"}}


@pytest.fixture
def dataset_info_boxes_only():
    """Dataset info for boxes-only data."""
    return {"has_boxes": True, "has_masks": False, "num_classes": 2, "class_mapping": {0: "ear", 1: "defect"}}


# =============================================================================
# DatasetFactory.create_dataset Tests
# =============================================================================


class TestDatasetFactoryCreateDataset:
    """Tests for DatasetFactory.create_dataset method."""

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_returns_teacher_dataset(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """Factory returns TeacherDataset instance."""
        # Setup mocks
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=None,
            is_training=True,
        )

        assert isinstance(dataset, TeacherDataset)

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_creates_preprocessor_for_models(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """Factory creates preprocessor with specified models."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam", "grounding_dino"],
            augmentation_config=None,
            is_training=True,
        )

        mock_create_preprocessor.assert_called_once_with(model_names=["sam", "grounding_dino"])

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_no_augmentation_when_disabled(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """No augmentation pipeline when config is None."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=None,
            is_training=True,
        )

        assert dataset.augmentation_pipeline is None

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_no_augmentation_for_validation(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """No augmentation for validation (is_training=False)."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        augmentation_config = {
            "enabled": True,
            "characteristics": "industrial",
            "environment": "controlled",
            "intensity": "medium",
        }

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=augmentation_config,
            is_training=False,  # Validation
        )

        assert dataset.augmentation_pipeline is None

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_augmentation_enabled_for_training(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """Augmentation enabled for training when configured."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        mock_pipeline = Mock()
        mock_registry_instance = Mock()
        mock_registry_instance.get_pipeline.return_value = mock_pipeline
        mock_registry.return_value = mock_registry_instance

        augmentation_config = {
            "enabled": True,
            "characteristics": "industrial",
            "environment": "controlled",
            "intensity": "medium",
        }

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=augmentation_config,
            is_training=True,
        )

        assert dataset.augmentation_pipeline is not None

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_respects_return_boxes_from_info(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_boxes_only,
    ):
        """Uses has_boxes from dataset_info for return_boxes."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_boxes_only,  # has_masks=False
            model_names=["grounding_dino"],
            augmentation_config=None,
            is_training=True,
        )

        assert dataset.return_boxes is True
        assert dataset.return_masks is False

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_sam_single_object_sampling(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """sam_single_object_sampling parameter is passed to dataset."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=None,
            is_training=True,
            sam_single_object_sampling=True,
        )

        assert dataset.sam_single_object_sampling is True

    @patch("ml_engine.data.dataset_factory.create_preprocessor_from_models")
    @patch("ml_engine.data.dataset_factory.get_augmentation_registry")
    def test_dataset_length_matches_data(
        self,
        mock_registry,
        mock_create_preprocessor,
        normalized_coco_data,
        mock_path_resolver,
        dataset_info_combined,
    ):
        """Dataset length matches number of images in COCO data."""
        mock_preprocessor = Mock()
        mock_create_preprocessor.return_value = mock_preprocessor

        dataset = DatasetFactory.create_dataset(
            coco_data=normalized_coco_data,
            image_path_resolver=mock_path_resolver,
            dataset_info=dataset_info_combined,
            model_names=["sam"],
            augmentation_config=None,
            is_training=True,
        )

        assert len(dataset) == len(normalized_coco_data["images"])


# =============================================================================
# Integration Tests
# =============================================================================


class TestDatasetFactoryIntegration:
    """Integration tests for DatasetFactory with real components."""

    def test_end_to_end_dataset_creation(self, temp_dir_with_images):
        """Full workflow: create dataset and access a sample."""
        data = temp_dir_with_images
        coco_data = normalize_coco_annotations(data["coco_data"], in_place=False)

        def path_resolver(file_name):
            return str(data["images_dir"] / file_name)

        dataset_info = {"has_boxes": True, "has_masks": True, "num_classes": 2}

        # Patch the config path lookup
        with patch("ml_engine.data.preprocessing.load_config") as mock_load_config:
            mock_load_config.return_value = {
                "preprocessing": {
                    "sam": {
                        "input_size": {"height": 1024, "width": 1024},
                        "normalization": {
                            "mean": [123.675, 116.28, 103.53],
                            "std": [58.395, 57.12, 57.375],
                        },
                        "padding_value": 0,
                        "mask_output_size": 256,
                    }
                }
            }

            dataset = DatasetFactory.create_dataset(
                coco_data=coco_data,
                image_path_resolver=path_resolver,
                dataset_info=dataset_info,
                model_names=["sam"],
                augmentation_config=None,
                is_training=True,
            )

            # Should be able to access sample
            sample = dataset[0]

            assert "preprocessed" in sample
            assert "sam" in sample["preprocessed"]
