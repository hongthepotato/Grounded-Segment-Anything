"""
Unit tests for ml_engine.data.manager module.

Tests:
- DataManager: Central orchestrator for data operations
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from ml_engine.data.manager import DataManager
from ml_engine.data.validators import normalize_coco_annotations


# =============================================================================
# Fixtures for Direct Construction
# =============================================================================

@pytest.fixture
def normalized_coco_data(valid_coco_data_combined):
    """Pre-normalized COCO data for direct construction tests."""
    return normalize_coco_annotations(valid_coco_data_combined, in_place=False)


@pytest.fixture
def simple_image_path_map():
    """Simple image path map for testing."""
    return {
        'img_0.jpg': '/fake/path/img_0.jpg',
        'img_1.jpg': '/fake/path/img_1.jpg',
        'img_2.jpg': '/fake/path/img_2.jpg',
    }


# =============================================================================
# Fixtures for from_file() Tests (need real files)
# =============================================================================

@pytest.fixture
def temp_data_environment(valid_coco_data_combined):
    """
    Create complete temporary environment for from_file() testing.
    
    Creates:
    - COCO JSON file
    - Image files matching the COCO data
    - Image paths list (simulating frontend input)
    """
    temp_dir = tempfile.mkdtemp()

    # Create image directory
    images_dir = Path(temp_dir) / 'images'
    images_dir.mkdir()

    # Create COCO JSON file
    coco_path = Path(temp_dir) / 'annotations.json'
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(valid_coco_data_combined, f)

    # Create images and build image_paths list
    image_paths = []
    for img_info in valid_coco_data_combined['images']:
        img_path = images_dir / img_info['file_name']
        img = Image.new('RGB', (img_info['width'], img_info['height']), color='blue')
        img.save(img_path)
        image_paths.append(img_info['file_name'])

    yield {
        'temp_dir': temp_dir,
        'coco_path': str(coco_path),
        'images_dir': images_dir,
        'image_paths': image_paths,
        'coco_data': valid_coco_data_combined
    }

    shutil.rmtree(temp_dir)


# =============================================================================
# DataManager Direct Construction Tests
# =============================================================================

class TestDataManagerDirectConstruction:
    """Tests for DataManager.__init__ - direct construction with pre-processed data."""

    def test_stores_raw_data(self, normalized_coco_data, simple_image_path_map):
        """Direct construction stores raw data."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        assert manager.raw_data is normalized_coco_data

    def test_stores_image_path_map(self, normalized_coco_data, simple_image_path_map):
        """Direct construction stores image path map."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        assert manager.image_path_map == simple_image_path_map

    def test_stores_annotation_mode(self, normalized_coco_data, simple_image_path_map):
        """Direct construction stores annotation mode."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='segmentation'
        )

        assert manager.original_annotation_mode == 'segmentation'

    def test_computes_dataset_info(self, normalized_coco_data, simple_image_path_map):
        """Direct construction computes dataset info."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        assert manager.dataset_info is not None
        assert manager.dataset_info['has_boxes'] is True
        assert manager.dataset_info['has_masks'] is True
        assert manager.dataset_info['num_classes'] == 2
        assert manager.dataset_info['class_mapping'] == {0: 'ear', 1: 'defect'}
        assert manager.dataset_info['num_images'] == 3
        assert manager.dataset_info['num_annotations'] == 4
        assert manager.dataset_info['annotation_mode'] == 'DETECTION_AND_SEGMENTATION'
        assert manager.dataset_info['class_counts'] == {0: 2, 1: 2}



    def test_computes_quality_report(self, normalized_coco_data, simple_image_path_map):
        """Direct construction computes quality report."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        assert manager.quality_report is not None
        assert manager.quality_report['total_images'] == 3
        assert manager.quality_report['total_annotations'] == 4
        assert manager.quality_report['images_without_annotations'] == 0
        assert 'total_images' in manager.quality_report

    def test_default_all_split(self, normalized_coco_data, simple_image_path_map):
        """Without splits, creates 'all' split."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined',
            splits=None
        )

        assert 'all' in manager.splits
        assert len(manager.splits) == 1

    def test_uses_provided_splits(self, normalized_coco_data, simple_image_path_map):
        """Uses provided splits when given."""
        custom_splits = {
            'train': normalized_coco_data,
            'val': normalized_coco_data,
        }

        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined',
            splits=custom_splits
        )

        assert 'train' in manager.splits
        assert 'val' in manager.splits
        assert len(manager.splits) == 2


# =============================================================================
# DataManager.get_dataset_info Tests
# =============================================================================

class TestDataManagerGetDatasetInfo:
    """Tests for DataManager.get_dataset_info method."""

    def test_returns_dataset_info(self, normalized_coco_data, simple_image_path_map):
        """get_dataset_info returns inspection results."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        info = manager.get_dataset_info()

        assert info is not None


# =============================================================================
# DataManager.get_required_models Tests
# =============================================================================

class TestDataManagerGetRequiredModels:
    """Tests for DataManager.get_required_models method."""

    def test_combined_returns_both_models(self, normalized_coco_data, simple_image_path_map):
        """Combined mode returns both grounding_dino and sam."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        models = manager.get_required_models()

        assert 'grounding_dino' in models
        assert 'sam' in models

    def test_detection_returns_dino_only(self, normalized_coco_data, simple_image_path_map):
        """Detection mode returns only grounding_dino."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='detection'
        )

        models = manager.get_required_models()

        assert 'grounding_dino' in models
        assert 'sam' not in models

    def test_segmentation_returns_sam_only(self, normalized_coco_data, simple_image_path_map):
        """Segmentation mode returns only sam."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='segmentation'
        )

        models = manager.get_required_models()

        assert 'sam' in models
        assert 'grounding_dino' not in models


# =============================================================================
# DataManager.get_quality_report Tests
# =============================================================================

class TestDataManagerGetQualityReport:
    """Tests for DataManager.get_quality_report method."""

    def test_returns_quality_report(self, normalized_coco_data, simple_image_path_map):
        """get_quality_report returns quality metrics."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        report = manager.get_quality_report()

        assert report is not None


# =============================================================================
# DataManager.get_split Tests
# =============================================================================

class TestDataManagerGetSplit:
    """Tests for DataManager.get_split method."""

    def test_get_all_split(self, normalized_coco_data, simple_image_path_map):
        """get_split('all') returns all data."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        all_data = manager.get_split('all')

        assert 'images' in all_data
        assert 'annotations' in all_data
        assert 'categories' in all_data

    def test_get_custom_split(self, normalized_coco_data, simple_image_path_map):
        """get_split returns custom splits."""
        custom_splits = {
            'train': {'images': [], 'annotations': [], 'categories': []},
            'val': {'images': [], 'annotations': [], 'categories': []},
        }

        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined',
            splits=custom_splits
        )

        train = manager.get_split('train')
        val = manager.get_split('val')

        assert train is not None
        assert val is not None

    def test_invalid_split_raises(self, normalized_coco_data, simple_image_path_map):
        """get_split with invalid name raises ValueError."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        with pytest.raises(ValueError):
            manager.get_split('nonexistent')


# =============================================================================
# DataManager.get_image_path Tests
# =============================================================================

class TestDataManagerGetImagePath:
    """Tests for DataManager.get_image_path method."""

    def test_returns_mapped_path(self, normalized_coco_data, simple_image_path_map):
        """get_image_path returns path from map."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        path = manager.get_image_path('img_0.jpg')

        assert path == '/fake/path/img_0.jpg'

    def test_fallback_to_transform(self, normalized_coco_data, simple_image_path_map):
        """get_image_path falls back to transform_image_path for unknown files."""
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        with patch('ml_engine.data.manager.transform_image_path') as mock_transform:
            mock_transform.return_value = '/transformed/path.jpg'
            path = manager.get_image_path('unknown.jpg')

        assert path == '/transformed/path.jpg'
        mock_transform.assert_called_once_with('unknown.jpg')


# =============================================================================
# DataManager.from_file Tests (need temp files)
# =============================================================================

class TestDataManagerFromFile:
    """Tests for DataManager.from_file factory method."""

    @patch('ml_engine.data.manager.transform_image_path')
    def test_from_file_success(self, mock_transform, temp_data_environment):
        """from_file loads and returns DataManager."""
        data = temp_data_environment
        mock_transform.side_effect = lambda p: str(data['images_dir'] / p)

        manager = DataManager.from_file(
            data_path=data['coco_path'],
            image_paths=data['image_paths']
        )

        assert manager is not None
        assert isinstance(manager, DataManager)

    @patch('ml_engine.data.manager.transform_image_path')
    def test_from_file_loads_data(self, mock_transform, temp_data_environment):
        """from_file correctly loads COCO data."""
        data = temp_data_environment
        mock_transform.side_effect = lambda p: str(data['images_dir'] / p)

        manager = DataManager.from_file(
            data_path=data['coco_path'],
            image_paths=data['image_paths']
        )

        assert manager.raw_data is not None
        assert len(manager.raw_data['images']) == 3
        assert len(manager.raw_data['annotations']) == 4
        assert len(manager.raw_data['categories']) == 2

    @patch('ml_engine.data.manager.transform_image_path')
    def test_from_file_with_splits(self, mock_transform, temp_data_environment):
        """from_file creates splits when configured.
        
        Note: With only 3 images, split function falls back to random split
        and allocates all images to train split (due to minimum split size logic).
        """
        data = temp_data_environment
        mock_transform.side_effect = lambda p: str(data['images_dir'] / p)

        manager = DataManager.from_file(
            data_path=data['coco_path'],
            image_paths=data['image_paths'],
            split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
        )

        # With 3 images, only train split is created
        assert 'train' in manager.splits
        assert len(manager.splits) == 1  # Only train split with small dataset
        train_split = manager.get_split('train')
        assert len(train_split['images']) == 3

    def test_from_file_nonexistent_raises(self):
        """from_file raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DataManager.from_file(
                data_path='/nonexistent/path.json',
                image_paths=['img1.jpg']
            )

    @patch('ml_engine.data.manager.transform_image_path')
    def test_from_file_invalid_coco_raises(self, mock_transform, temp_data_environment):
        """from_file raises ValueError for invalid COCO format."""
        data = temp_data_environment

        # Create invalid COCO file
        invalid_path = Path(data['temp_dir']) / 'invalid.json'
        with open(invalid_path, 'w', encoding='utf-8') as f:
            json.dump({'not': 'valid'}, f)

        mock_transform.side_effect = lambda p: str(data['images_dir'] / p)

        with pytest.raises(ValueError):
            DataManager.from_file(
                data_path=str(invalid_path),
                image_paths=data['image_paths']
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestDataManagerIntegration:
    """Integration tests for DataManager."""

    @patch('ml_engine.data.manager.transform_image_path')
    def test_full_from_file_workflow(self, mock_transform, temp_data_environment):
        """Test complete workflow with from_file."""
        data = temp_data_environment
        mock_transform.side_effect = lambda p: str(data['images_dir'] / p)

        # Load from file
        manager = DataManager.from_file(
            data_path=data['coco_path'],
            image_paths=data['image_paths'],
            split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
        )

        # Get info
        info = manager.get_dataset_info()
        assert info['has_boxes'] is True
        assert info['has_masks'] is True

        # Get models
        models = manager.get_required_models()
        assert 'grounding_dino' in models
        assert 'sam' in models

        # Get splits
        train = manager.get_split('train')
        assert len(train['images']) == 3

        # Resolve image path
        file_name = data['image_paths'][0]
        path = manager.get_image_path(file_name)
        assert Path(path).exists()

    def test_direct_construction_workflow(self, normalized_coco_data, simple_image_path_map):
        """Test complete workflow with direct construction (for testing)."""
        # Direct construction - no files needed!
        manager = DataManager(
            raw_data=normalized_coco_data,
            image_path_map=simple_image_path_map,
            original_annotation_mode='combined'
        )

        # All methods work without any file I/O
        info = manager.get_dataset_info()
        assert info['has_boxes'] is True
        assert info['has_masks'] is True

        models = manager.get_required_models()
        assert 'grounding_dino' in models
        assert 'sam' in models

        report = manager.get_quality_report()
        assert report['total_images'] == 3

        all_data = manager.get_split('all')
        assert len(all_data['images']) == 3
