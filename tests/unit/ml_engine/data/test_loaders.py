"""
Unit tests for ml_engine.data.loaders module.

Tests:
- COCODataset: Base dataset class for COCO format data
- TeacherDataset: Extended dataset with preprocessing/augmentation
- collate_fn: Custom batch collation for variable-sized data
- create_dataloader: DataLoader factory function
"""

from pathlib import Path
from unittest.mock import Mock
import tempfile
import shutil
import pytest
import torch
import numpy as np
from PIL import Image

from ml_engine.data.loaders import (
    COCODataset,
    TeacherDataset,
    collate_fn,
    create_dataloader,
)
from ml_engine.data.validators import normalize_coco_annotations


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def normalized_coco_data(valid_coco_data_combined):
    """COCO data normalized to compressed RLE format."""
    return normalize_coco_annotations(valid_coco_data_combined, in_place=False)


@pytest.fixture
def temp_images_with_data(valid_coco_data_combined):
    """
    Create temporary directory with images matching COCO data.
    
    Returns tuple of (images_dir, coco_data, path_resolver).
    """
    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir)

    for img_info in valid_coco_data_combined['images']:
        img_path = images_dir / img_info['file_name']
        # Create image with correct dimensions
        img = Image.new('RGB', (img_info['width'], img_info['height']), color='blue')
        img.save(img_path)

    def path_resolver(file_name: str) -> str:
        return str(images_dir / file_name)

    yield images_dir, valid_coco_data_combined, path_resolver

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def normalized_data_with_images(temp_images_with_data):
    """Normalized COCO data with corresponding images."""
    images_dir, coco_data, path_resolver = temp_images_with_data
    normalized = normalize_coco_annotations(coco_data, in_place=False)
    return images_dir, normalized, path_resolver


# =============================================================================
# COCODataset Tests
# =============================================================================

class TestCOCODataset:
    """Tests for COCODataset class."""

    def test_initialization(self, normalized_data_with_images):
        """Dataset initializes correctly."""
        images_dir, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            return_boxes=True,
            return_masks=True
        )

        assert len(dataset) == len(coco_data['images'])

    def test_getitem_returns_dict(self, normalized_data_with_images):
        """__getitem__ returns dictionary."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)
        sample = dataset[0]

        assert isinstance(sample, dict)

    def test_getitem_contains_image(self, normalized_data_with_images):
        """Sample contains PIL Image."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)
        sample = dataset[0]

        assert 'image' in sample
        assert isinstance(sample['image'], Image.Image)


    def test_getitem_contains_metadata(self, normalized_data_with_images):
        """Sample contains required metadata."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)
        sample = dataset[0]

        assert sample['image_id'] == 0
        assert sample['file_name'] == 'img_0.jpg'
        assert sample['image_size'] == (640, 480)
        assert len(sample['labels']) == 2
        assert sample['labels'][0] == 0
        assert sample['labels'][1] == 1

    def test_getitem_with_boxes(self, normalized_data_with_images):
        """Sample contains boxes when return_boxes=True."""
        _, coco_data, path_resolver = normalized_data_with_images
        dataset = COCODataset(coco_data, path_resolver, return_boxes=True)
        sample = dataset[0]

        assert 'boxes' in sample
        assert isinstance(sample['boxes'], list)
        assert sample['boxes'][0] == [100, 100, 50, 50]
        assert sample['boxes'][1] == [200, 200, 80, 60]

    def test_getitem_with_masks(self, normalized_data_with_images):
        """Sample contains masks when return_masks=True."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver, return_masks=True)
        sample = dataset[0]

        assert 'masks' in sample
        assert isinstance(sample['masks'], list)

        first_mask = sample['masks'][0]
        indices = np.argwhere(first_mask == 1)
        assert len(indices) == 2500
        assert indices[:,0].min() == 100
        assert indices[:,0].max() == 149
        assert indices[:,1].min() == 100
        assert indices[:,1].max() == 149

        second_mask = sample['masks'][1]
        indices = np.argwhere(second_mask == 1)
        assert len(indices) == 4800
        assert indices[:,0].min() == 200
        assert indices[:,0].max() == 259
        assert indices[:,1].min() == 200
        assert indices[:,1].max() == 279

    def test_getitem_without_boxes(self, normalized_data_with_images):
        """Sample excludes boxes when return_boxes=False."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver, return_boxes=False)
        sample = dataset[0]

        assert 'boxes' not in sample

    def test_getitem_without_masks(self, normalized_data_with_images):
        """Sample excludes masks when return_masks=False."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver, return_masks=False)
        sample = dataset[0]

        assert 'masks' not in sample

    def test_masks_are_numpy(self, normalized_data_with_images):
        """Masks are numpy arrays."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver, return_masks=True)
        sample = dataset[0]

        if sample['masks']:
            mask = sample['masks'][0]
            assert isinstance(mask, np.ndarray)

    def test_labels_are_category_ids(self, normalized_data_with_images):
        """Labels are category IDs (not indices)."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)
        sample = dataset[0]

        # Labels should be category IDs from the data
        valid_cat_ids = {cat['id'] for cat in coco_data['categories']}
        for label in sample['labels']:
            assert label in valid_cat_ids

    def test_class_names_attribute(self, normalized_data_with_images):
        """Dataset has class_names attribute."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)

        assert hasattr(dataset, 'class_names')
        assert len(dataset.class_names) == len(coco_data['categories'])
        assert dataset.class_names == ['ear', 'defect']

    def test_image_not_found_raises(self, normalized_data_with_images):
        """Raises FileNotFoundError for missing image."""
        _, coco_data, _ = normalized_data_with_images

        # Path resolver returns nonexistent path
        def bad_resolver(file_name):
            return '/nonexistent/path/' + file_name

        dataset = COCODataset(coco_data, bad_resolver)

        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_iteration(self, normalized_data_with_images):
        """Can iterate through dataset."""
        _, coco_data, path_resolver = normalized_data_with_images

        dataset = COCODataset(coco_data, path_resolver)

        count = 0
        for sample in dataset:
            count += 1
            assert 'image' in sample

        assert count == len(dataset)


# =============================================================================
# TeacherDataset Tests
# =============================================================================

class TestTeacherDataset:
    """Tests for TeacherDataset class."""

    def test_inherits_from_coco_dataset(self):
        """TeacherDataset inherits from COCODataset."""
        assert issubclass(TeacherDataset, COCODataset)

    def test_initialization_with_preprocessor(self, normalized_data_with_images):
        """Initializes with preprocessor."""
        _, coco_data, path_resolver = normalized_data_with_images

        # Mock preprocessor - new interface returns dict with all data
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.zeros((1, 4), dtype=np.float32),
                'masks': np.zeros((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        assert dataset.preprocessor is not None

    def test_getitem_returns_preprocessed(self, normalized_data_with_images):
        """Sample contains 'preprocessed' key with model data."""
        _, coco_data, path_resolver = normalized_data_with_images

        # Mock preprocessor - new interface returns dict with all data
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.zeros((1, 4), dtype=np.float32),
                'masks': np.zeros((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]

        assert 'preprocessed' in sample
        assert isinstance(sample['preprocessed'], dict)

    def test_augmentation_applied(self, normalized_data_with_images):
        """Augmentation pipeline is applied when provided."""
        _, coco_data, path_resolver = normalized_data_with_images

        # Mock preprocessor - new interface returns dict with all data
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.zeros((1, 4), dtype=np.float32),
                'masks': np.zeros((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        # Mock augmentation
        mock_augmentation = Mock()
        mock_augmentation.return_value = {
            'image': np.zeros((480, 640, 3), dtype=np.uint8),
            'masks': [np.zeros((480, 640), dtype=np.uint8)],
            'bboxes': [[100, 100, 50, 50]]
        }

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor,
            augmentation_pipeline=mock_augmentation
        )

        _ = dataset[0]

        # Augmentation should be called
        mock_augmentation.assert_called()

    # =========================================================================
    # SAM Single Object Sampling Tests
    # =========================================================================

    def test_sam_single_object_sampling_picks_one_mask(self, normalized_data_with_images):
        """When sam_single_object_sampling=True, SAM output has only 1 object."""
        _, coco_data, path_resolver = normalized_data_with_images

        # Mock preprocessor returning multiple objects
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50], [100, 100, 60, 60]], dtype=np.float32),
                'masks': np.ones((2, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor,
            sam_single_object_sampling=True
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have exactly 1 object
        assert sam_data['boxes'].shape[0] == 1
        assert sam_data['masks'].shape[0] == 1
        assert len(sam_data['labels']) == 1

    def test_sam_single_object_sampling_disabled_keeps_all(self, normalized_data_with_images):
        """When sam_single_object_sampling=False, SAM output has all objects."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50], [100, 100, 60, 60]], dtype=np.float32),
                'masks': np.ones((2, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor,
            sam_single_object_sampling=False  # Disabled
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have all objects
        assert sam_data['boxes'].shape[0] == 2
        assert sam_data['masks'].shape[0] == 2

    # =========================================================================
    # Multi-Model Output Tests
    # =========================================================================

    def test_multi_model_output_contains_all_models(self, normalized_data_with_images):
        """Output contains data for all configured models (SAM + DINO)."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            },
            'grounding_dino': {
                'image': torch.zeros(3, 800, 1067),
                'boxes': np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32),
                'masks': None,
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (800, 1067),
                    'model_name': 'grounding_dino'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]

        assert 'sam' in sample['preprocessed']
        assert 'grounding_dino' in sample['preprocessed']

        # Verify each model has required keys
        for model_name in ['sam', 'grounding_dino']:
            model_data = sample['preprocessed'][model_name]
            assert 'image' in model_data
            assert 'boxes' in model_data
            assert 'masks' in model_data
            assert 'labels' in model_data
            assert 'metadata' in model_data

    # =========================================================================
    # Empty/None Annotations Handling Tests
    # =========================================================================

    def test_handles_none_boxes_from_preprocessor(self, normalized_data_with_images):
        """Returns empty array when preprocessor returns None boxes."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': None,  # No boxes
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have empty boxes array, not None
        assert sam_data['boxes'] is not None
        assert sam_data['boxes'].shape == (0, 4)
        assert sam_data['boxes'].dtype == np.float32

    def test_handles_none_masks_from_preprocessor(self, normalized_data_with_images):
        """Returns empty array when preprocessor returns None masks."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': None,  # No masks
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have empty masks array with correct shape
        assert sam_data['masks'] is not None
        assert sam_data['masks'].shape[0] == 0
        assert sam_data['masks'].shape[1] == 1024  # final_size height
        assert sam_data['masks'].shape[2] == 1024  # final_size width

    # =========================================================================
    # Labels Tests
    # =========================================================================

    def test_labels_included_in_output(self, normalized_data_with_images):
        """Labels array is included in each model's output."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50], [100, 100, 60, 60]], dtype=np.float32),
                'masks': np.ones((2, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        assert 'labels' in sam_data
        assert isinstance(sam_data['labels'], np.ndarray)
        assert sam_data['labels'].dtype == np.int64

    def test_labels_empty_when_no_annotations(self, normalized_data_with_images):
        """Labels array is empty when no annotations."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': None,
                'masks': None,
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        assert len(sam_data['labels']) == 2

    # =========================================================================
    # return_boxes / return_masks Flag Tests
    # =========================================================================

    def test_return_boxes_false_gives_empty_boxes(self, normalized_data_with_images):
        """return_boxes=False results in empty boxes array regardless of input."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),  # Has boxes
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor,
            return_boxes=False  # Disable boxes
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have empty boxes array
        assert sam_data['boxes'].shape == (0, 4)

    def test_return_masks_false_gives_empty_masks(self, normalized_data_with_images):
        """return_masks=False results in empty masks array regardless of input."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),  # Has masks
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor,
            return_masks=False  # Disable masks
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        # Should have empty masks array
        assert sam_data['masks'].shape[0] == 0

    # =========================================================================
    # Preprocessor Input Verification Tests
    # =========================================================================

    def test_preprocessor_receives_pil_image(self, normalized_data_with_images):
        """Preprocessor receives PIL Image, not numpy array."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': None,
                'masks': None,
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (1024, 1024),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        _ = dataset[0]

        # Verify preprocessor was called with PIL Image
        call_args = mock_preprocessor.preprocess_batch.call_args
        image_arg = call_args[0][0]  # First positional arg
        assert isinstance(image_arg, Image.Image)

    def test_preprocessor_receives_numpy_boxes(self, normalized_data_with_images):
        """Preprocessor receives numpy array for boxes, not list."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        _ = dataset[0]

        # Verify preprocessor was called with numpy array for boxes
        call_args = mock_preprocessor.preprocess_batch.call_args
        boxes_arg = call_args[1]['boxes']  # keyword arg 'boxes'
        assert isinstance(boxes_arg, np.ndarray)
        assert boxes_arg.dtype == np.float32

    def test_preprocessor_receives_stacked_masks(self, normalized_data_with_images):
        """Preprocessor receives stacked numpy array for masks, not list."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        _ = dataset[0]

        # Verify preprocessor was called with stacked numpy array for masks
        call_args = mock_preprocessor.preprocess_batch.call_args
        masks_arg = call_args[1]['masks']  # keyword arg 'masks'
        assert isinstance(masks_arg, np.ndarray)
        assert masks_arg.ndim == 3  # (N, H, W)

    # =========================================================================
    # Output Data Types Tests
    # =========================================================================

    def test_output_image_is_tensor(self, normalized_data_with_images):
        """Output image is torch.Tensor."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {
                    'original_size': (480, 640),
                    'final_size': (256, 256),
                    'model_name': 'sam'
                }
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        assert isinstance(sam_data['image'], torch.Tensor)

    def test_output_boxes_dtype_float32(self, normalized_data_with_images):
        """Output boxes have dtype float32."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {'original_size': (480, 640), 'final_size': (256, 256), 'model_name': 'sam'}
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        assert sam_data['boxes'].dtype == np.float32

    def test_output_labels_dtype_int64(self, normalized_data_with_images):
        """Output labels have dtype int64."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': np.array([[10, 10, 50, 50]], dtype=np.float32),
                'masks': np.ones((1, 256, 256), dtype=np.uint8),
                'metadata': {'original_size': (480, 640), 'final_size': (256, 256), 'model_name': 'sam'}
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]
        sam_data = sample['preprocessed']['sam']

        assert sam_data['labels'].dtype == np.int64

    # =========================================================================
    # Output Structure Tests
    # =========================================================================

    def test_output_contains_metadata_keys(self, normalized_data_with_images):
        """Output contains image_id, file_name, image_size, preprocessed."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': None,
                'masks': None,
                'metadata': {'original_size': (480, 640), 'final_size': (1024, 1024), 'model_name': 'sam'}
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]

        assert 'image_id' in sample
        assert 'file_name' in sample
        assert 'image_size' in sample
        assert 'preprocessed' in sample

    def test_output_excludes_raw_image(self, normalized_data_with_images):
        """Output does NOT contain raw 'image' key (only preprocessed)."""
        _, coco_data, path_resolver = normalized_data_with_images

        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_batch = Mock(return_value={
            'sam': {
                'image': torch.zeros(3, 1024, 1024),
                'boxes': None,
                'masks': None,
                'metadata': {'original_size': (480, 640), 'final_size': (1024, 1024), 'model_name': 'sam'}
            }
        })

        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=path_resolver,
            preprocessor=mock_preprocessor
        )

        sample = dataset[0]

        # Raw 'image' should NOT be at top level (only in preprocessed)
        assert 'image' not in sample


# =============================================================================
# collate_fn Tests
# =============================================================================

class TestCollateFn:
    """Tests for collate_fn function."""

    def test_batches_metadata(self):
        """Collates metadata lists correctly."""
        batch = [
            {
                'image_id': 0,
                'file_name': 'img_0.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            },
            {
                'image_id': 1,
                'file_name': 'img_1.jpg',
                'image_size': (800, 600),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[200, 150, 80, 60]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([1], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'image_ids' in collated
        assert 'file_names' in collated
        assert 'image_sizes' in collated
        assert len(collated['image_ids']) == 2
        assert len(collated['file_names']) == 2

    def test_batches_preprocessed_data(self):
        """Collates preprocessed model data correctly."""
        batch = [
            {
                'image_id': 0,
                'file_name': 'img_0.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            },
            {
                'image_id': 1,
                'file_name': 'img_1.jpg',
                'image_size': (800, 600),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[200, 150, 80, 60]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([1], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'preprocessed' in collated
        assert 'sam' in collated['preprocessed']

    def test_stacks_sam_images(self):
        """SAM images are stacked into batch tensor."""
        batch = [
            {
                'image_id': i,
                'file_name': f'img_{i}.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.randn(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
            for i in range(4)
        ]
        
        collated = collate_fn(batch)
        sam_images = collated['preprocessed']['sam']['images']
        
        # Should be [B, C, H, W]
        assert sam_images.shape == (4, 3, 1024, 1024)

    def test_pads_boxes_to_max(self):
        """Boxes are padded to max number of objects."""
        batch = [
            {
                'image_id': 0,
                'file_name': 'img_0.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),  # 1 box
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            },
            {
                'image_id': 1,
                'file_name': 'img_1.jpg',
                'image_size': (800, 600),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50], [200, 200, 60, 60], [300, 300, 70, 70]], dtype=np.float32),  # 3 boxes
                        'masks': np.zeros((3, 256, 256), dtype=np.float32),
                        'labels': np.array([0, 1, 0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
        ]
        
        collated = collate_fn(batch)
        boxes = collated['preprocessed']['sam']['boxes']
        
        # Should be [B, max_objs, 4] where max_objs=3
        assert boxes.shape == (2, 3, 4)

    def test_pads_labels_with_ignore(self):
        """Labels are padded with -1 (ignore index)."""
        batch = [
            {
                'image_id': 0,
                'file_name': 'img_0.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),  # 1 box
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            },
            {
                'image_id': 1,
                'file_name': 'img_1.jpg',
                'image_size': (800, 600),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50], [200, 200, 60, 60]], dtype=np.float32),  # 2 boxes
                        'masks': np.zeros((2, 256, 256), dtype=np.float32),
                        'labels': np.array([0, 1], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
        ]
        
        collated = collate_fn(batch)
        labels = collated['preprocessed']['sam']['labels']
        
        # First sample should have padding with -1
        assert labels[0, 1].item() == -1  # Padded position

    def test_handles_empty_boxes(self):
        """Handles samples with no boxes."""
        batch = [
            {
                'image_id': 0,
                'file_name': 'img_0.jpg',
                'image_size': (640, 480),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.zeros((0, 4), dtype=np.float32),  # Empty
                        'masks': np.zeros((0, 256, 256), dtype=np.float32),
                        'labels': np.array([], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            },
            {
                'image_id': 1,
                'file_name': 'img_1.jpg',
                'image_size': (800, 600),
                'preprocessed': {
                    'sam': {
                        'image': torch.zeros(3, 1024, 1024),
                        'boxes': np.array([[100, 100, 50, 50]], dtype=np.float32),
                        'masks': np.zeros((1, 256, 256), dtype=np.float32),
                        'labels': np.array([0], dtype=np.int64),
                        'metadata': {'final_size': (1024, 1024)}
                    }
                }
            }
        ]
        
        collated = collate_fn(batch)
        
        # Should still produce valid tensors
        assert 'boxes' in collated['preprocessed']['sam']
        assert collated['preprocessed']['sam']['boxes'].shape[0] == 2


# =============================================================================
# create_dataloader Tests
# =============================================================================

class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_returns_dataloader(self, normalized_data_with_images):
        """Returns a DataLoader instance."""
        _, coco_data, path_resolver = normalized_data_with_images
        
        # Create simple mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={
            'image_id': 0,
            'file_name': 'img.jpg',
            'image_size': (640, 480),
            'preprocessed': {
                'sam': {
                    'image': torch.zeros(3, 1024, 1024),
                    'boxes': np.zeros((1, 4), dtype=np.float32),
                    'masks': np.zeros((1, 256, 256), dtype=np.float32),
                    'labels': np.array([0], dtype=np.int64),
                    'metadata': {'final_size': (1024, 1024)}
                }
            }
        })
        
        dataloader = create_dataloader(mock_dataset, batch_size=2, num_workers=0)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)

    def test_batch_size_respected(self, normalized_data_with_images):
        """DataLoader uses specified batch size."""
        _, coco_data, path_resolver = normalized_data_with_images
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        
        dataloader = create_dataloader(mock_dataset, batch_size=4, num_workers=0)
        
        assert dataloader.batch_size == 4

    def test_uses_custom_collate_fn(self, normalized_data_with_images):
        """DataLoader uses our custom collate_fn."""
        _, coco_data, path_resolver = normalized_data_with_images
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        
        dataloader = create_dataloader(mock_dataset, num_workers=0)
        
        assert dataloader.collate_fn == collate_fn

    def test_shuffle_parameter(self, normalized_data_with_images):
        """Shuffle parameter is respected."""
        _, coco_data, path_resolver = normalized_data_with_images
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        
        dataloader_shuffle = create_dataloader(mock_dataset, shuffle=True, num_workers=0)
        dataloader_no_shuffle = create_dataloader(mock_dataset, shuffle=False, num_workers=0)
        
        # Both should be valid DataLoaders
        assert isinstance(dataloader_shuffle, torch.utils.data.DataLoader)
        assert isinstance(dataloader_no_shuffle, torch.utils.data.DataLoader)
