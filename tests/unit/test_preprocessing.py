"""
Unit tests for preprocessing pipeline.

Tests:
- MultiModelPreprocessor
- SingleModelPreprocessor
- Model-specific preprocessing
- Normalization and resizing
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import yaml


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing components."""
    
    def setUp(self):
        """Create test config and sample image."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'preprocessing.yaml'
        
        config = {
            'preprocessing': {
                'grounding_dino': {
                    'input_size': {'min_size': 800, 'max_size': 1333},
                    'normalization': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225],
                        'pixel_range': [0, 1]
                    },
                    'resize_mode': 'keep_aspect_ratio',
                    'padding_value': 0,
                    'pixel_format': 'RGB'
                },
                'sam': {
                    'input_size': {'height': 1024, 'width': 1024},
                    'normalization': {
                        'mean': [123.675, 116.28, 103.53],
                        'std': [58.395, 57.12, 57.375],
                        'pixel_range': [0, 255]
                    },
                    'resize_mode': 'resize_longest_side',
                    'padding_value': 0,
                    'pixel_format': 'RGB'
                },
                'yolov8': {
                    'input_size': 640,
                    'normalization': {
                        'mean': [0.0, 0.0, 0.0],
                        'std': [1.0, 1.0, 1.0],
                        'pixel_range': [0, 1]
                    },
                    'resize_mode': 'letterbox',
                    'padding_value': 114,
                    'pixel_format': 'RGB'
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create sample image
        self.sample_image = Image.new('RGB', (640, 480), color='red')
    
    def test_multi_model_preprocessor_initialization(self):
        """Test MultiModelPreprocessor initialization."""
        from ml_engine.data.preprocessing import MultiModelPreprocessor
        
        preprocessor = MultiModelPreprocessor(
            active_models=['grounding_dino', 'sam'],
            config_path=str(self.config_path)
        )
        
        self.assertIn('grounding_dino', preprocessor.preprocessors)
        self.assertIn('sam', preprocessor.preprocessors)
        self.assertEqual(len(preprocessor.preprocessors), 2)
    
    def test_single_model_preprocessing_dino(self):
        """Test preprocessing for Grounding DINO."""
        from ml_engine.data.preprocessing import MultiModelPreprocessor
        
        preprocessor = MultiModelPreprocessor(
            active_models=['grounding_dino'],
            config_path=str(self.config_path)
        )
        
        tensor, metadata = preprocessor.preprocess_for_model(self.sample_image, 'grounding_dino')
        
        # Check tensor shape and type
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 3)  # [C, H, W]
        self.assertEqual(tensor.size(0), 3)  # 3 channels
        
        # Check metadata
        self.assertIn('original_size', metadata)
        self.assertIn('scale_factor', metadata)
        self.assertIn('final_size', metadata)
    
    def test_single_model_preprocessing_sam(self):
        """Test preprocessing for SAM."""
        from ml_engine.data.preprocessing import MultiModelPreprocessor
        
        preprocessor = MultiModelPreprocessor(
            active_models=['sam'],
            config_path=str(self.config_path)
        )
        
        tensor, metadata = preprocessor.preprocess_for_model(self.sample_image, 'sam')
        
        # SAM should output 1024x1024
        self.assertEqual(tensor.size(1), 1024)
        self.assertEqual(tensor.size(2), 1024)
    
    def test_batch_preprocessing(self):
        """Test preprocessing for multiple models at once."""
        from ml_engine.data.preprocessing import MultiModelPreprocessor
        
        preprocessor = MultiModelPreprocessor(
            active_models=['grounding_dino', 'sam'],
            config_path=str(self.config_path)
        )
        
        preprocessed = preprocessor.preprocess_batch(self.sample_image)
        
        # Should have results for both models
        self.assertIn('grounding_dino', preprocessed)
        self.assertIn('sam', preprocessed)
        
        # Each should be a tuple of (tensor, metadata)
        dino_tensor, dino_meta = preprocessed['grounding_dino']
        sam_tensor, sam_meta = preprocessed['sam']
        
        self.assertIsInstance(dino_tensor, torch.Tensor)
        self.assertIsInstance(sam_tensor, torch.Tensor)
        self.assertIsInstance(dino_meta, dict)
        self.assertIsInstance(sam_meta, dict)
    
    def test_normalization_different_models(self):
        """Test that different models get different normalization."""
        from ml_engine.data.preprocessing import MultiModelPreprocessor
        
        preprocessor = MultiModelPreprocessor(
            active_models=['grounding_dino', 'sam'],
            config_path=str(self.config_path)
        )
        
        preprocessed = preprocessor.preprocess_batch(self.sample_image)
        
        dino_tensor = preprocessed['grounding_dino'][0]
        sam_tensor = preprocessed['sam'][0]
        
        # Different normalization should give different results
        self.assertFalse(torch.allclose(dino_tensor, sam_tensor))


class TestBboxGeneration(unittest.TestCase):
    """Test automatic bbox generation from masks."""
    
    def test_compute_bbox_from_polygon(self):
        """Test bbox generation from polygon."""
        polygon = [[100, 100, 200, 100, 200, 200, 100, 200]]
        bbox = compute_bbox_from_mask(polygon)
        
        self.assertEqual(len(bbox), 4)
        self.assertAlmostEqual(bbox[0], 100.0)
        self.assertAlmostEqual(bbox[1], 100.0)
        self.assertAlmostEqual(bbox[2], 100.0)
        self.assertAlmostEqual(bbox[3], 100.0)
    
    def test_compute_bbox_empty_polygon(self):
        """Test bbox generation from empty polygon."""
        polygon = [[]]
        bbox = compute_bbox_from_mask(polygon)
        
        self.assertEqual(bbox, [0, 0, 0, 0])
    
    def test_preprocess_coco_adds_bbox(self):
        """Test that preprocessing adds bbox to mask-only annotations."""
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'segmentation': [[100, 100, 200, 100, 200, 200, 100, 200]]
                }
            ],
            'categories': [{'id': 0, 'name': 'cat'}]
        }
        
        processed = preprocess_coco_dataset(coco_data, in_place=False)
        
        ann = processed['annotations'][0]
        self.assertIn('bbox', ann)
        self.assertIn('area', ann)


if __name__ == '__main__':
    unittest.main()


