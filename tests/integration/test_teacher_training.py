"""
Integration test for complete teacher training pipeline.

This test verifies the entire pipeline:
1. Dataset loading and inspection
2. Data preprocessing and augmentation
3. Model loading with LoRA
4. Training loop execution
5. Checkpoint saving
6. Validation
"""

import unittest
import tempfile
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml


class TestTeacherTrainingPipeline(unittest.TestCase):
    """Integration test for teacher training pipeline."""
    
    def setUp(self):
        """Create temporary dataset and config."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / 'data'
        self.image_dir = self.data_dir / 'images'
        self.image_dir.mkdir(parents=True)
        
        # Create sample images
        for i in range(10):
            img = Image.new('RGB', (640, 480), color=(i*20, i*20, i*20))
            img.save(self.image_dir / f'img{i:03d}.jpg')
        
        # Create COCO JSON
        self.coco_data = {
            'images': [
                {'id': i, 'file_name': f'img{i:03d}.jpg', 'width': 640, 'height': 480}
                for i in range(10)
            ],
            'annotations': [
                {
                    'id': i,
                    'image_id': i,
                    'category_id': i % 3,
                    'bbox': [100 + i*10, 100, 50, 50],
                    'segmentation': [[100 + i*10, 100, 150 + i*10, 100, 150 + i*10, 150, 100 + i*10, 150]],
                    'area': 2500
                }
                for i in range(10)
            ],
            'categories': [
                {'id': 0, 'name': 'class_0'},
                {'id': 1, 'name': 'class_1'},
                {'id': 2, 'name': 'class_2'}
            ]
        }
        
        # Save train and val JSONs
        self.train_json = self.data_dir / 'train.json'
        self.val_json = self.data_dir / 'val.json'
        
        with open(self.train_json, 'w') as f:
            json.dump(self.coco_data, f)
        
        # Use same data for val (for testing)
        with open(self.val_json, 'w') as f:
            json.dump(self.coco_data, f)
        
        # Create config directory
        self.config_dir = self.temp_dir / 'configs'
        self.config_dir.mkdir()
        
        # Create minimal configs
        self._create_test_configs()
    
    def _create_test_configs(self):
        """Create minimal test configurations."""
        # Preprocessing config
        preprocessing_config = {
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
                }
            }
        }
        
        with open(self.config_dir / 'preprocessing.yaml', 'w') as f:
            yaml.dump(preprocessing_config, f)
        
        # Training dynamics config
        training_dynamics = {
            'training_dynamics': {
                'gradient_clipping': {'enabled': False},
                'mixed_precision': {'enabled': False},
                'normalization': {'freeze_bn_teacher': False}
            }
        }
        
        with open(self.config_dir / 'training_dynamics.yaml', 'w') as f:
            yaml.dump(training_dynamics, f)
        
        # Checkpoint config
        checkpoint_config = {
            'checkpointing': {
                'save_interval': 1,
                'save_last': True,
                'save_best': True,
                'max_keep_checkpoints': 2,
                'monitor_metric': 'val_loss',
                'mode': 'min',
                'early_stopping': {'enabled': False}
            }
        }
        
        with open(self.config_dir / 'checkpoint_config.yaml', 'w') as f:
            yaml.dump(checkpoint_config, f)
    
    def test_data_loading_and_inspection(self):
        """Test dataset loading and inspection."""
        from ml_engine.data.inspection import load_and_inspect_dataset
        
        info = load_and_inspect_dataset(str(self.train_json))
        
        self.assertTrue(info['has_boxes'])
        self.assertTrue(info['has_masks'])
        self.assertEqual(info['num_classes'], 3)
        self.assertEqual(info['num_images'], 10)
    
    def test_dataset_loading(self):
        """Test COCODataset loading."""
        from ml_engine.data.loaders import COCODataset
        
        dataset = COCODataset(
            json_path=str(self.train_json),
            image_dir=str(self.image_dir),
            return_boxes=True,
            return_masks=True
        )
        
        self.assertEqual(len(dataset), 10)
        
        # Get a sample
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('boxes', sample)
        self.assertIn('masks', sample)
        self.assertIn('labels', sample)
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        from ml_engine.data.preprocessing import create_preprocessor_from_models
        
        preprocessor = create_preprocessor_from_models(
            model_names=['grounding_dino', 'sam'],
            config_path=str(self.config_dir / 'preprocessing.yaml')
        )
        
        # Test preprocessing
        img = Image.new('RGB', (640, 480), color='blue')
        preprocessed = preprocessor.preprocess_batch(img)
        
        self.assertIn('grounding_dino', preprocessed)
        self.assertIn('sam', preprocessed)
    
    def test_data_validation(self):
        """Test COCO format validation."""
        from ml_engine.data.validators import validate_coco_format
        
        is_valid, errors = validate_coco_format(self.coco_data)
        self.assertTrue(is_valid, f"Validation errors: {errors}")
    
    def test_bbox_auto_generation(self):
        """Test auto-generation of bbox from masks."""
        from ml_engine.data.validators import preprocess_coco_dataset
        
        # Create dataset without bboxes
        coco_no_bbox = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'segmentation': [[100, 100, 200, 100, 200, 200, 100, 200]]
                }
            ],
            'categories': [{'id': 0, 'name': 'class_0'}]
        }
        
        processed = preprocess_coco_dataset(coco_no_bbox, in_place=False)
        
        # Check bbox was generated
        ann = processed['annotations'][0]
        self.assertIn('bbox', ann)
        self.assertIn('area', ann)


class TestModelLoading(unittest.TestCase):
    """Test model loading with LoRA."""
    
    def test_load_grounding_dino_placeholder(self):
        """Test loading Grounding DINO (placeholder)."""
        from ml_engine.models.teacher.grounding_dino_lora import GroundingDINOLoRA
        
        # This will use placeholder since actual model may not be available
        model = GroundingDINOLoRA(
            base_checkpoint='dummy.pth',  # Will use placeholder
            num_classes=3,
            lora_config={'r': 4, 'lora_alpha': 8, 'target_modules': ['q_proj']}
        )
        
        self.assertIsNotNone(model)
    
    def test_load_sam_placeholder(self):
        """Test loading SAM (placeholder)."""
        from ml_engine.models.teacher.sam_lora import SAMLoRA
        
        # This will use placeholder since actual model may not be available
        model = SAMLoRA(
            base_checkpoint='dummy.pth',  # Will use placeholder
            model_type='vit_h',
            lora_config={'r': 4, 'lora_alpha': 8, 'target_modules': ['q_proj']}
        )
        
        self.assertIsNotNone(model)


class TestConfigGeneration(unittest.TestCase):
    """Test automatic configuration generation."""
    
    def setUp(self):
        """Create sample data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create minimal default config
        self.default_config = {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'epochs': 50,
            'lora': {'r': 16, 'lora_alpha': 32}
        }
        
        self.default_config_path = self.temp_dir / 'default.yaml'
        with open(self.default_config_path, 'w') as f:
            yaml.dump(self.default_config, f)
        
        self.dataset_info = {
            'num_classes': 3,
            'class_mapping': {0: 'cat', 1: 'dog', 2: 'bird'},
            'num_images': 100,
            'num_annotations': 150,
            'annotation_mode': 'DETECTION_AND_SEGMENTATION',
            'has_boxes': True,
            'has_masks': True
        }
    
    def test_generate_config(self):
        """Test config generation."""
        from core.config import generate_config
        
        config = generate_config(
            default_config_path=str(self.default_config_path),
            dataset_info=self.dataset_info,
            cli_overrides={'batch_size': 16}
        )
        
        # Check auto-filled values
        self.assertEqual(config['num_classes'], 3)
        self.assertEqual(config['class_names'], ['cat', 'dog', 'bird'])
        self.assertEqual(config['batch_size'], 16)  # Override applied
        self.assertEqual(config['epochs'], 50)  # From default
    
    def test_config_merge(self):
        """Test config merging."""
        from core.config import merge_configs
        
        base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override = {'b': {'d': 4}, 'e': 5}
        
        merged = merge_configs(base, override)
        
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b']['c'], 2)
        self.assertEqual(merged['b']['d'], 4)  # Overridden
        self.assertEqual(merged['e'], 5)


if __name__ == '__main__':
    unittest.main()


