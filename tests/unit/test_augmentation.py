"""
Unit tests for characteristic-based augmentation system.

Tests:
- Augmentation registry
- Characteristic translator
- Pipeline generation
- Augmentation application
"""

import unittest
import numpy as np
from PIL import Image


class TestAugmentationRegistry(unittest.TestCase):
    """Test augmentation registry."""
    
    def setUp(self):
        """Setup test data."""
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.sample_masks = [
            np.random.randint(0, 2, (480, 640), dtype=np.uint8),
            np.random.randint(0, 2, (480, 640), dtype=np.uint8)
        ]
        self.sample_boxes = np.array([[100, 100, 50, 50], [200, 200, 60, 60]], dtype=np.float32)
    
    def test_get_augmentation_registry(self):
        """Test getting augmentation registry."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        self.assertIsNotNone(registry)
    
    def test_get_pipeline_basic(self):
        """Test creating basic augmentation pipeline."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        pipeline = registry.get_pipeline(
            characteristics=["changes_shape"],
            environment={"lighting": "variable"},
            intensity="medium"
        )
        
        self.assertIsNotNone(pipeline)
    
    def test_pipeline_application(self):
        """Test applying augmentation pipeline."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        pipeline = registry.get_pipeline(
            characteristics=["changes_shape", "reflective_surface"],
            environment={"lighting": "variable"},
            intensity="low"
        )
        
        # Apply augmentation
        result = pipeline(
            image=self.sample_image,
            masks=self.sample_masks,
            bboxes=self.sample_boxes
        )
        
        # Check outputs
        self.assertIn('image', result)
        self.assertIn('masks', result)
        self.assertIn('bboxes', result)
        
        # Check shapes preserved
        self.assertEqual(result['image'].shape, self.sample_image.shape)
    
    def test_intensity_levels(self):
        """Test different intensity levels."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        
        for intensity in ['low', 'medium', 'high']:
            pipeline = registry.get_pipeline(
                characteristics=["changes_shape"],
                environment={},
                intensity=intensity
            )
            self.assertIsNotNone(pipeline)
    
    def test_invalid_intensity(self):
        """Test that invalid intensity raises error."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        
        with self.assertRaises(ValueError):
            pipeline = registry.get_pipeline(
                characteristics=["changes_shape"],
                environment={},
                intensity="invalid"
            )
    
    def test_multiple_characteristics(self):
        """Test combining multiple characteristics."""
        from augmentation import get_augmentation_registry
        
        registry = get_augmentation_registry()
        pipeline = registry.get_pipeline(
            characteristics=[
                "changes_shape",
                "reflective_surface",
                "low_contrast"
            ],
            environment={
                "lighting": "variable",
                "camera": "fixed"
            },
            intensity="medium"
        )
        
        # Apply to sample
        result = pipeline(
            image=self.sample_image,
            masks=self.sample_masks
        )
        
        self.assertIsNotNone(result['image'])


if __name__ == '__main__':
    unittest.main()


