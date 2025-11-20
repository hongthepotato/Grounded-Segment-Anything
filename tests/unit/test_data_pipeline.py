"""
Unit tests for data pipeline components.

Tests:
- Dataset inspection
- COCO format validation
- Bbox auto-generation from masks
- Dataset loading
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path

from ml_engine.data.inspection import (
    inspect_dataset,
    get_required_models,
    get_recommended_student_model
)
from ml_engine.data.validators import (
    validate_coco_format,
    compute_bbox_from_mask,
    compute_area_from_mask,
    preprocess_coco_dataset
)


class TestDatasetInspection(unittest.TestCase):
    """Test dataset inspection utilities."""
    
    def setUp(self):
        """Create sample COCO data."""
        self.coco_data_both = {
            'images': [
                {'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}
            ],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'bbox': [100, 100, 50, 50],
                    'segmentation': [[100, 100, 150, 100, 150, 150, 100, 150]],
                    'area': 2500
                }
            ],
            'categories': [
                {'id': 0, 'name': 'cat'},
                {'id': 1, 'name': 'dog'}
            ]
        }
        
        self.coco_data_boxes_only = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [100, 100, 50, 50]}
            ],
            'categories': [{'id': 0, 'name': 'cat'}]
        }
        
        self.coco_data_masks_only = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'segmentation': [[100, 100, 150, 100, 150, 150, 100, 150]]
                }
            ],
            'categories': [{'id': 0, 'name': 'cat'}]
        }
    
    def test_inspect_dataset_both(self):
        """Test inspection with both boxes and masks."""
        info = inspect_dataset(self.coco_data_both)
        
        self.assertTrue(info['has_boxes'])
        self.assertTrue(info['has_masks'])
        self.assertEqual(info['num_classes'], 2)
        self.assertEqual(info['class_mapping'], {0: 'cat', 1: 'dog'})
        self.assertEqual(info['annotation_mode'], 'DETECTION_AND_SEGMENTATION')
    
    def test_inspect_dataset_boxes_only(self):
        """Test inspection with boxes only."""
        info = inspect_dataset(self.coco_data_boxes_only)
        
        self.assertTrue(info['has_boxes'])
        self.assertFalse(info['has_masks'])
        self.assertEqual(info['annotation_mode'], 'DETECTION_ONLY')
    
    def test_inspect_dataset_masks_only(self):
        """Test inspection with masks only."""
        info = inspect_dataset(self.coco_data_masks_only)
        
        self.assertFalse(info['has_boxes'])
        self.assertTrue(info['has_masks'])
        self.assertEqual(info['annotation_mode'], 'SEGMENTATION_ONLY')
    
    def test_get_required_models(self):
        """Test model requirement detection."""
        # Both annotations
        info = inspect_dataset(self.coco_data_both)
        models = get_required_models(info)
        self.assertIn('grounding_dino', models)
        self.assertIn('sam', models)
        
        # Boxes only
        info = inspect_dataset(self.coco_data_boxes_only)
        models = get_required_models(info)
        self.assertIn('grounding_dino', models)
        self.assertNotIn('sam', models)
        
        # Masks only
        info = inspect_dataset(self.coco_data_masks_only)
        models = get_required_models(info)
        self.assertNotIn('grounding_dino', models)
        self.assertIn('sam', models)
    
    def test_get_recommended_student_model(self):
        """Test student model recommendation."""
        # Both annotations
        info = inspect_dataset(self.coco_data_both)
        student = get_recommended_student_model(info, size='s')
        self.assertEqual(student, 'yolov8s-seg')
        
        # Boxes only
        info = inspect_dataset(self.coco_data_boxes_only)
        student = get_recommended_student_model(info, size='n')
        self.assertEqual(student, 'yolov8n')
        
        # Masks only
        info = inspect_dataset(self.coco_data_masks_only)
        student = get_recommended_student_model(info, size='s')
        self.assertEqual(student, 'fastsam-s')


class TestDataValidation(unittest.TestCase):
    """Test data validation utilities."""
    
    def test_validate_coco_format_valid(self):
        """Test validation with valid COCO data."""
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]}],
            'categories': [{'id': 0, 'name': 'cat'}]
        }
        
        is_valid, errors = validate_coco_format(coco_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_coco_format_missing_keys(self):
        """Test validation with missing required keys."""
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg'}],
            # Missing 'annotations' and 'categories'
        }
        
        is_valid, errors = validate_coco_format(coco_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_compute_bbox_from_polygon(self):
        """Test bbox computation from polygon."""
        # Square polygon
        polygon = [[100, 100, 150, 100, 150, 150, 100, 150]]
        bbox = compute_bbox_from_mask(polygon, height=480, width=640)
        
        # Expected: [x_min, y_min, width, height]
        self.assertEqual(len(bbox), 4)
        self.assertAlmostEqual(bbox[0], 100.0)  # x_min
        self.assertAlmostEqual(bbox[1], 100.0)  # y_min
        self.assertAlmostEqual(bbox[2], 50.0)   # width
        self.assertAlmostEqual(bbox[3], 50.0)   # height
    
    def test_preprocess_coco_dataset(self):
        """Test auto-generation of bbox from masks."""
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'segmentation': [[100, 100, 150, 100, 150, 150, 100, 150]]
                    # No bbox or area
                }
            ],
            'categories': [{'id': 0, 'name': 'cat'}]
        }
        
        # Preprocess
        processed = preprocess_coco_dataset(coco_data, in_place=False)
        
        # Check that bbox and area were generated
        ann = processed['annotations'][0]
        self.assertIn('bbox', ann)
        self.assertIn('area', ann)
        self.assertEqual(len(ann['bbox']), 4)
        self.assertGreater(ann['area'], 0)


class TestBboxComputation(unittest.TestCase):
    """Test bounding box computation from masks."""
    
    def test_compute_bbox_from_simple_polygon(self):
        """Test with simple square polygon."""
        polygon = [[10, 10, 20, 10, 20, 20, 10, 20]]
        bbox = compute_bbox_from_mask(polygon)
        
        self.assertEqual(len(bbox), 4)
        self.assertAlmostEqual(bbox[0], 10.0)
        self.assertAlmostEqual(bbox[1], 10.0)
        self.assertAlmostEqual(bbox[2], 10.0)
        self.assertAlmostEqual(bbox[3], 10.0)
    
    def test_compute_bbox_from_multiple_polygons(self):
        """Test with multiple disconnected polygons."""
        # Two separate squares
        polygons = [
            [10, 10, 20, 10, 20, 20, 10, 20],
            [30, 30, 40, 30, 40, 40, 30, 40]
        ]
        bbox = compute_bbox_from_mask(polygons)
        
        # Should encompass both polygons
        self.assertAlmostEqual(bbox[0], 10.0)  # x_min
        self.assertAlmostEqual(bbox[1], 10.0)  # y_min
        self.assertAlmostEqual(bbox[2], 30.0)  # width (40 - 10)
        self.assertAlmostEqual(bbox[3], 30.0)  # height (40 - 10)
    
    def test_compute_area_from_polygon(self):
        """Test area computation from polygon."""
        # Square 10x10
        polygon = [[0, 0, 10, 0, 10, 10, 0, 10]]
        area = compute_area_from_mask(polygon, height=100, width=100)
        
        # Area should be approximately 100 (10x10)
        self.assertGreater(area, 90)
        self.assertLess(area, 110)


if __name__ == '__main__':
    unittest.main()


