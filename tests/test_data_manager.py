"""
Test DataManager refactoring.

This test verifies that:
1. DataManager loads data only once
2. No double inspection
3. Canonical form data loading works correctly
4. Everything is cached properly
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json


def test_data_manager_loads_once():
    """Verify DataManager loads JSON only once."""
    
    # Mock COCO data
    mock_coco_data = {
        'images': [
            {'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}
        ],
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 0,
                'bbox': [10, 10, 100, 100],
                'segmentation': [[10, 10, 110, 10, 110, 110, 10, 110]],
                'area': 10000
            }
        ],
        'categories': [
            {'id': 0, 'name': 'class1'}
        ]
    }
    
    with patch('ml_engine.data.manager.load_json', return_value=mock_coco_data) as mock_load:
        with patch('ml_engine.data.manager.validate_coco_format', return_value=(True, [])):
            with patch('ml_engine.data.manager.detect_annotation_mode', return_value='combined'):
                with patch('ml_engine.data.manager.normalize_coco_annotations', return_value=mock_coco_data):
                    with patch('ml_engine.data.manager.inspect_dataset') as mock_inspect:
                        with patch('ml_engine.data.manager.check_data_quality', return_value={'warnings': []}):
                            with patch.object(Path, 'exists', return_value=True):
                                from ml_engine.data.manager import DataManager
                                
                                mock_inspect.return_value = {
                                    'has_boxes': True,
                                    'has_masks': True,
                                    'num_classes': 1,
                                    'class_mapping': {0: 'class1'},
                                    'category_id_to_index': {0: 0},
                                    'index_to_category_id': {0: 0},
                                    'num_images': 1,
                                    'num_annotations': 1,
                                    'annotation_mode': 'DETECTION_AND_SEGMENTATION',
                                    'class_counts': {0: 1}
                                }
                                
                                # Create manager
                                manager = DataManager(
                                    data_path='fake/path.json',
                                    image_paths=['img1.jpg'],
                                    split_config=None
                                )
                                
                                # Verify JSON loaded exactly once
                                assert mock_load.call_count == 1
                                
                                # Access dataset info multiple times
                                info1 = manager.get_dataset_info()
                                info2 = manager.get_dataset_info()
                                info3 = manager.get_dataset_info()
                                
                                # JSON should still only be loaded once (cached)
                                assert mock_load.call_count == 1
                                
                                # All should return same cached object
                                assert info1 is info2
                                assert info2 is info3
                                
                                print("✅ DataManager loads JSON only once and caches results")


def test_no_double_inspection():
    """Verify inspection happens only once."""
    
    mock_coco_data = {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
        'annotations': [{
            'id': 1, 'image_id': 1, 'category_id': 0,
            'bbox': [10, 10, 100, 100]
        }],
        'categories': [{'id': 0, 'name': 'class1'}]
    }
    
    with patch('ml_engine.data.manager.load_json', return_value=mock_coco_data):
        with patch('ml_engine.data.manager.validate_coco_format', return_value=(True, [])):
            with patch('ml_engine.data.manager.detect_annotation_mode', return_value='detection'):
                with patch('ml_engine.data.manager.normalize_coco_annotations', return_value=mock_coco_data):
                    with patch('ml_engine.data.manager.inspect_dataset') as mock_inspect:
                        with patch('ml_engine.data.manager.check_data_quality', return_value={'warnings': []}):
                            with patch.object(Path, 'exists', return_value=True):
                                from ml_engine.data.manager import DataManager
                                
                                mock_inspect.return_value = {
                                    'has_boxes': True,
                                    'has_masks': False,
                                    'num_classes': 1,
                                    'class_mapping': {0: 'class1'},
                                    'category_id_to_index': {0: 0},
                                    'index_to_category_id': {0: 0},
                                    'num_images': 1,
                                    'num_annotations': 1,
                                    'annotation_mode': 'DETECTION_ONLY',
                                    'class_counts': {0: 1}
                                }
                                
                                # Create manager
                                manager = DataManager(
                                    data_path='fake/path.json',
                                    image_paths=['img1.jpg'],
                                    split_config=None
                                )
                                
                                # Verify inspection called exactly once
                                assert mock_inspect.call_count == 1
                                
                                # Access results multiple times
                                manager.get_dataset_info()
                                manager.get_dataset_info()
                                manager.get_required_models()
                                manager.get_required_models()
                                
                                # Inspection should still only be called once
                                assert mock_inspect.call_count == 1
                                
                                print("✅ Dataset inspected only once, results cached")


def test_canonical_form_data_loading():
    """Verify canonical form: detect mode -> normalize -> inspect."""
    
    # Start with segmentation-only data (no bbox)
    mock_coco_data = {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
        'annotations': [{
            'id': 1, 'image_id': 1, 'category_id': 0,
            'segmentation': [[10, 10, 110, 10, 110, 110, 10, 110]]
            # No bbox!
        }],
        'categories': [{'id': 0, 'name': 'class1'}]
    }
    
    # After normalization, bbox will be added
    normalized_data = {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
        'annotations': [{
            'id': 1, 'image_id': 1, 'category_id': 0,
            'segmentation': [[10, 10, 110, 10, 110, 110, 10, 110]],
            'bbox': [10, 10, 100, 100]  # Added by normalization
        }],
        'categories': [{'id': 0, 'name': 'class1'}]
    }
    
    call_order = []
    
    def track_detect_mode(data):
        call_order.append('detect_annotation_mode')
        return 'segmentation'  # Original data has only masks
    
    def track_normalize(data, in_place=True):
        call_order.append('normalize_coco_annotations')
        return normalized_data
    
    def track_inspect(data):
        call_order.append('inspect_dataset')
        return {
            'has_boxes': True,  # After normalization, has_boxes=True
            'has_masks': True,
            'num_classes': 1,
            'class_mapping': {0: 'class1'},
            'category_id_to_index': {0: 0},
            'index_to_category_id': {0: 0},
            'num_images': 1,
            'num_annotations': 1,
            'annotation_mode': 'DETECTION_AND_SEGMENTATION',
            'class_counts': {0: 1}
        }
    
    with patch('ml_engine.data.manager.load_json', return_value=mock_coco_data):
        with patch('ml_engine.data.manager.validate_coco_format', return_value=(True, [])):
            with patch('ml_engine.data.manager.detect_annotation_mode', side_effect=track_detect_mode):
                with patch('ml_engine.data.manager.normalize_coco_annotations', side_effect=track_normalize):
                    with patch('ml_engine.data.manager.inspect_dataset', side_effect=track_inspect):
                        with patch('ml_engine.data.manager.check_data_quality', return_value={'warnings': []}):
                            with patch.object(Path, 'exists', return_value=True):
                                from ml_engine.data.manager import DataManager
                                
                                manager = DataManager(
                                    data_path='fake/path.json',
                                    image_paths=['img1.jpg'],
                                    split_config=None
                                )
                                
                                # Verify call order: detect -> normalize -> inspect
                                assert call_order == [
                                    'detect_annotation_mode',
                                    'normalize_coco_annotations',
                                    'inspect_dataset'
                                ], f"Wrong order: {call_order}"
                                
                                # Verify original mode captured
                                assert manager.get_original_annotation_mode() == 'segmentation'
                                
                                # Verify required models based on ORIGINAL mode
                                assert manager.get_required_models() == ['sam']
                                
                                # Verify dataset_info reflects NORMALIZED state
                                info = manager.get_dataset_info()
                                assert info['has_boxes'] == True  # After normalization
                                assert info['has_masks'] == True
                                
                                print("✅ Canonical form data loading works correctly")


def test_data_flow_no_redundancy():
    """Integration test: Verify entire data flow has no redundancy."""
    
    mock_coco_data = {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
        'annotations': [{
            'id': 1, 'image_id': 1, 'category_id': 0,
            'bbox': [10, 10, 100, 100],
            'segmentation': [[10, 10, 110, 10, 110, 110, 10, 110]]
        }],
        'categories': [{'id': 0, 'name': 'class1'}]
    }
    
    with patch('ml_engine.data.manager.load_json', return_value=mock_coco_data) as mock_load:
        with patch('ml_engine.data.manager.validate_coco_format', return_value=(True, [])):
            with patch('ml_engine.data.manager.detect_annotation_mode', return_value='combined') as mock_detect:
                with patch('ml_engine.data.manager.normalize_coco_annotations', return_value=mock_coco_data) as mock_normalize:
                    with patch('ml_engine.data.manager.inspect_dataset') as mock_inspect:
                        with patch('ml_engine.data.manager.check_data_quality', return_value={'warnings': []}):
                            with patch.object(Path, 'exists', return_value=True):
                                from ml_engine.data.manager import DataManager
                                
                                mock_inspect.return_value = {
                                    'has_boxes': True,
                                    'has_masks': True,
                                    'num_classes': 1,
                                    'class_mapping': {0: 'class1'},
                                    'category_id_to_index': {0: 0},
                                    'index_to_category_id': {0: 0},
                                    'num_images': 1,
                                    'num_annotations': 1,
                                    'annotation_mode': 'DETECTION_AND_SEGMENTATION',
                                    'class_counts': {0: 1}
                                }
                                
                                # Create manager
                                manager = DataManager(
                                    data_path='fake/path.json',
                                    image_paths=['img1.jpg'],
                                    split_config=None
                                )
                                
                                # Access data multiple ways
                                info1 = manager.get_dataset_info()
                                info2 = manager.get_dataset_info()
                                models1 = manager.get_required_models()
                                models2 = manager.get_required_models()
                                mode1 = manager.get_original_annotation_mode()
                                mode2 = manager.get_original_annotation_mode()
                                split1 = manager.get_split('all')
                                split2 = manager.get_split('all')
                                
                                # Verify operations happened exactly once
                                assert mock_load.call_count == 1, "JSON should be loaded only once"
                                assert mock_detect.call_count == 1, "Mode detection should happen only once"
                                assert mock_normalize.call_count == 1, "Normalization should happen only once"
                                assert mock_inspect.call_count == 1, "Inspection should happen only once"
                                
                                # Verify all return same cached objects
                                assert info1 is info2
                                assert models1 == models2
                                assert mode1 == mode2
                                assert split1 is split2
                                
                                print("✅ Complete data flow has no redundant operations")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing DataManager Refactoring")
    print("=" * 60 + "\n")
    
    test_data_manager_loads_once()
    test_no_double_inspection()
    test_canonical_form_data_loading()
    test_data_flow_no_redundancy()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Refactoring is correct.")
    print("=" * 60)
