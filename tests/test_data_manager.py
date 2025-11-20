"""
Test DataManager refactoring.

This test verifies that:
1. DataManager loads data only once
2. No double inspection
3. PyTorch datasets receive pre-loaded data
4. Everything is cached properly
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
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
        from ml_engine.data.manager import DataManager
        
        # Create manager
        manager = DataManager(
            data_path='fake/path.json',
            image_dir='fake/images/',
            split_config=None,
            validate=False  # Skip validation for test
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
        with patch('ml_engine.data.manager.inspect_dataset') as mock_inspect:
            from ml_engine.data.manager import DataManager
            
            mock_inspect.return_value = {
                'has_boxes': True,
                'has_masks': False,
                'num_classes': 1,
                'class_mapping': {0: 'class1'},
                'num_images': 1,
                'num_annotations': 1,
                'annotation_mode': 'DETECTION_ONLY',
                'class_counts': {0: 1}
            }
            
            # Create manager
            manager = DataManager(
                data_path='fake/path.json',
                image_dir='fake/images/',
                split_config=None,
                validate=False,
                auto_preprocess=False
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


def test_pytorch_dataset_receives_preloaded_data():
    """Verify PyTorch datasets receive data, don't load files."""
    
    mock_coco_data = {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
        'annotations': [{
            'id': 1, 'image_id': 1, 'category_id': 0,
            'bbox': [10, 10, 100, 100],
            'segmentation': [[10, 10, 110, 10, 110, 110, 10, 110]]
        }],
        'categories': [{'id': 0, 'name': 'class1'}]
    }
    
    with patch('ml_engine.data.manager.load_json', return_value=mock_coco_data):
        from ml_engine.data.manager import DataManager
        
        # Create manager
        manager = DataManager(
            data_path='fake/path.json',
            image_dir='fake/images/',
            split_config=None,
            validate=False,
            auto_preprocess=False
        )
        
        # Mock TeacherDataset to verify it receives data
        with patch('ml_engine.data.manager.TeacherDataset') as mock_dataset_class:
            mock_dataset_instance = Mock()
            mock_dataset_class.return_value = mock_dataset_instance
            
            # Create PyTorch dataset through manager
            dataset = manager.create_pytorch_dataset(
                split='all',
                preprocessor=None,
                augmentation_pipeline=None
            )
            
            # Verify TeacherDataset was called with coco_data, not json_path
            call_kwargs = mock_dataset_class.call_args[1]
            assert 'coco_data' in call_kwargs
            assert 'json_path' not in call_kwargs
            
            # Verify data is the actual dict, not a file path
            assert isinstance(call_kwargs['coco_data'], dict)
            assert 'images' in call_kwargs['coco_data']
            
            print("✅ PyTorch dataset receives pre-loaded data, doesn't load files")


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
        with patch('ml_engine.data.manager.inspect_dataset') as mock_inspect:
            with patch('ml_engine.data.manager.validate_coco_format', return_value=(True, [])):
                with patch('ml_engine.data.manager.preprocess_coco_dataset', return_value=mock_coco_data):
                    with patch('ml_engine.data.manager.check_data_quality', return_value={'warnings': []}):
                        from ml_engine.data.manager import DataManager
                        
                        mock_inspect.return_value = {
                            'has_boxes': True,
                            'has_masks': True,
                            'num_classes': 1,
                            'class_mapping': {0: 'class1'},
                            'num_images': 1,
                            'num_annotations': 1,
                            'annotation_mode': 'DETECTION_AND_SEGMENTATION',
                            'class_counts': {0: 1}
                        }
                        
                        # Create manager
                        manager = DataManager(
                            data_path='fake/path.json',
                            image_dir='fake/images/',
                            split_config=None
                        )
                        
                        # Access data multiple ways
                        info1 = manager.get_dataset_info()
                        info2 = manager.get_dataset_info()
                        models1 = manager.get_required_models()
                        models2 = manager.get_required_models()
                        split1 = manager.get_split('all')
                        split2 = manager.get_split('all')
                        
                        # Verify operations happened exactly once
                        assert mock_load.call_count == 1, "JSON should be loaded only once"
                        assert mock_inspect.call_count == 1, "Inspection should happen only once"
                        
                        # Verify all return same cached objects
                        assert info1 is info2
                        assert models1 == models2
                        assert split1 is split2
                        
                        print("✅ Complete data flow has no redundant operations")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing DataManager Refactoring")
    print("=" * 60 + "\n")
    
    test_data_manager_loads_once()
    test_no_double_inspection()
    test_pytorch_dataset_receives_preloaded_data()
    test_data_flow_no_redundancy()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Refactoring is correct.")
    print("=" * 60)


