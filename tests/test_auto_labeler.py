"""
Tests for the AutoLabeler service.

To run:
    pytest tests/test_auto_labeler.py -v

Note: Some tests require model checkpoints to be downloaded:
    - data/models/pretrained/groundingdino_swint_ogc.pth
    - data/models/pretrained/mobile_sam.pt (download from MobileSAM repo)
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAutoLabelerConfig:
    """Test AutoLabelerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from ml_engine.inference.auto_labeler import AutoLabelerConfig, OUTPUT_BOTH
        
        config = AutoLabelerConfig()
        
        assert config.box_threshold == 0.25
        assert config.text_threshold == 0.25
        assert config.nms_threshold == 0.5
        assert config.output_mode == OUTPUT_BOTH
        assert "groundingdino" in config.grounding_dino_config.lower()
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from ml_engine.inference.auto_labeler import AutoLabelerConfig
        
        config = AutoLabelerConfig(
            box_threshold=0.3,
            nms_threshold=0.7,
            device="cpu"
        )
        
        assert config.box_threshold == 0.3
        assert config.nms_threshold == 0.7
        assert config.device == "cpu"
    
    def test_output_modes(self):
        """Test different output mode configurations."""
        from ml_engine.inference.auto_labeler import (
            AutoLabelerConfig, 
            OUTPUT_BOXES_ONLY, 
            OUTPUT_MASKS_ONLY, 
            OUTPUT_BOTH
        )
        
        # Boxes only mode
        config_boxes = AutoLabelerConfig(output_mode=OUTPUT_BOXES_ONLY)
        assert config_boxes.output_mode == "boxes"
        
        # Masks only mode
        config_masks = AutoLabelerConfig(output_mode=OUTPUT_MASKS_ONLY)
        assert config_masks.output_mode == "masks"
        
        # Both mode
        config_both = AutoLabelerConfig(output_mode=OUTPUT_BOTH)
        assert config_both.output_mode == "both"


class TestMaskToPolygon:
    """Test mask to polygon conversion."""
    
    def test_empty_mask(self):
        """Test conversion of empty mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        # Empty mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = AutoLabeler._mask_to_polygon(mask)
        
        assert polygons == []
    
    def test_simple_mask(self):
        """Test conversion of simple rectangular mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        # Simple square mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        polygons = AutoLabeler._mask_to_polygon(mask)
        
        assert len(polygons) >= 1
        assert len(polygons[0]) >= 6  # At least 3 points (x,y pairs)
    
    def test_complex_mask(self):
        """Test conversion of mask with multiple regions."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        # Two separate regions
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Region 1
        mask[60:90, 60:90] = 1  # Region 2
        
        polygons = AutoLabeler._mask_to_polygon(mask)
        
        # Should have 2 polygons for 2 regions
        assert len(polygons) == 2
    
    def test_none_mask(self):
        """Test handling of None mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        polygons = AutoLabeler._mask_to_polygon(None)
        assert polygons == []


class TestBboxFromMask:
    """Test bounding box generation from mask."""
    
    def test_empty_mask(self):
        """Test bbox from empty mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = AutoLabeler._bbox_from_mask(mask)
        
        assert bbox == [0.0, 0.0, 0.0, 0.0]
    
    def test_simple_mask(self):
        """Test bbox from simple rectangular mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:80] = 1  # y: 20-49, x: 30-79
        
        bbox = AutoLabeler._bbox_from_mask(mask)
        
        # [x, y, width, height]
        assert bbox[0] == 30.0  # x_min
        assert bbox[1] == 20.0  # y_min
        assert bbox[2] == 49.0  # width (79 - 30)
        assert bbox[3] == 29.0  # height (49 - 20)
    
    def test_none_mask(self):
        """Test bbox from None mask."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        bbox = AutoLabeler._bbox_from_mask(None)
        assert bbox == [0.0, 0.0, 0.0, 0.0]


class TestExportToCoco:
    """Test COCO export function."""
    
    def test_export_empty_detections(self):
        """Test export with no detections."""
        from ml_engine.inference.auto_labeler import export_to_coco
        
        detections_list = [{
            'boxes': [],
            'masks': [],
            'class_ids': [],
            'scores': [],
            'image_info': {
                'file_name': 'test.jpg',
                'width': 640,
                'height': 480
            }
        }]
        
        coco = export_to_coco(detections_list, ['class1', 'class2'])
        
        assert len(coco['images']) == 1
        assert len(coco['annotations']) == 0
        assert len(coco['categories']) == 2
    
    def test_export_with_detections_both_mode(self):
        """Test export with mock detections (both boxes and masks)."""
        from ml_engine.inference.auto_labeler import export_to_coco, OUTPUT_BOTH
        
        # Create mock mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        detections_list = [{
            'boxes': [[10.0, 20.0, 50.0, 60.0]],
            'masks': [mask],
            'class_ids': [0],
            'scores': [0.95],
            'image_info': {
                'file_name': 'test.jpg',
                'width': 640,
                'height': 480
            }
        }]
        
        coco = export_to_coco(detections_list, ['dog', 'cat'], output_mode=OUTPUT_BOTH)
        
        assert len(coco['images']) == 1
        assert len(coco['annotations']) == 1
        assert len(coco['categories']) == 2
        
        # Check annotation format
        ann = coco['annotations'][0]
        assert ann['id'] == 1
        assert ann['image_id'] == 1
        assert ann['category_id'] == 0
        assert ann['bbox'] == [10.0, 20.0, 50.0, 60.0]
        assert ann['score'] == 0.95
        assert ann['iscrowd'] == 0
        assert 'segmentation' in ann
        assert 'area' in ann
    
    def test_export_boxes_only_mode(self):
        """Test export with boxes only mode."""
        from ml_engine.inference.auto_labeler import export_to_coco, OUTPUT_BOXES_ONLY
        
        detections_list = [{
            'boxes': [[10.0, 20.0, 50.0, 60.0]],
            'class_ids': [0],
            'scores': [0.95],
            'image_info': {
                'file_name': 'test.jpg',
                'width': 640,
                'height': 480
            }
        }]
        
        coco = export_to_coco(detections_list, ['dog'], output_mode=OUTPUT_BOXES_ONLY)
        
        ann = coco['annotations'][0]
        assert 'bbox' in ann
        assert ann['bbox'] == [10.0, 20.0, 50.0, 60.0]
        assert 'segmentation' not in ann
    
    def test_export_masks_only_mode(self):
        """Test export with masks only mode (bbox auto-generated from mask)."""
        from ml_engine.inference.auto_labeler import export_to_coco, OUTPUT_MASKS_ONLY
        
        # Create mock mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:80] = 1  # y: 20-49, x: 30-79
        
        detections_list = [{
            'masks': [mask],
            'class_ids': [0],
            'scores': [0.95],
            'image_info': {
                'file_name': 'test.jpg',
                'width': 640,
                'height': 480
            }
        }]
        
        coco = export_to_coco(detections_list, ['dog'], output_mode=OUTPUT_MASKS_ONLY)
        
        ann = coco['annotations'][0]
        assert 'segmentation' in ann
        assert 'bbox' in ann  # Auto-generated from mask
        # Check auto-generated bbox is approximately correct
        assert ann['bbox'][0] == 30.0  # x_min
        assert ann['bbox'][1] == 20.0  # y_min


class TestAutoLabelerMocked:
    """Tests for AutoLabeler with mocked models."""
    
    @patch('ml_engine.inference.auto_labeler.GroundingDINOModel')
    @patch('ml_engine.inference.auto_labeler.setup_mobile_sam')
    @patch('ml_engine.inference.auto_labeler.SamPredictor')
    @patch('torch.load')
    def test_model_loading(self, mock_torch_load, mock_sam_predictor, 
                           mock_setup_sam, mock_dino):
        """Test that models are loaded correctly."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        
        # Setup mocks
        mock_sam = MagicMock()
        mock_setup_sam.return_value = mock_sam
        mock_torch_load.return_value = {}
        
        config = AutoLabelerConfig(device="cpu")
        labeler = AutoLabeler(config)
        
        # Models not loaded yet (lazy loading)
        assert not labeler._models_loaded
        
        # Trigger model loading
        labeler._load_models()
        
        # Verify models were loaded
        assert labeler._models_loaded
        mock_dino.assert_called_once()
        mock_setup_sam.assert_called_once()


@pytest.mark.skipif(
    not Path("data/models/pretrained/groundingdino_swint_ogc.pth").exists() or
    not Path("data/models/pretrained/mobile_sam.pt").exists(),
    reason="Model checkpoints not found. Download them first."
)
class TestAutoLabelerIntegration:
    """Integration tests that require actual model checkpoints."""
    
    def test_label_single_image(self):
        """Test labeling a single image."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        labeler = AutoLabeler()
        
        # Use a demo image
        image_path = "assets/demo2.jpg"
        if not Path(image_path).exists():
            pytest.skip(f"Demo image not found: {image_path}")
        
        result = labeler.label_single_image(
            image_path=image_path,
            class_prompts=["dog"]
        )
        
        assert 'boxes' in result
        assert 'masks' in result
        assert 'class_ids' in result
        assert 'scores' in result
        assert 'image_info' in result
    
    def test_label_images_batch(self):
        """Test labeling multiple images."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        import tempfile
        
        labeler = AutoLabeler()
        
        # Use demo images
        image_paths = [
            "assets/demo2.jpg",
            "assets/demo3.jpg"
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        
        if len(image_paths) < 2:
            pytest.skip("Not enough demo images found")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            coco = labeler.label_images(
                image_paths=image_paths,
                class_prompts=["dog", "cat"],
                output_path=output_path
            )
            
            # Verify COCO format
            assert 'images' in coco
            assert 'annotations' in coco
            assert 'categories' in coco
            assert len(coco['images']) == len(image_paths)
            
            # Verify file was saved
            assert Path(output_path).exists()
            with open(output_path) as f:
                saved_coco = json.load(f)
            assert saved_coco == coco
            
        finally:
            if Path(output_path).exists():
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

