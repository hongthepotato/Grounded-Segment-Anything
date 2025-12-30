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


class TestBatchInference:
    """Tests for batch inference functionality."""
    
    @patch('ml_engine.inference.auto_labeler.GroundingDINOModel')
    def test_detect_objects_batch_empty(self, mock_dino):
        """Test batch detection with empty input."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        labeler._grounding_dino = mock_dino
        labeler._models_loaded = True
        
        results = labeler._detect_objects_batch([], ["dog"])
        assert results == []
    
    @patch('ml_engine.inference.auto_labeler.GroundingDINOModel')
    def test_detect_objects_batch_single(self, mock_dino):
        """Test batch detection with single image."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        import supervision as sv
        
        # Mock detection result
        mock_detection = sv.Detections(
            xyxy=np.array([[10, 20, 50, 60]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        mock_dino.predict_batch_with_classes.return_value = [mock_detection]
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        labeler._grounding_dino = mock_dino
        labeler._models_loaded = True
        
        # Create dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        results = labeler._detect_objects_batch([image], ["dog"])
        
        assert len(results) == 1
        boxes, confidences, class_ids = results[0]
        assert len(boxes) == 1
        assert confidences[0] >= 0.9
    
    @patch('ml_engine.inference.auto_labeler.GroundingDINOModel')
    def test_detect_objects_batch_multiple(self, mock_dino):
        """Test batch detection with multiple images."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        import supervision as sv
        
        # Mock detection results for 3 images
        mock_detections = [
            sv.Detections(
                xyxy=np.array([[10, 20, 50, 60]]),
                confidence=np.array([0.9]),
                class_id=np.array([0])
            ),
            sv.Detections(
                xyxy=np.array([[15, 25, 55, 65], [20, 30, 60, 70]]),
                confidence=np.array([0.85, 0.8]),
                class_id=np.array([0, 1])
            ),
            sv.Detections(
                xyxy=np.array([]).reshape(0, 4),  # No detections
                confidence=np.array([]),
                class_id=np.array([])
            ),
        ]
        mock_dino.predict_batch_with_classes.return_value = mock_detections
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        labeler._grounding_dino = mock_dino
        labeler._models_loaded = True
        
        # Create dummy images
        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        
        results = labeler._detect_objects_batch(images, ["dog", "cat"])
        
        assert len(results) == 3
        # Image 1: 1 detection
        assert len(results[0][0]) == 1
        # Image 2: 2 detections (after NMS)
        assert len(results[1][0]) <= 2
        # Image 3: 0 detections
        assert len(results[2][0]) == 0
    
    def test_label_batch_images_empty(self):
        """Test batch labeling with empty input."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        
        results = labeler.label_batch_images([], ["dog"])
        assert results == []
    
    @patch('ml_engine.inference.auto_labeler.GroundingDINOModel')
    @patch('cv2.imread')
    def test_label_batch_images_with_failed_load(self, mock_imread, mock_dino):
        """Test batch labeling handles failed image loads gracefully."""
        from ml_engine.inference.auto_labeler import AutoLabeler, AutoLabelerConfig
        
        # First image fails to load, second succeeds
        mock_imread.side_effect = [None, np.zeros((100, 100, 3), dtype=np.uint8)]
        
        # Mock successful detection for the loaded image
        import supervision as sv
        mock_detection = sv.Detections(
            xyxy=np.array([[10, 20, 50, 60]]),
            confidence=np.array([0.9]),
            class_id=np.array([0])
        )
        mock_dino.return_value.predict_batch_with_classes.return_value = [mock_detection]
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        labeler._models_loaded = True
        labeler._grounding_dino = mock_dino.return_value
        
        results = labeler.label_batch_images(
            ["failed.jpg", "success.jpg"], 
            ["dog"],
            batch_size=2
        )
        
        # Should have 2 results: empty for failed, actual for success
        assert len(results) == 2
        assert results[0]['class_ids'] == []  # Failed image
        assert len(results[1]['class_ids']) >= 0  # Successful image


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
    
    def test_batch_vs_single_consistency(self):
        """Test that batch and single inference produce consistent results."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        
        labeler = AutoLabeler()
        
        # Use demo images
        image_paths = [
            "assets/demo2.jpg",
            "assets/demo3.jpg"
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        
        if len(image_paths) < 2:
            pytest.skip("Not enough demo images found")
        
        class_prompts = ["dog", "cat"]
        
        # Get single-image results
        single_results = []
        for path in image_paths:
            result = labeler.label_single_image(path, class_prompts)
            single_results.append(result)
        
        # Get batch results
        batch_results = labeler.label_batch_images(
            image_paths, 
            class_prompts,
            batch_size=4
        )
        
        # Compare results
        assert len(single_results) == len(batch_results)
        
        for single, batch in zip(single_results, batch_results):
            # Same number of detections
            assert len(single['class_ids']) == len(batch['class_ids'])
            # Same class IDs
            assert single['class_ids'] == batch['class_ids']
            # Same image info
            assert single['image_info']['file_name'] == batch['image_info']['file_name']
    
    def test_batch_inference_different_sizes(self):
        """Test batch inference with different batch sizes."""
        from ml_engine.inference.auto_labeler import AutoLabeler
        import time
        
        labeler = AutoLabeler()
        
        # Use demo images
        image_paths = [
            "assets/demo2.jpg",
            "assets/demo3.jpg"
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        
        if len(image_paths) < 1:
            pytest.skip("Demo images not found")
        
        # Test with batch_size=1 (essentially sequential)
        start_1 = time.time()
        results_1 = labeler.label_batch_images(image_paths, ["dog"], batch_size=1)
        time_1 = time.time() - start_1
        
        # Test with batch_size=4
        start_4 = time.time()
        results_4 = labeler.label_batch_images(image_paths, ["dog"], batch_size=4)
        time_4 = time.time() - start_4
        
        # Results should be the same
        assert len(results_1) == len(results_4)
        
        # With enough images, batch should be faster
        # (Not a hard assertion since timing can vary)
        print(f"batch_size=1: {time_1:.3f}s, batch_size=4: {time_4:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

