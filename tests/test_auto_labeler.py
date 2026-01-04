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
        from ml_engine.inference import AutoLabelerConfig, OUTPUT_BOXES_ONLY
        
        config = AutoLabelerConfig()
        
        assert config.box_threshold == 0.5
        assert config.text_threshold == 0.5
        assert config.nms_threshold == 0.7
        assert config.output_mode == OUTPUT_BOXES_ONLY
        assert "groundingdino" in config.grounding_dino_config.lower()
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from ml_engine.inference import AutoLabelerConfig
        
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
        from ml_engine.inference import (
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


class TestCOCOExporter:
    """Test COCO exporter functions."""
    
    def test_mask_to_polygon_empty(self):
        """Test conversion of empty mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        # Empty mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        polygons = COCOExporter.mask_to_polygon(mask)
        
        assert polygons == []
    
    def test_mask_to_polygon_simple(self):
        """Test conversion of simple rectangular mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        # Simple square mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        polygons = COCOExporter.mask_to_polygon(mask)
        
        assert len(polygons) >= 1
        assert len(polygons[0]) >= 6  # At least 3 points (x,y pairs)
    
    def test_mask_to_polygon_complex(self):
        """Test conversion of mask with multiple regions."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        # Two separate regions
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Region 1
        mask[60:90, 60:90] = 1  # Region 2
        
        polygons = COCOExporter.mask_to_polygon(mask)
        
        # Should have 2 polygons for 2 regions
        assert len(polygons) == 2
    
    def test_mask_to_polygon_none(self):
        """Test handling of None mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        polygons = COCOExporter.mask_to_polygon(None)
        assert polygons == []
    
    def test_bbox_from_mask_empty(self):
        """Test bbox from empty mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = COCOExporter.bbox_from_mask(mask)
        
        assert bbox == [0.0, 0.0, 0.0, 0.0]
    
    def test_bbox_from_mask_simple(self):
        """Test bbox from simple rectangular mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 30:80] = 1  # y: 20-49, x: 30-79
        
        bbox = COCOExporter.bbox_from_mask(mask)
        
        # [x, y, width, height]
        assert bbox[0] == 30.0  # x_min
        assert bbox[1] == 20.0  # y_min
        assert bbox[2] == 49.0  # width (79 - 30)
        assert bbox[3] == 29.0  # height (49 - 20)
    
    def test_bbox_from_mask_none(self):
        """Test bbox from None mask."""
        from ml_engine.inference.exporters.coco import COCOExporter
        
        bbox = COCOExporter.bbox_from_mask(None)
        assert bbox == [0.0, 0.0, 0.0, 0.0]


class TestExportToCoco:
    """Test COCO export function."""
    
    def test_export_empty_detections(self):
        """Test export with no detections."""
        from ml_engine.inference import COCOExporter
        
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
        
        coco = COCOExporter.export(detections_list, ['class1', 'class2'])
        
        assert len(coco['images']) == 1
        assert len(coco['annotations']) == 0
        assert len(coco['categories']) == 2
    
    def test_export_with_detections_both_mode(self):
        """Test export with mock detections (both boxes and masks)."""
        from ml_engine.inference import COCOExporter, OUTPUT_BOTH
        
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
        
        coco = COCOExporter.export(detections_list, ['dog', 'cat'], output_mode=OUTPUT_BOTH)
        
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
        from ml_engine.inference import COCOExporter, OUTPUT_BOXES_ONLY
        
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
        
        coco = COCOExporter.export(detections_list, ['dog'], output_mode=OUTPUT_BOXES_ONLY)
        
        ann = coco['annotations'][0]
        assert 'bbox' in ann
        assert ann['bbox'] == [10.0, 20.0, 50.0, 60.0]
        assert 'segmentation' not in ann
    
    def test_export_masks_only_mode(self):
        """Test export with masks only mode (bbox auto-generated from mask)."""
        from ml_engine.inference import COCOExporter, OUTPUT_MASKS_ONLY
        
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
        
        coco = COCOExporter.export(detections_list, ['dog'], output_mode=OUTPUT_MASKS_ONLY)
        
        ann = coco['annotations'][0]
        assert 'segmentation' in ann
        assert 'bbox' in ann  # Auto-generated from mask
        # Check auto-generated bbox is approximately correct
        assert ann['bbox'][0] == 30.0  # x_min
        assert ann['bbox'][1] == 20.0  # y_min


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""
    
    def test_export_to_coco_function(self):
        """Test that export_to_coco function still works."""
        from ml_engine.inference import export_to_coco
        
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
        
        coco = export_to_coco(detections_list, ['dog'])
        
        assert len(coco['images']) == 1
        assert len(coco['annotations']) == 1
    
    def test_visualize_functions_exported(self):
        """Test that visualization functions are still exported."""
        from ml_engine.inference import visualize_detections, visualize_batch
        
        # Just check they are callable
        assert callable(visualize_detections)
        assert callable(visualize_batch)
    
    def test_label_single_image_deprecated(self):
        """Test that deprecated label_single_image still works."""
        from ml_engine.inference import AutoLabeler, AutoLabelerConfig
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        
        # Mock the internal method to avoid loading models
        labeler.label_images = Mock(return_value=[{
            'class_ids': [0],
            'scores': [0.9],
            'boxes': [[10, 20, 30, 40]],
            'image_info': {'file_name': 'test.jpg', 'width': 100, 'height': 100}
        }])
        
        result = labeler.label_single_image("test.jpg", ["dog"])
        
        assert 'class_ids' in result
        assert 'image_info' in result


class TestAutoLabeler:
    """Tests for AutoLabeler coordinator class."""
    
    def test_label_images_empty(self):
        """Test labeling with empty input."""
        from ml_engine.inference import AutoLabeler, AutoLabelerConfig
        
        config = AutoLabelerConfig(device="cpu", output_mode="boxes")
        labeler = AutoLabeler(config)
        
        results = labeler.label_images([], ["dog"])
        assert results == []
    
    def test_label_images_signature(self):
        """Test that label_images has correct signature (no batch_size)."""
        from ml_engine.inference import AutoLabeler
        import inspect
        
        sig = inspect.signature(AutoLabeler.label_images)
        params = list(sig.parameters.keys())
        
        # Should have: self, image_paths, class_prompts, progress_callback
        assert 'image_paths' in params
        assert 'class_prompts' in params
        assert 'progress_callback' in params
        assert 'batch_size' not in params  # Removed!


class TestDetectorProtocol:
    """Test detector protocol and implementation."""
    
    def test_detection_result_empty(self):
        """Test DetectionResult with no detections."""
        from ml_engine.inference.detectors.base import DetectionResult
        
        result = DetectionResult(
            boxes_xyxy=np.array([]),
            confidences=np.array([]),
            class_ids=np.array([])
        )
        
        assert len(result) == 0
        assert result.is_empty
    
    def test_detection_result_with_data(self):
        """Test DetectionResult with detections."""
        from ml_engine.inference.detectors.base import DetectionResult
        
        result = DetectionResult(
            boxes_xyxy=np.array([[10, 20, 50, 60]]),
            confidences=np.array([0.9]),
            class_ids=np.array([0])
        )
        
        assert len(result) == 1
        assert not result.is_empty
    
    def test_detector_protocol_no_batch(self):
        """Test that DetectorProtocol has no detect_batch method."""
        from ml_engine.inference.detectors.base import DetectorProtocol
        import inspect
        
        # Get all method names from protocol
        methods = [m for m in dir(DetectorProtocol) if not m.startswith('_')]
        
        assert 'detect' in methods
        assert 'detect_batch' not in methods  # Removed!


class TestProtocolsAreCheckable:
    """Test that protocol classes are runtime checkable."""
    
    def test_detector_protocol(self):
        """Test DetectorProtocol is runtime checkable."""
        from ml_engine.inference.detectors.base import DetectorProtocol
        from ml_engine.inference.detectors.grounding_dino import GroundingDINODetector
        
        # Check that the implementation is an instance of the protocol
        detector = GroundingDINODetector(device="cpu")
        assert isinstance(detector, DetectorProtocol)
    
    def test_segmenter_protocol(self):
        """Test SegmenterProtocol is runtime checkable."""
        from ml_engine.inference.segmenters.base import SegmenterProtocol
        from ml_engine.inference.segmenters.mobile_sam import MobileSAMSegmenter
        
        # Check that the implementation is an instance of the protocol
        segmenter = MobileSAMSegmenter(device="cpu")
        assert isinstance(segmenter, SegmenterProtocol)


@pytest.mark.skipif(
    not Path("data/models/pretrained/groundingdino_swint_ogc.pth").exists() or
    not Path("data/models/pretrained/mobile_sam.pt").exists(),
    reason="Model checkpoints not found. Download them first."
)
class TestAutoLabelerIntegration:
    """Integration tests that require actual model checkpoints."""
    
    def test_label_single_image(self):
        """Test labeling a single image."""
        from ml_engine.inference import AutoLabeler, AutoLabelerConfig
        
        config = AutoLabelerConfig(output_mode="both")
        labeler = AutoLabeler(config)
        
        # Use a demo image
        image_path = "assets/demo2.jpg"
        if not Path(image_path).exists():
            pytest.skip(f"Demo image not found: {image_path}")
        
        results = labeler.label_images(
            image_paths=[image_path],
            class_prompts=["dog"]
        )
        
        assert len(results) == 1
        result = results[0]
        assert 'boxes' in result
        assert 'masks' in result
        assert 'class_ids' in result
        assert 'scores' in result
        assert 'image_info' in result
    
    def test_label_images_with_export(self):
        """Test labeling multiple images and exporting to COCO."""
        from ml_engine.inference import AutoLabeler, AutoLabelerConfig, COCOExporter
        
        config = AutoLabelerConfig(output_mode="boxes")
        labeler = AutoLabeler(config)
        
        # Use demo images
        image_paths = [
            "assets/demo2.jpg",
            "assets/demo3.jpg"
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        
        if len(image_paths) < 2:
            pytest.skip("Not enough demo images found")
        
        # Label images (sequential, no batch_size parameter)
        results = labeler.label_images(
            image_paths=image_paths,
            class_prompts=["dog", "cat"],
        )
        
        # Export to COCO
        coco = COCOExporter.export(results, ["dog", "cat"])
        
        # Verify COCO format
        assert 'images' in coco
        assert 'annotations' in coco
        assert 'categories' in coco
        assert len(coco['images']) == len(image_paths)
        assert len(coco['categories']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
