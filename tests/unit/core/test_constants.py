"""
Unit tests for core.constants module.

Tests constant values and utility functions defined in constants.
These are mostly sanity checks to ensure constants are correctly defined.
"""

from pathlib import Path
import pytest
from core import constants


@pytest.mark.unit
class TestDirectoryPaths:
    """Test directory path constants."""

    def test_project_root_is_path(self):
        """Test that PROJECT_ROOT is a Path object."""
        assert isinstance(constants.PROJECT_ROOT, Path)

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to existing directory."""
        assert constants.PROJECT_ROOT.exists()

    def test_data_dir_structure(self):
        """Test data directory constant structure."""
        assert constants.DATA_DIR == constants.PROJECT_ROOT / 'data'
        assert constants.MODELS_DIR == constants.DATA_DIR / 'models'

    def test_config_dir_structure(self):
        """Test config directory constant structure."""
        assert constants.CONFIGS_DIR == constants.PROJECT_ROOT / 'configs'
        assert constants.DEFAULT_CONFIGS_DIR == constants.CONFIGS_DIR / 'defaults'


@pytest.mark.unit
class TestImagePathTransformation:
    """Test image path transformation function."""

    def test_transform_upload_path(self):
        """Test transforming upload path."""
        input_path = 'upload/2025/12/17/image.png'
        result = constants.transform_image_path(input_path)
        
        assert result == '/srv/shared/images/upload/2025/12/17/image.png'
        assert result.startswith(constants.IMAGE_PATH_BASE)

    def test_transform_non_upload_path(self):
        """Test that non-upload paths are returned as-is."""
        input_path = '/absolute/path/to/image.png'
        result = constants.transform_image_path(input_path)
        
        assert result == input_path
    
    def test_transform_relative_non_upload_path(self):
        """Test that relative non-upload paths are returned as-is."""
        input_path = 'images/test.png'
        result = constants.transform_image_path(input_path)
        
        assert result == input_path


@pytest.mark.unit
class TestModelNames:
    """Test model name constants."""
    
    def test_teacher_model_names(self):
        """Test teacher model name constants are defined."""
        assert constants.GROUNDING_DINO == 'grounding_dino'
        assert constants.SAM == 'sam'
        assert constants.POSE_MODEL == 'pose_model'
    
    def test_yolov8_detection_models(self):
        """Test YOLOv8 detection model names."""
        expected_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
        
        actual_models = [
            constants.YOLOV8_N,
            constants.YOLOV8_S,
            constants.YOLOV8_M,
            constants.YOLOV8_L,
            constants.YOLOV8_X
        ]
        
        assert actual_models == expected_models
    
    def test_yolov8_segmentation_models(self):
        """Test YOLOv8 segmentation model names."""
        assert constants.YOLOV8_N_SEG == 'yolov8n-seg'
        assert constants.YOLOV8_X_SEG == 'yolov8x-seg'
    
    def test_alternative_sam_models(self):
        """Test alternative SAM model names."""
        assert constants.FASTSAM_S == 'fastsam-s'
        assert constants.FASTSAM_X == 'fastsam-x'
        assert constants.MOBILESAM == 'mobilesam'
    
    def test_teacher_models_list(self):
        """Test TEACHER_MODELS list contains all teacher models."""
        assert constants.GROUNDING_DINO in constants.TEACHER_MODELS
        assert constants.SAM in constants.TEACHER_MODELS
        assert constants.POSE_MODEL in constants.TEACHER_MODELS
    
    def test_student_model_lists(self):
        """Test student model lists are properly defined."""
        # Detection models
        assert constants.YOLOV8_N in constants.STUDENT_DETECTION_MODELS
        assert constants.YOLOV8_X in constants.STUDENT_DETECTION_MODELS
        
        # Segmentation models
        assert constants.YOLOV8_N_SEG in constants.STUDENT_SEGMENTATION_MODELS
        assert constants.FASTSAM_S in constants.STUDENT_SEGMENTATION_MODELS


@pytest.mark.unit
class TestAnnotationModes:
    """Test annotation mode constants."""
    
    def test_annotation_mode_values(self):
        """Test annotation mode constant values."""
        assert constants.ANNOTATION_MODE_DETECTION == 'DETECTION_ONLY'
        assert constants.ANNOTATION_MODE_SEGMENTATION == 'SEGMENTATION_ONLY'
        assert constants.ANNOTATION_MODE_COMBINED == 'DETECTION_AND_SEGMENTATION'
    
    def test_annotation_modes_list(self):
        """Test ANNOTATION_MODES list contains all modes."""
        assert constants.ANNOTATION_MODE_DETECTION in constants.ANNOTATION_MODES
        assert constants.ANNOTATION_MODE_SEGMENTATION in constants.ANNOTATION_MODES
        assert constants.ANNOTATION_MODE_COMBINED in constants.ANNOTATION_MODES
    
    def test_mode_constants(self):
        """Test simple mode constants for model selection."""
        assert constants.MODE_DETECTION == 'detection'
        assert constants.MODE_SEGMENTATION == 'segmentation'
        assert constants.MODE_COMBINED == 'combined'


@pytest.mark.unit
class TestModelInputSizes:
    """Test model input size configurations."""
    
    def test_model_input_sizes_defined(self):
        """Test that input sizes are defined for key models."""
        assert constants.GROUNDING_DINO in constants.MODEL_INPUT_SIZES
        assert constants.SAM in constants.MODEL_INPUT_SIZES
        assert 'yolov8' in constants.MODEL_INPUT_SIZES
    
    def test_sam_input_size(self):
        """Test SAM input size is 1024x1024."""
        sam_size = constants.MODEL_INPUT_SIZES[constants.SAM]
        assert sam_size['height'] == 1024
        assert sam_size['width'] == 1024
    
    def test_normalization_params(self):
        """Test normalization parameters are defined."""
        assert constants.GROUNDING_DINO in constants.MODEL_NORMALIZATION
        assert constants.SAM in constants.MODEL_NORMALIZATION
        
        # Each should have mean, std, pixel_range
        for model in [constants.GROUNDING_DINO, constants.SAM]:
            config = constants.MODEL_NORMALIZATION[model]
            assert 'mean' in config
            assert 'std' in config
            assert 'pixel_range' in config
            assert len(config['mean']) == 3  # RGB
            assert len(config['std']) == 3


@pytest.mark.unit
class TestMetricsConstants:
    """Test evaluation metrics constants."""
    
    def test_detection_metrics(self):
        """Test detection metrics list."""
        assert isinstance(constants.DETECTION_METRICS, list)
        assert 'mAP50' in constants.DETECTION_METRICS
        assert 'precision' in constants.DETECTION_METRICS
        assert 'recall' in constants.DETECTION_METRICS
    
    def test_segmentation_metrics(self):
        """Test segmentation metrics list."""
        assert isinstance(constants.SEGMENTATION_METRICS, list)
        assert 'mask_IoU' in constants.SEGMENTATION_METRICS


@pytest.mark.unit
class TestExportConfig:
    """Test export format constants."""
    
    def test_supported_export_formats(self):
        """Test export formats list."""
        assert 'onnx' in constants.SUPPORTED_EXPORT_FORMATS
        assert 'tensorrt' in constants.SUPPORTED_EXPORT_FORMATS
    
    def test_quantization_modes(self):
        """Test quantization modes."""
        assert 'int8' in constants.QUANTIZATION_MODES
        assert 'fp16' in constants.QUANTIZATION_MODES
        assert 'fp32' in constants.QUANTIZATION_MODES


@pytest.mark.unit
class TestMiscConstants:
    """Test miscellaneous constants."""
    
    def test_platform_version(self):
        """Test platform version is defined."""
        assert isinstance(constants.PLATFORM_VERSION, str)
        assert len(constants.PLATFORM_VERSION) > 0
    
    def test_platform_name(self):
        """Test platform name is defined."""
        assert isinstance(constants.PLATFORM_NAME, str)
        assert 'Grounded' in constants.PLATFORM_NAME or 'SAM' in constants.PLATFORM_NAME
    
    def test_log_format(self):
        """Test log format string is defined."""
        assert isinstance(constants.LOG_FORMAT, str)
        assert '%(asctime)s' in constants.LOG_FORMAT
        assert '%(levelname)s' in constants.LOG_FORMAT
