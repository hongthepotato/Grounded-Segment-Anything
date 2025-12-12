"""
Constants and default values for the platform.
"""

from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CONFIGS_DIR = PROJECT_ROOT / 'configs'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / 'raw'
MODELS_DIR = DATA_DIR / 'models'
PRETRAINED_MODELS_DIR = MODELS_DIR / 'pretrained'

# Config subdirectories
DEFAULT_CONFIGS_DIR = CONFIGS_DIR / 'defaults'
EXPERIMENT_CONFIGS_DIR = CONFIGS_DIR / 'experiments'

# ============================================================================
# Model Names
# ============================================================================

# Teacher models
GROUNDING_DINO = 'grounding_dino'
SAM = 'sam'
POSE_MODEL = 'pose_model'

# Student models
YOLOV8_N = 'yolov8n'
YOLOV8_S = 'yolov8s'
YOLOV8_M = 'yolov8m'
YOLOV8_L = 'yolov8l'
YOLOV8_X = 'yolov8x'

YOLOV8_N_SEG = 'yolov8n-seg'
YOLOV8_S_SEG = 'yolov8s-seg'
YOLOV8_M_SEG = 'yolov8m-seg'
YOLOV8_L_SEG = 'yolov8l-seg'
YOLOV8_X_SEG = 'yolov8x-seg'

FASTSAM_S = 'fastsam-s'
FASTSAM_X = 'fastsam-x'
MOBILESAM = 'mobilesam'

# ============================================================================
# Pretrained Model URLs and Paths
# ============================================================================

PRETRAINED_MODEL_URLS = {
    'groundingdino_swint_ogc': {
        'url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
        'filename': 'groundingdino_swint_ogc.pth',
        'size_mb': 2900
    },
    'sam_vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth',
        'size_mb': 2400
    },
    'sam_vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth',
        'size_mb': 1200
    },
    'sam_vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth',
        'size_mb': 375
    }
}

# ============================================================================
# Default Training Hyperparameters
# ============================================================================

# Teacher training (Grounding DINO with LoRA)
DEFAULT_DINO_LORA_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 8,
    'epochs': 50,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'warmup_steps': 500,
    'gradient_accumulation': 2,
    'mixed_precision': 'fp16',
    'lora': {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        # Target modules that actually exist in Grounding DINO:
        # - MSDeformAttn: value_proj, output_proj
        # - MultiheadAttention: out_proj
        # - Vision-Language fusion: v_proj, l_proj, out_l_proj, values_v_proj, values_l_proj
        'target_modules': [
            'value_proj',      # MSDeformAttn value projection
            'output_proj',     # MSDeformAttn output projection
            'out_proj',        # MultiheadAttention output projection
            'v_proj',          # Vision projection in fusion
            'l_proj',          # Language projection in fusion
            'out_l_proj',      # Language output projection
            'values_v_proj',   # Vision values projection
            'values_l_proj'    # Language values projection
        ]
    }
}

# Teacher training (SAM with LoRA)
DEFAULT_SAM_LORA_CONFIG = {
    'learning_rate': 5e-4,
    'batch_size': 16,
    'epochs': 100,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'mixed_precision': 'fp16',
    'lora': {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'target_modules': [
            'mask_decoder.transformer.layers.*.self_attn.q_proj',
            'mask_decoder.transformer.layers.*.self_attn.k_proj',
            'mask_decoder.transformer.layers.*.self_attn.v_proj',
            'mask_decoder.output_upscaling.*.weight'
        ]
    }
}

# Student training (distillation)
DEFAULT_DISTILLATION_CONFIG = {
    'epochs': 300,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'optimizer': 'SGD',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'grad_clip': 10.0,
    'loss_weights': {
        'detection': 0.3,
        'segmentation': 0.3,
        'logit': 0.2,
        'feature': 0.2
    },
    'temperature': 4.0
}

# ============================================================================
# Data Augmentation
# ============================================================================

# Available object characteristics
OBJECT_CHARACTERISTICS = [
    'changes_shape',
    'changes_size',
    'reflective_surface',
    'low_contrast',
    'moves_or_vibrates',
    'semi_transparent',
    'similar_to_background',
    'multiple_objects',
    'partially_hidden'
]

# Environment conditions
ENVIRONMENT_CONDITIONS = {
    'lighting': ['stable', 'variable', 'poor'],
    'camera': ['fixed', 'moving', 'shaky'],
    'background': ['clean', 'busy', 'changing'],
    'distance': ['fixed', 'variable', 'close']
}

# Augmentation intensity levels
AUGMENTATION_INTENSITIES = ['low', 'medium', 'high']

# ============================================================================
# Model Input Sizes
# ============================================================================

# Preprocessing input sizes
MODEL_INPUT_SIZES = {
    'grounding_dino': {
        'min_size': 800,
        'max_size': 1333
    },
    'sam': {
        'height': 1024,
        'width': 1024
    },
    'yolov8': 640,
    'fastsam': 1024
}

# Normalization parameters
MODEL_NORMALIZATION = {
    'grounding_dino': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'pixel_range': [0, 1]
    },
    'sam': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'pixel_range': [0, 255]
    },
    'yolov8': {
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'pixel_range': [0, 1]
    }
}

# ============================================================================
# Training Dynamics
# ============================================================================

# Gradient clipping
DEFAULT_GRAD_CLIP_NORMS = {
    'lora': 0.1,          # For LoRA training (small gradients)
    'full_ft': 1.0,       # For full fine-tuning
    'distillation': 10.0  # For student distillation
}

# Mixed precision
AMP_INIT_SCALE = 65536  # 2^16

# ============================================================================
# Checkpointing
# ============================================================================

DEFAULT_CHECKPOINT_CONFIG = {
    'save_interval': 5,
    'save_last': True,
    'save_best': True,
    'max_keep_checkpoints': 5,
    'monitor_metric': 'mAP50',
    'mode': 'max',
    'early_stopping': {
        'enabled': True,
        'patience': 15,
        'min_delta': 0.001
    }
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

# Detection metrics
DETECTION_METRICS = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']

# Segmentation metrics
SEGMENTATION_METRICS = ['mask_IoU', 'mask_precision', 'mask_recall']

# ============================================================================
# Export Formats
# ============================================================================

SUPPORTED_EXPORT_FORMATS = ['onnx', 'tensorrt', 'tflite', 'openvino']

QUANTIZATION_MODES = ['int8', 'fp16', 'fp32']

# ============================================================================
# Annotation Modes (for reporting only)
# ============================================================================

ANNOTATION_MODES = [
    'DETECTION_ONLY',
    'SEGMENTATION_ONLY',
    'DETECTION_AND_SEGMENTATION'
]

# ============================================================================
# Device Settings
# ============================================================================

# Edge device targets
EDGE_DEVICES = {
    'jetson_orin': {'compute': 'high', 'memory': 'high'},
    'jetson_xavier': {'compute': 'medium', 'memory': 'medium'},
    'jetson_nano': {'compute': 'low', 'memory': 'low'},
    'raspberry_pi': {'compute': 'very_low', 'memory': 'low'},
    'mobile': {'compute': 'low', 'memory': 'low'}
}

# ============================================================================
# Logging
# ============================================================================

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# Version Info
# ============================================================================

PLATFORM_VERSION = '0.1.0'
PLATFORM_NAME = 'Grounded-SAM Edge Deployment Platform'

