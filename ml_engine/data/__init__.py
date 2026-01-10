"""Data processing module for COCO datasets."""

from .inspection import (
    inspect_dataset,
    get_required_models,
)
from .validators import (
    validate_coco_format,
    compute_bbox_from_mask,
    compute_area_from_mask,
    normalize_coco_annotations,
    check_data_quality,
    split_dataset
)
from .loaders import (
    COCODataset,
    TeacherDataset,
    collate_fn,
    create_dataloader
)
from .preprocessing import (
    MultiModelPreprocessor,
    BaseModelPreprocessor,
    SAMPreprocessor,
    GroundingDINOPreprocessor,
    YOLOPreprocessor,
    create_preprocessor_from_models
)
from .dataset_factory import DatasetFactory

__all__ = [
    # Inspection
    'inspect_dataset',
    'get_required_models',
    # Validation
    'validate_coco_format',
    'compute_bbox_from_mask',
    'compute_area_from_mask',
    'normalize_coco_annotations',
    'check_data_quality',
    'split_dataset',
    # Loading
    'COCODataset',
    'TeacherDataset',
    'collate_fn',
    'create_dataloader',
    # Dataset Factory
    'DatasetFactory',
    # Preprocessing
    'MultiModelPreprocessor',
    'BaseModelPreprocessor',
    'SAMPreprocessor',
    'GroundingDINOPreprocessor',
    'YOLOPreprocessor',
    'create_preprocessor_from_models'
]
