"""Data processing module for COCO datasets."""

from .dataset_factory import DatasetFactory
from .inspection import inspect_dataset
from .loaders import COCODataset, TeacherDataset, collate_fn, create_dataloader
from .preprocessing import (
    BaseModelPreprocessor,
    GroundingDINOPreprocessor,
    MultiModelPreprocessor,
    SAMPreprocessor,
    YOLOPreprocessor,
    create_preprocessor_from_models,
)
from .validators import (
    check_data_quality,
    compute_area_from_mask,
    compute_bbox_from_mask,
    normalize_coco_annotations,
    split_dataset,
    validate_coco_format,
)

__all__ = [
    # Inspection
    "inspect_dataset",
    # Validation
    "validate_coco_format",
    "compute_bbox_from_mask",
    "compute_area_from_mask",
    "normalize_coco_annotations",
    "check_data_quality",
    "split_dataset",
    # Loading
    "COCODataset",
    "TeacherDataset",
    "collate_fn",
    "create_dataloader",
    # Dataset Factory
    "DatasetFactory",
    # Preprocessing
    "MultiModelPreprocessor",
    "BaseModelPreprocessor",
    "SAMPreprocessor",
    "GroundingDINOPreprocessor",
    "YOLOPreprocessor",
    "create_preprocessor_from_models",
]
