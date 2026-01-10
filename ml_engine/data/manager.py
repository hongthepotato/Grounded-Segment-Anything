"""
Data Manager - Single Source of Truth for DATA operations.

This module owns ALL data operations:
- Load raw COCO data once
- Validate and preprocess data
- Inspect dataset metadata
- Split into train/val/test
- Cache all results
- Expose data through accessors

No other module should directly load COCO JSON files.
Everyone gets DATA from this manager.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from core.config import load_json
from core.constants import transform_image_path
from ml_engine.data.inspection import (
    inspect_dataset,
    detect_annotation_mode,
    get_required_models_from_mode
)
from ml_engine.data.validators import (
    validate_coco_format,
    normalize_coco_annotations,
    split_dataset,
    check_data_quality
)

logger = logging.getLogger(__name__)


class DataManager:
    """
    Central orchestrator for all DATA operations.
    
    Two ways to create:
    1. DataManager.from_file(path, image_paths) - loads from disk, validates, normalizes
    2. DataManager(raw_data, image_path_map, ...) - direct construction for testing

    Responsibilities:
    - Load COCO JSON once (single source of truth)
    - Validate and auto-fix data (bbox from masks, etc.)
    - Inspect dataset once (cache results)
    - Split train/val/test (if needed)
    - Expose data through clean accessors
    - Resolve image paths from COCO file_name to actual filesystem paths
    
    Example (production):
        >>> manager = DataManager.from_file(
        >>>     data_path='data/raw/annotations.json',
        >>>     image_paths=[
        >>>         'upload/2025/12/16/xxx1.jpeg',
        >>>         'upload/2025/12/16/xxx2.jpeg'
        >>>     ],
        >>>     split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> )
        >>> 
        >>> train_data = manager.get_split('train')
        >>> dataset_info = manager.get_dataset_info()
        >>> required_models = manager.get_required_models()
    
    Example (testing):
        >>> manager = DataManager(
        >>>     raw_data=mock_coco_data,
        >>>     image_path_map={'img.jpg': '/tmp/img.jpg'},
        >>>     original_annotation_mode='combined'
        >>> )
    """

    def __init__(
        self,
        raw_data: Dict[str, Any],
        image_path_map: Dict[str, str],
        original_annotation_mode: str,
        splits: Optional[Dict[str, Dict[str, Any]]] = None,
        data_path: Optional[Path] = None
    ):
        """
        Direct constructor - stores pre-processed data.
        
        Args:
            raw_data: Normalized COCO data (already validated)
            image_path_map: Mapping from file_name to filesystem path
            original_annotation_mode: 'detection', 'segmentation', or 'combined'
            splits: Pre-computed splits, or None for single 'all' split
            data_path: Optional original file path (for logging/debugging)
        """
        self.raw_data = raw_data
        self.image_path_map = image_path_map
        self.original_annotation_mode = original_annotation_mode
        self.data_path = data_path

        # Compute derived data (cheap operations only)
        self.dataset_info = inspect_dataset(raw_data)
        self.quality_report = check_data_quality(raw_data)
        self.splits = splits if splits else {'all': raw_data}

    @classmethod
    def from_file(
        cls,
        data_path: str,
        image_paths: List[str],
        split_config: Optional[Dict[str, float]] = None,
        stratify: bool = True,
        random_seed: int = 42
    ) -> 'DataManager':
        """
        Factory method - loads from disk with full validation.

        - File I/O
        - COCO validation
        - Annotation normalization
        - Image path resolution and validation
        - Dataset splitting
        
        Args:
            data_path: Path to COCO JSON file
            image_paths: List of image paths from frontend
            split_config: Optional split ratios, e.g., {'train': 0.7, 'val': 0.2, 'test': 0.1}
                         If None, uses all data as single split
            stratify: Whether to use stratified splitting (default: True)
            random_seed: Random seed for splitting (default: 42)
        
        Returns:
            Fully initialized DataManager
        
        Raises:
            FileNotFoundError: If data file or images don't exist
            ValueError: If COCO format is invalid
        """
        data_path_obj = Path(data_path)

        logger.info("=" * 60)
        logger.info("Loading DataManager from file")
        logger.info("=" * 60)

        if not data_path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path_obj}")

        logger.info("Building image path mapping from %d paths...", len(image_paths))
        image_path_map = cls._build_image_path_map(image_paths)
        logger.info("  Image path map built with %d entries", len(image_path_map))

        logger.info("Loading dataset: %s", data_path_obj)
        raw_data = load_json(str(data_path_obj))

        logger.info("Validating COCO format...")
        is_valid, errors = validate_coco_format(raw_data)
        if not is_valid:
            logger.error("Dataset validation failed:")
            for error in errors:
                logger.error("  - %s", error)
            raise ValueError(f"Invalid COCO format: {len(errors)} errors found")
        logger.info("  Dataset format is valid")

        original_mode = detect_annotation_mode(raw_data)
        logger.info("  Original annotation mode: %s", original_mode)

        logger.info("Normalizing annotations...")
        raw_data = normalize_coco_annotations(raw_data, in_place=True)
        logger.info("  Normalization complete")

        cls._validate_image_paths_exist(raw_data, image_path_map)

        splits = None
        if split_config:
            logger.info("Splitting dataset: %s", split_config)
            splits = split_dataset(
                raw_data,
                splits=split_config,
                stratify=stratify,
                random_seed=random_seed
            )
            for split_name, split_data in splits.items():
                logger.info("  - %s: %d images, %d annotations",
                          split_name,
                          len(split_data['images']),
                          len(split_data['annotations']))

        logger.info("=" * 60)
        logger.info("DataManager loaded successfully")
        logger.info("=" * 60)

        return cls(
            raw_data=raw_data,
            image_path_map=image_path_map,
            original_annotation_mode=original_mode,
            splits=splits,
            data_path=data_path_obj
        )

    @staticmethod
    def _build_image_path_map(image_paths: List[str]) -> Dict[str, str]:
        """
        Build mapping from COCO file_name to actual filesystem path.
        
        Frontend sends paths like: upload/2025/12/17/xxx.png
        COCO file_name contains:   upload/2025/12/17/xxx.png (same format)
        Actual filesystem path:    /srv/shared/images/upload/2025/12/17/xxx.png
        
        Args:
            image_paths: List of image paths from frontend
            
        Returns:
            Dictionary mapping COCO file_name to actual filesystem path
        """
        path_map = {}
        for path in image_paths:
            actual_path = transform_image_path(path)
            path_map[path] = actual_path
        return path_map

    @staticmethod
    def _validate_image_paths_exist(
        raw_data: Dict[str, Any],
        image_path_map: Dict[str, str]
    ) -> None:
        """
        Validate that all COCO file_names resolve to existing files.
        
        Raises:
            FileNotFoundError: If any image path cannot be resolved or doesn't exist
        """
        annotated_filenames = {img['file_name'] for img in raw_data['images']}
        missing_files = []

        for file_name in annotated_filenames:
            actual_path = image_path_map.get(file_name, transform_image_path(file_name))
            if not Path(actual_path).exists():
                missing_files.append((file_name, actual_path))

        if missing_files:
            logger.error("Found %d images that do not exist on filesystem:", len(missing_files))
            for file_name, actual_path in missing_files[:5]:
                logger.error("  - %s -> %s", file_name, actual_path)
            if len(missing_files) > 5:
                logger.error("  ... and %d more", len(missing_files) - 5)
            raise FileNotFoundError(
                f"{len(missing_files)} images referenced in annotations not found on filesystem"
            )

        logger.info("  All %d image paths validated successfully", len(annotated_filenames))

    # =========================================================================
    # Public API
    # =========================================================================

    def get_image_path(self, file_name: str) -> str:
        """
        Get actual filesystem path for a COCO file_name.
        
        Args:
            file_name: The file_name from COCO annotation (e.g., upload/2025/12/16/xxx.jpeg)
            
        Returns:
            Actual filesystem path (e.g., /srv/shared/images/upload/2025/12/16/xxx.jpeg)
        """
        if file_name in self.image_path_map:
            return self.image_path_map[file_name]
        return transform_image_path(file_name)

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get cached dataset inspection results.
        
        Returns:
            Dictionary with:
                - has_boxes: bool
                - has_masks: bool
                - num_classes: int
                - class_mapping: Dict[int, str] - category_id to name
                - category_id_to_index: Dict[int, int] - category_id to 0-based index
                - index_to_category_id: Dict[int, int] - 0-based index to category_id
                - num_images: int
                - num_annotations: int
                - annotation_mode: str
                - class_counts: Dict[int, int]
        """
        return self.dataset_info

    def get_required_models(self) -> List[str]:
        """
        Get list of required models based on ORIGINAL annotation mode.
        
        This uses the annotation mode captured BEFORE normalization to determine
        which teacher models should be loaded. This ensures model selection
        reflects user intent, not auto-generated data.
        
        Returns:
            List of model names, e.g., ['grounding_dino', 'sam']
            - 'detection' mode -> ['grounding_dino']
            - 'segmentation' mode -> ['sam']
            - 'combined' mode -> ['grounding_dino', 'sam']
        """
        return get_required_models_from_mode(self.original_annotation_mode)

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get cached data quality report.
        
        Returns:
            Dictionary with quality metrics and warnings
        """
        return self.quality_report

    def get_split(self, split_name: str) -> Dict[str, Any]:
        """
        Get a specific data split.
        
        Args:
            split_name: Name of split ('train', 'val', 'test', or 'all')
        
        Returns:
            COCO format dictionary for the split
        
        Raises:
            ValueError: If split_name doesn't exist
        """
        if split_name not in self.splits:
            available = list(self.splits.keys())
            raise ValueError(f"Split '{split_name}' not found. Available: {available}")
        return self.splits[split_name]


    def __repr__(self) -> str:
        """String representation of DataManager."""
        splits_info = ', '.join([f"{k}: {len(v['images'])}" for k, v in self.splits.items()])
        path_info = f"data_path={self.data_path}, " if self.data_path else ""
        return (f"DataManager(\n"
                f"  {path_info}\n"
                f"  annotation_mode={self.original_annotation_mode},\n"
                f"  num_classes={self.dataset_info['num_classes']},\n"
                f"  splits={{{splits_info}}}\n"
                f")")
