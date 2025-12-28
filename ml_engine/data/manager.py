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

from core.config import load_json, save_json
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
    
    Responsibilities:
    - Load COCO JSON once (single source of truth)
    - Validate and auto-fix data (bbox from masks, etc.)
    - Inspect dataset once (cache results)
    - Split train/val/test (if needed)
    - Expose data through clean accessors
    - Resolve image paths from COCO file_name to actual filesystem paths
    
    Design principles:
    - Data is loaded ONCE in __init__
    - All operations are cached
    - Pure functions are called from here
    - Expose data
    
    Example:
        >>> from ml_engine.data.manager import DataManager
        >>> from ml_engine.data.dataset_factory import DatasetFactory
        >>> 
        >>> # Create manager (loads and validates once)
        >>> manager = DataManager(
        >>>     data_path='data/raw/annotations.json',
        >>>     image_paths=[
        >>>         '/profile/upload/2025/12/16/xxx1.jpeg',
        >>>         '/profile/upload/2025/12/16/xxx2.jpeg'
        >>>     ],
        >>>     split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
        >>> )
        >>> 
        >>> # Get data and metadata
        >>> train_data = manager.get_split('train')
        >>> dataset_info = manager.get_dataset_info()
        >>> required_models = manager.get_required_models()
    """

    def __init__(
        self,
        data_path: str,
        image_paths: List[str],
        split_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize DataManager - loads and processes data once.
        
        Data is always normalized to canonical form during initialization:
        - Original annotation mode is captured FIRST (for model selection)
        - Data is then normalized (bboxes generated from masks if missing)
        - Inspection happens on normalized data (for data loading decisions)
        
        This ensures no temporal coupling between mode detection and data state.
        
        Args:
            data_path: Path to COCO JSON file
            image_paths: List of image paths from frontend (with /profile/ prefix)
            split_config: Optional split ratios, e.g., {'train': 0.7, 'val': 0.2, 'test': 0.1}
                         If None, uses all data as single split
        """
        self.data_path = Path(data_path)
        self.split_config = split_config

        logger.info("=" * 60)
        logger.info("Initializing DataManager")
        logger.info("=" * 60)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        # Build file_name -> actual_path mapping from image_paths
        logger.info("Building image path mapping from %d image paths...", len(image_paths))
        self.image_path_map = self._build_image_path_map(image_paths)
        logger.info("  Image path map built with %d entries", len(self.image_path_map))

        logger.info("Loading dataset: %s", self.data_path)
        self.raw_data = load_json(str(self.data_path))

        logger.info("Validating COCO format...")
        is_valid, errors = validate_coco_format(self.raw_data)
        if not is_valid:
            logger.error("Dataset validation failed:")
            for error in errors:
                logger.error("  - %s", error)
            raise ValueError(f"Invalid COCO format: {len(errors)} errors found")
        logger.info(" Dataset format is valid")

        # Detect original annotation mode BEFORE normalization
        self.original_annotation_mode = detect_annotation_mode(self.raw_data)
        logger.info(" Original annotation mode: %s", self.original_annotation_mode)

        # This generates bboxes from masks if missing, ensuring canonical form
        logger.info("Normalizing annotations...")
        self.raw_data = normalize_coco_annotations(self.raw_data, in_place=True)
        logger.info(" Normalization complete")

        # Now has_boxes/has_masks reflect actual data state (for DATA LOADING)
        self.dataset_info = inspect_dataset(self.raw_data)
        logger.info(" Dataset inspection complete:")
        logger.info("  - Has boxes: %s", self.dataset_info['has_boxes'])
        logger.info("  - Has masks: %s", self.dataset_info['has_masks'])
        logger.info("  - Num classes: %d", self.dataset_info['num_classes'])
        logger.info("  - Classes: %s", list(self.dataset_info['class_mapping'].values()))

        # Step 6: Quality check
        logger.info("Checking data quality...")
        self.quality_report = check_data_quality(self.raw_data)

        # Validate that all COCO file_names can be resolved
        self._validate_image_paths()

        if self.quality_report['warnings']:
            logger.warning("Data quality warnings:")
            for warning in self.quality_report['warnings']:
                logger.warning("  - %s", warning)
        else:
            logger.info("  No data quality issues found")

        # Step 7: Split dataset (if configured)
        if split_config:
            logger.info("Splitting dataset: %s", split_config)
            self.splits = split_dataset(
                self.raw_data,
                splits=split_config,
                stratify=True,
                random_seed=42
            )
            logger.info("  Dataset split complete:")
            for split_name, split_data in self.splits.items():
                logger.info("  - %s: %d images, %d annotations",
                          split_name,
                          len(split_data['images']),
                          len(split_data['annotations']))
        else:
            self.splits = {'all': self.raw_data}
            logger.info("No splitting configured - using all data")

        logger.info("=" * 60)
        logger.info("DataManager initialized successfully")
        logger.info("=" * 60)

    def _build_image_path_map(self, image_paths: List[str]) -> Dict[str, str]:
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
            file_name = path
            path_map[file_name] = actual_path
        return path_map

    def _validate_image_paths(self):
        """
        Validate that all COCO file_names can be resolved to actual paths.
        
        Raises:
            FileNotFoundError: If any image path cannot be resolved or doesn't exist
        """
        annotated_filenames = {img['file_name'] for img in self.raw_data['images']}

        missing_files = []

        for file_name in annotated_filenames:
            actual_path = self.get_image_path(file_name)

            # Check if path exists on filesystem
            if not Path(actual_path).exists():
                missing_files.append((file_name, actual_path))

        if missing_files:
            logger.error("Found %d images that do not exist on filesystem:", len(missing_files))
            for file_name, actual_path in missing_files[:5]:  # Show first 5
                logger.error("  - %s -> %s", file_name, actual_path)
            if len(missing_files) > 5:
                logger.error("  ... and %d more", len(missing_files) - 5)
            raise FileNotFoundError(
                f"{len(missing_files)} images referenced in annotations not found on filesystem"
            )

        logger.info("  All %d image paths validated successfully", len(annotated_filenames))

    def get_image_path(self, file_name: str) -> str:
        """
        Get actual filesystem path for a COCO file_name.
        
        Args:
            file_name: The file_name from COCO annotation (e.g., /upload/2025/12/16/xxx.jpeg)
            
        Returns:
            Actual filesystem path (e.g., /srv/shared/images/upload/2025/12/16/xxx.jpeg)
        """
        if file_name in self.image_path_map:
            return self.image_path_map[file_name]

        # Fallback: apply transform directly to file_name
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

    def save_splits(self, output_dir: str) -> None:
        """
        Save all splits to separate JSON files.
        
        Useful after splitting to save train/val/test for later use.
        
        Args:
            output_dir: Directory to save split files
        
        Example:
            >>> manager.save_splits('data/processed/')
            >>> # Creates:
            >>> #   data/processed/train.json
            >>> #   data/processed/val.json
            >>> #   data/processed/test.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in self.splits.items():
            output_file = output_path / f'{split_name}.json'
            save_json(split_data, str(output_file))
            logger.info("Saved %s split to: %s", split_name, output_file)

    def __repr__(self) -> str:
        """String representation of DataManager."""
        splits_info = ', '.join([f"{k}: {len(v['images'])}" for k, v in self.splits.items()])
        return (f"DataManager(\n"
                f"  data_path={self.data_path},\n"
                f"  annotation_mode={self.dataset_info['annotation_mode']},\n"
                f"  num_classes={self.dataset_info['num_classes']},\n"
                f"  splits={{{splits_info}}}\n"
                f")")
