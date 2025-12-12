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
from ml_engine.data.inspection import inspect_dataset, get_required_models
from ml_engine.data.validators import (
    validate_coco_format,
    preprocess_coco_dataset,
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
        >>>     image_dir='data/raw/images',
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
        image_dir: str,
        split_config: Optional[Dict[str, float]] = None,
        auto_preprocess: bool = True
    ):
        """
        Initialize DataManager - loads and processes data once.
        
        Args:
            data_path: Path to COCO JSON file
            image_dir: Directory containing images
            split_config: Optional split ratios, e.g., {'train': 0.7, 'val': 0.2, 'test': 0.1}
                         If None, uses all data as single split
            validate: Whether to validate COCO format
            auto_preprocess: Whether to auto-generate missing bbox/area from masks
        """
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.split_config = split_config

        logger.info("=" * 60)
        logger.info("Initializing DataManager")
        logger.info("=" * 60)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image path is not a directory: {self.image_dir}")

        # Step 1: Load JSON
        logger.info("Loading dataset: %s", self.data_path)
        self.raw_data = load_json(str(self.data_path))

        # Step 2: Validate COCO format
        logger.info("Validating COCO format...")
        is_valid, errors = validate_coco_format(self.raw_data)
        if not is_valid:
            logger.error("Dataset validation failed:")
            for error in errors:
                logger.error("  - %s", error)
            raise ValueError(f"Invalid COCO format: {len(errors)} errors found")
        logger.info(" Dataset format is valid")

        # Step 3: Auto-preprocess (generate bbox from masks, etc.)
        if auto_preprocess:
            logger.info("Auto-preprocessing dataset...")
            self.raw_data = preprocess_coco_dataset(self.raw_data, in_place=True)
            logger.info(" Auto-preprocessing complete")

        # Step 4: Inspect dataset ONCE (cache results)
        logger.info("Inspecting dataset...")
        self.dataset_info = inspect_dataset(self.raw_data)
        logger.info(" Dataset inspection complete:")
        logger.info("  - Annotation mode: %s", self.dataset_info['annotation_mode'])
        logger.info("  - Has boxes: %s", self.dataset_info['has_boxes'])
        logger.info("  - Has masks: %s", self.dataset_info['has_masks'])
        logger.info("  - Num classes: %d", self.dataset_info['num_classes'])
        logger.info("  - Classes: %s", list(self.dataset_info['class_mapping'].values()))

        # Step 5: Quality check
        logger.info("Checking data quality...")
        self.quality_report = check_data_quality(self.raw_data)

        # Check for image directory sync (warn if extra images exist)
        self._check_image_directory_sync()

        if self.quality_report['warnings']:
            logger.warning("Data quality warnings:")
            for warning in self.quality_report['warnings']:
                logger.warning("  - %s", warning)
        else:
            logger.info(" No data quality issues found")

        # Step 6: Split dataset (if configured)
        if split_config:
            logger.info("Splitting dataset: %s", split_config)
            self.splits = split_dataset(
                self.raw_data,
                splits=split_config,
                stratify=True,
                random_seed=42
            )
            logger.info(" Dataset split complete:")
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

    def _check_image_directory_sync(self):
        """
        Check if image directory has more images than annotated.
        
        Warns user if there are extra images in the directory that
        are not referenced in the annotations.
        """
        # Get annotated image filenames from COCO data
        annotated_filenames = {img['file_name'] for img in self.raw_data['images']}

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        directory_images = set()
        for file_path in self.image_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                directory_images.add(file_path.name)

        extra_images = directory_images - annotated_filenames

        if extra_images:
            logger.warning(
                "Found %d images in directory but NOT in annotations (will be ignored during training)",
                len(extra_images)
            )
            logger.warning("  - Annotated images: %d", len(annotated_filenames))
            logger.warning("  - Total images in directory: %d", len(directory_images))
            logger.warning("  - Extra images: %d", len(extra_images))

        missing_images = annotated_filenames - directory_images
        if missing_images:
            logger.error(
                "Found %d images in annotations but NOT in directory (training will fail!)",
                len(missing_images)
            )
            raise FileNotFoundError(
                f"{len(missing_images)} images referenced in annotations not found in {self.image_dir}"
            )

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
        Get list of required models based on dataset annotations.
        
        Data-driven: Determines which models to load based on available annotations.
        
        Returns:
            List of model names, e.g., ['grounding_dino', 'sam']
        """
        return get_required_models(self.dataset_info)

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

    # def get_available_splits(self) -> List[str]:
    #     """Get list of available split names."""
    #     return list(self.splits.keys())

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
