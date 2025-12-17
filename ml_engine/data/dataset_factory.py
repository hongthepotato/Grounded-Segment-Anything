"""
Dataset Factory - Single source of truth for dataset creation.

This module handles all PyTorch dataset creation, including:
- Preprocessor creation (based on required models)
- Augmentation pipeline creation (based on config)
- TeacherDataset instantiation

Benefits:
- Decouples DataManager from PyTorch/preprocessing concerns
- Decouples TeacherTrainer from dataset creation details
- Single place for dataset creation logic
- Easy to test and maintain
"""

import logging
from typing import Dict, List, Any, Optional, Callable

from ml_engine.data.loaders import TeacherDataset
from ml_engine.data.preprocessing import create_preprocessor_from_models
from augmentation.augmentation_registry import get_augmentation_registry

logger = logging.getLogger(__name__)


class DatasetFactory:
    """
    Factory for creating PyTorch datasets with proper preprocessing and augmentation.
    
    This is the single source of truth for DATASET CREATION.
    All trainers should use this factory to create datasets.

    Responsibilities:
    - Create preprocessor based on required models
    - Create augmentation pipeline based on config
    - Instantiate TeacherDataset with all components

    Example:
        >>> from ml_engine.data.dataset_factory import DatasetFactory
        >>> 
        >>> train_dataset = DatasetFactory.create_dataset(
        >>>     coco_data=data_manager.get_split('train'),
        >>>     image_path_resolver=data_manager.get_image_path,
        >>>     dataset_info=data_manager.get_dataset_info(),
        >>>     model_names=['grounding_dino', 'sam'],
        >>>     augmentation_config={'enabled': True, ...},
        >>>     is_training=True
        >>> )
    """

    @staticmethod
    def create_dataset(
        coco_data: Dict[str, Any],
        image_path_resolver: Callable[[str], str],
        dataset_info: Dict[str, Any],
        model_names: List[str],
        augmentation_config: Optional[Dict[str, Any]] = None,
        is_training: bool = True
    ) -> TeacherDataset:
        """
        Create a complete PyTorch dataset with preprocessing and augmentation.
        
        This method handles all the complexity of dataset creation:
        1. Creates preprocessor based on model_names
        2. Creates augmentation pipeline based on config (only for training)
        3. Returns fully configured TeacherDataset

        Args:
            coco_data: COCO format dictionary (from DataManager.get_split())
            image_path_resolver: Function that resolves COCO file_name to actual path
            dataset_info: Dataset metadata (from DataManager.get_dataset_info())
                         Must contain 'has_boxes' and 'has_masks' keys
            model_names: List of model names requiring preprocessing
                        e.g., ['grounding_dino', 'sam']
            augmentation_config: Augmentation configuration dictionary
                                Must contain: 'enabled', 'characteristics', 
                                'environment', 'intensity'
                                Pass None to disable augmentation
            is_training: Whether this is a training dataset
                        Augmentation only applied if is_training=True

        Returns:
            TeacherDataset instance ready for use in DataLoader

        Example:
            >>> # Training dataset with augmentation
            >>> train_dataset = DatasetFactory.create_dataset(
            >>>     coco_data=train_data,
            >>>     image_path_resolver=data_manager.get_image_path,
            >>>     dataset_info={'has_boxes': True, 'has_masks': True},
            >>>     model_names=['grounding_dino', 'sam'],
            >>>     augmentation_config={
            >>>         'enabled': True,
            >>>         'characteristics': 'industrial',
            >>>         'environment': 'controlled',
            >>>         'intensity': 'medium'
            >>>     },
            >>>     is_training=True
            >>> )
            >>> 
            >>> # Validation dataset without augmentation
            >>> val_dataset = DatasetFactory.create_dataset(
            >>>     coco_data=val_data,
            >>>     image_path_resolver=data_manager.get_image_path,
            >>>     dataset_info={'has_boxes': True, 'has_masks': True},
            >>>     model_names=['grounding_dino', 'sam'],
            >>>     augmentation_config=None,  # No augmentation
            >>>     is_training=False
            >>> )
        """
        # Create preprocessor for required models
        preprocessor = create_preprocessor_from_models(model_names=model_names)
        logger.debug("Created preprocessor for models: %s", model_names)

        # Create augmentation pipeline (only for training with enabled config)
        augmentation_pipeline = None
        if is_training and augmentation_config and augmentation_config.get('enabled'):
            registry = get_augmentation_registry()
            augmentation_pipeline = registry.get_pipeline(
                characteristics=augmentation_config['characteristics'],
                environment=augmentation_config['environment'],
                intensity=augmentation_config['intensity']
            )
            logger.debug(
                "Created augmentation pipeline: %s/%s/%s",
                augmentation_config['characteristics'],
                augmentation_config['environment'],
                augmentation_config['intensity']
            )

        # Create dataset with all components
        dataset = TeacherDataset(
            coco_data=coco_data,
            image_path_resolver=image_path_resolver,
            preprocessor=preprocessor,
            augmentation_pipeline=augmentation_pipeline,
            return_boxes=dataset_info['has_boxes'],
            return_masks=dataset_info['has_masks']
        )

        split_type = "training" if is_training else "validation"
        aug_status = "with augmentation" if augmentation_pipeline else "without augmentation"
        logger.info(
            "Created %s dataset: %d samples, %s",
            split_type, len(dataset), aug_status
        )

        return dataset
