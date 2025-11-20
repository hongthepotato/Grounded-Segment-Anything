"""
Characteristic-Based Augmentation System
Flexible, user-friendly augmentation configuration without hardcoded domains

Usage:
    registry = get_augmentation_registry()
    pipeline = registry.get_pipeline(
        characteristics=["changes_shape", "reflective_surface"],
        environment={"lighting": "variable"},
        intensity="medium"
    )
    
    # Apply augmentation
    result = pipeline(image=img, masks=masks, keypoints=keypoints)
"""

# Primary API
from .augmentation_registry import (
    get_augmentation_registry,     # Main entry point
    AugmentationRegistry,           # Registry class
)

# Core components (for advanced usage)
from .characteristic_translator import CharacteristicTranslator
from .augmentation_factory import ConfigurableAugmentationPipeline
from .parameter_system import (
    AlbumentationsParameter,
    RangeParameter,
    NestedParameter,
    convert_to_numeric
)
from .transform_builders import TransformParameterBuilder

__all__ = [
    # Primary API
    "get_augmentation_registry",
    "AugmentationRegistry",
    # Core components
    "CharacteristicTranslator",
    "ConfigurableAugmentationPipeline",
    "AlbumentationsParameter",
    "RangeParameter",
    "NestedParameter",
    "TransformParameterBuilder",
    "convert_to_numeric",
]

__version__ = "2.0.0"  # Characteristic-based system
