"""
Augmentation Registry - Unified pipeline access without YAML complexity
Provide single entry point for both legacy domain-based and new characteristic-based augmentation
"""

import logging
from typing import Any, Dict, List, Optional


from .characteristic_translator import CharacteristicTranslator
from .augmentation_factory import ConfigurableAugmentationPipeline

logger = logging.getLogger(__name__)


class AugmentationRegistry:
    """
    Unified registry for augmentation pipeline access
    Simplified design: no YAML loading, pure computational approach
    """

    def _validate_intensity(self, intensity: str) -> None:
        """Validate intensity at registry entry point"""
        valid_intensities = {"low", "medium", "high"}
        if intensity not in valid_intensities:
            raise ValueError(
                f"Invalid intensity '{intensity}'. "
                f"Must be one of: {', '.join(sorted(valid_intensities))}"
            )

    def __init__(self):
        """
        Initialize registry with characteristic translator
        No file system dependencies - pure in-memory operation
        """
        self.translator = CharacteristicTranslator()
        logger.info("Augmentation Registry initialized")

    def get_pipeline(
        self,
        characteristics: List[str],
        environment: Optional[Dict[str, str]] = None,
        intensity: str = "medium"
    ) -> ConfigurableAugmentationPipeline:
        """
        Get augmentation pipeline using characteristic-based approach
        
        Usage:
            pipeline = registry.get_pipeline(
                characteristics=["changes_shape", "reflective_surface"],
                environment={"lighting": "variable"},
                intensity="medium"
            )

        Args:
            characteristics: List of object characteristics (required)
            environment: Environment conditions (optional)
            intensity: Augmentation intensity (low, medium, high)
            augmentation_prob: Probability of applying augmentations

        Returns:
            ConfigurableAugmentationPipeline ready for use

        Raises:
            ValueError: If characteristics are invalid
        """

        # Validate inputs at entry point (fail fast)
        self._validate_intensity(intensity)

        # Allow empty inputs - will create identity pipeline (no-op)
        # Useful for dynamic configurations or conditional augmentation scenarios
        logger.info("Creating pipeline for characteristics: %s", characteristics)

        # Validate characteristics if provided - fail on invalid
        if characteristics:
            validation = self.translator.validate_characteristics(characteristics)
            if not validation["valid"]:
                raise ValueError(
                    f"Invalid characteristics: {validation['unsupported_characteristics']}. "
                    f"Available: {validation['available_characteristics']}"
                )

        # Validate environment if provided - fail on invalid
        if environment:
            env_validation = self.translator.validate_environment(environment)
            if not env_validation["valid"]:
                raise ValueError(f"Invalid environment: {env_validation['errors']}")

        # Generate configuration
        config = self.translator.translate_from_characteristics(
            characteristics=characteristics,
            environment=environment,
            intensity=intensity
        )

        logger.info("Generated config: %d augmentations from %d rules",
            len(config['augmentations']),
            len(config['metadata']['applied_rules']))
        return ConfigurableAugmentationPipeline(config['augmentations'])

    def get_available_characteristics(self) -> List[str]:
        """Get all available characteristics for GUI selection"""
        return self.translator.get_available_characteristics()

    def get_available_environments(self) -> Dict[str, List[str]]:
        """Get all available environment options for GUI"""
        return self.translator.get_available_environments()


    def get_pipeline_info(
        self,
        characteristics: List[str],
        environment: Optional[Dict[str, str]] = None,
        intensity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Get information about what pipeline would be generated (without creating it)
        Useful for GUI preview and debugging
        
        Args:
            characteristics: Characteristics to analyze (required)
            environment: Environment conditions (optional)
            intensity: Intensity level
            
        Returns:
            Pipeline information without creating actual pipeline
        """
        config = self.translator.translate_from_characteristics(
            characteristics, environment, intensity
        )

        # Extract pipeline information
        augmentation_types = list(config["augmentations"].keys())

        return {
            "description": config["metadata"]["description"],
            "total_augmentations": len(augmentation_types),
            "augmentation_types": augmentation_types,
            "intensity": config["intensity"],
            "characteristics": config["characteristics"],
            "environment": config["environment"],
            "applied_rules": config["metadata"]["applied_rules"]
        }

# Global registry instance (singleton pattern for performance)
_registry_instance: Optional[AugmentationRegistry] = None

def get_augmentation_registry() -> AugmentationRegistry:
    """
    Get global augmentation registry instance (singleton)
    
    Returns:
        Shared AugmentationRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AugmentationRegistry()
        logger.info("Created global AugmentationRegistry instance")
    return _registry_instance
