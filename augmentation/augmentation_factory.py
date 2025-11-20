"""
Augmentation Factory - Builds albumentations pipeline from configuration
Maintains API compatibility while using dict-based configs
"""

import logging
from typing import Dict, Any, Optional, List
import albumentations as A
import numpy as np
from .transform_builders import TransformParameterBuilder

logger = logging.getLogger(__name__)

class ConfigurableAugmentationPipeline:
    """
    Configurable augmentation pipeline with dict-based configurations
    Accept augmentations dict directly, no metadata needed
    """

    def __init__(self, augmentations: Dict[str, Dict[str, Any]]):
        r"""
        Initializa pipeline from augmentations dictonary

        Args:
            augmentations: Dict mapping augmentation_type -> unified_params
            Example: {
                "ElasticTransform": {"alpha": RangeParameter(...), "p": RangeParameter(...)}
                "MotionBlur": {"blue_limit": RangeParameter(...), "p": RangeParameter(...)}
            }

        Raises:
            ValueError: If augmentations is invalid
            TypeError: If augmentations has wrong type
        """
        self._validate_augmentations(augmentations)

        self.augmentations = augmentations
        self.parameter_builder = TransformParameterBuilder()

        # Build albumentations pipeline from configuration
        self.pipeline = self._create_pipeline()

        logger.info(
            "ConfigurableAugmentationPipeline initialized with %d augmentations",
            len(augmentations)
        )

    def _validate_augmentations(self, augmentations: Dict[str, Dict[str,Any]]) -> None:
        """
        Validate augmentations dictionary structure

        Args:
            augmentations: Augmentations dict to valid

        Raises:
            ValueError: If config structure is invalid
            TypeError: If config types are wrong
        """

        # Type validation
        if not isinstance(augmentations, dict):
            raise TypeError(f"augmentations must be a dictionary, got {type(augmentations).__name__}")

        # Allow empty dict - will create identity pipeline (no-op)
        if len(augmentations) == 0:
            logger.info("Empty augmentations dict - will create identity pipeline (no-op)")
            return

        # Validate each transform section
        for aug_type, params in augmentations.items():
            # Check augmentation type
            if not isinstance(aug_type, str):
                raise TypeError(
                    f"Augmentation type must be string, "
                    f"got {type(aug_type).__name__}"
                )

            # Check params is dict
            if not isinstance(params, dict):
                raise TypeError(
                    f"Parameters for '{aug_type}' must be dict, "
                    f"got {type(params).__name__}"
                )

            # Check if augmentation exists in albumentations (cheap check)
            if not hasattr(A, aug_type):
                raise ValueError(
                    f"Augmentation '{aug_type}' does not exist in albumentations."
                )

            if 'p' not in params:
                raise KeyError(
                    f"Probability for augmentation '{aug_type}' is not set."
                )

    def _validate_input_data(
            self, image: np.ndarray, masks: List[np.ndarray],
            keypoints: Optional[List[List[int]]] = None, bboxes: Optional[List[List[int]]] = None
        ) -> None:
        """
        Validate input data for augmentation
        
        Args:
            image: Input image (H, W, C)
            masks: Input masks (N, H, W), typical N = 1
            keypoints: Input keypoints (N, 2), typical N = 5
            bboxes: Input bounding boxes (N, 4), typical N = 1
        
        Raises:
            ValueError: If input data is invalid
            TypeError: If input data has wrong type
        """
        # Validate image
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy.ndarray, got {type(image).__name__}")

        if image.ndim not in [2, 3]:
            raise ValueError(
                f"image must be 2D or 3D array, got shape {image.shape}"
            )

        if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError(
                f"image must have 1, 3, or 4 channels, got {image.shape[2]}"
            )

        image_h, image_w = image.shape[:2]

        self._validate_masks(masks, image_h, image_w)

        self._validate_keypoints(keypoints, image_h, image_w)

        self._validate_bboxes(bboxes, image_h, image_w)

    def _validate_masks(self, masks: List[np.ndarray], image_h: int, image_w: int) -> None:
        """
        Validate masks, check the type of the masks and the shape of the masks

        Args:
            masks: List of masks (N, H, W), typical N = 1
            image_h: Image height
            image_w: Image width
        Raises:
            ValueError: If masks are invalid
            TypeError: If masks have wrong type
        """
        # None is valid (no masks)
        if masks is None:
            return

        # [] is valid
        if len(masks) == 0:
            return

        if not isinstance(masks, list):
            raise TypeError(f"masks must be a list, got {type(masks).__name__}")

        for mask in masks:
            if not isinstance(mask, np.ndarray):
                raise TypeError(
                    f"masks must be a list of numpy.ndarray, "
                    f"got {type(mask).__name__}"
                )

            if mask.ndim != 2:
                raise ValueError(
                    f"masks must be a list of 2D numpy.ndarray (H, W), "
                    f"got {mask.shape}"
                )

            if mask.shape[0] != image_h or mask.shape[1] != image_w:
                raise ValueError(
                    f"masks must have the same height and width as image, "
                    f"got {mask.shape[0]}x{mask.shape[1]} and {image_h}x{image_w}"
                )

    def _validate_keypoints(self, keypoints: List[List[int]], image_h: int, image_w: int) -> None:
        """
        Validate keypoints, check the type of the keypoints and the shape of the keypoints

        Args:
            keypoints: List of keypoints (N, 2), typical N = 5
                [] is valid (no keypoints)
                [[]] is invalid (malformed keypoint)
            image_h: Image height
            image_w: Image width
            
        Raises:
            TypeError: If keypoints is not a list or has wrong element types
            ValueError: If keypoint coordinates are invalid
        """
        # None is valid (no keypoints)
        if keypoints is None:
            return

        # Type check FIRST - before any operations
        if not isinstance(keypoints, list):
            raise TypeError(f"keypoints must be a list, got {type(keypoints).__name__}")

        # [] is valid (empty list = no keypoints)
        if len(keypoints) == 0:
            return

        # Now validate each keypoint
        for i, kp in enumerate(keypoints):
            if not isinstance(kp, list):
                raise TypeError(f"keypoint {i+1} must be a list, got {type(kp).__name__}")

            if len(kp) != 2:
                raise ValueError(
                    f"keypoint {i+1} must have exactly 2 coordinates (x, y), "
                    f"got {len(kp)}"
                )

            x, y = kp

            # Try to convert string-integers to int, but reject floats and string-floats
            try:
                # Handle string coordinates
                if isinstance(x, str):
                    # Check if it's a float string (contains '.' or 'e'/'E')
                    if '.' in x or 'e' in x.lower():
                        raise TypeError(
                            f"keypoint {i+1} coordinates must be integers, "
                            f"got x={type(x).__name__} (string-float '{x}'), y={type(y).__name__}"
                        )
                    x = int(x)

                if isinstance(y, str):
                    # Check if it's a float string (contains '.' or 'e'/'E')
                    if '.' in y or 'e' in y.lower():
                        raise TypeError(
                            f"keypoint {i+1} coordinates must be integers, "
                            f"got x={type(x).__name__}, y={type(y).__name__} (string-float '{y}')"
                        )
                    y = int(y)

                # Now check if they're integers (or convertible to integers)
                if not isinstance(x, (int, np.integer)):
                    raise TypeError(
                        f"keypoint {i+1} coordinates must be integers, "
                        f"got x={type(x).__name__}, y={type(y).__name__}"
                    )

                if not isinstance(y, (int, np.integer)):
                    raise TypeError(
                        f"keypoint {i+1} coordinates must be integers, "
                        f"got x={type(x).__name__}, y={type(y).__name__}"
                    )

            except ValueError as e:
                # int() conversion failed (e.g., "abc")
                raise TypeError(
                    f"keypoint {i+1} coordinates must be integers or string-integers, "
                    f"got x={type(kp[0]).__name__}, y={type(kp[1]).__name__}"
                ) from e

            if not (0 <= x < image_w and 0 <= y < image_h):
                raise ValueError(
                    f"keypoint {i+1} ({x}, {y}) out of bounds for image size "
                    f"(width={image_w}, height={image_h})"
                )

    def _validate_bboxes(self, bboxes: List[List[int]], image_h: int, image_w: int) -> None:
        """
        Validate bboxes, check the type of the bboxes and the shape of the bboxes

        Args:
            bboxes: List of bboxes (N, 4), typical N = 1
                [] is valid (no bboxes)
                [[]] is invalid (malformed bbox)
            image_h: Image height
            image_w: Image width
            
        Raises:
            TypeError: If bboxes is not a list or has wrong element types
            ValueError: If bbox coordinates are invalid
        """
        # None is valid (no bboxes)
        if bboxes is None:
            return

        # Type check FIRST - before any operations
        if not isinstance(bboxes, list):
            raise TypeError(f"bboxes must be a list, got {type(bboxes).__name__}")

        # [] is valid (empty list = no bboxes)
        if len(bboxes) == 0:
            return

        # Now validate each bbox
        for i, bbox in enumerate(bboxes):
            if not isinstance(bbox, list):
                raise TypeError(f"bbox {i+1} must be a list, got {type(bbox).__name__}")

            if len(bbox) != 4:
                raise ValueError(f"bbox {i+1} must have exactly 4 elements, got {len(bbox)}")

            x_min, y_min, x_max, y_max = bbox

            # Convert string-integers to int, but reject floats and string-floats
            try:
                coords = []
                coord_names = ['x_min', 'y_min', 'x_max', 'y_max']
                for coord, name in zip([x_min, y_min, x_max, y_max], coord_names):
                    if isinstance(coord, str):
                        # Check if it's a float string
                        if '.' in coord or 'e' in coord.lower():
                            raise TypeError(
                                f"bbox {i+1} coordinates must be integers, "
                                f"{name} is string-float '{coord}'"
                            )
                        coords.append(int(coord))
                    elif isinstance(coord, (int, np.integer)):
                        coords.append(int(coord))
                    else:
                        raise TypeError(
                            f"bbox {i+1} coordinates must be integers, "
                            f"{name}={type(coord).__name__}"
                        )

                x_min, y_min, x_max, y_max = coords

            except ValueError as e:
                # int() conversion failed
                raise TypeError(
                    f"bbox {i+1} coordinates must be integers or string-integers, "
                    f"got x_min={type(bbox[0]).__name__}, y_min={type(bbox[1]).__name__}, "
                    f"x_max={type(bbox[2]).__name__}, y_max={type(bbox[3]).__name__}"
                ) from e

            # Validate bounds
            x_in_bounds = all(0 <= x < image_w for x in [x_min, x_max])
            y_in_bounds = all(0 <= y < image_h for y in [y_min, y_max])

            if not (x_in_bounds and y_in_bounds):
                raise ValueError(
                    f"bbox {i+1} ({x_min}, {y_min}, {x_max}, {y_max}) out of bounds for image size "
                    f"(width={image_w}, height={image_h})"
                )

            # Validate bbox format (x_min <= x_max, y_min <= y_max)
            if x_min > x_max:
                raise ValueError(
                    f"bbox {i+1} invalid: x_min ({x_min}) > x_max ({x_max})"
                )
            if y_min > y_max:
                raise ValueError(
                    f"bbox {i+1} invalid: y_min ({y_min}) > y_max ({y_max})"
                )


    def _create_pipeline(self) -> A.Compose:
        """
        Create albumentations pipeline from augmentations dict

        Returns:
            Albumentations Compose pipeline ready for use
        """
        all_transforms = []

        for aug_type, unified_params in self.augmentations.items():
            transform = self._instantiate_transform(aug_type, unified_params)
            if transform:
                all_transforms.append(transform)
                logger.debug("  - Added: %s", aug_type)

        logger.info("Created pipeline with %d total transforms", len(all_transforms))

        # Create albumentations pipeline
        return A.Compose(
            all_transforms,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            bbox_params=A.BboxParams(
                format='coco', # [x, y, w, h] for bounding boxes
                label_fields=[],     # Temporarily empty
                min_visibility=0.3
            ),
            # additional_targets={}  # Regard masks as image
        )

    def _instantiate_transform(
        self, aug_type: str, unified_params: Dict[str, Any]
    ) -> Optional[A.BasicTransform]:
        """
        Create albumentations transform from range-based configuration
        Unknown augmentation will just skip

        Args:
            aug_type: Augmentation type name (e.g. "ElasticTransform")
            unified_params: Unified parameter dict

        Returns:
            Albumentations transform instance or None if failed
        """
        try:
            # Get the appropriate parameter builder method
            builder_method = self.parameter_builder.get_builder_method(aug_type)
            processed_params = builder_method(unified_params)

            # Create transform
            transform_class = getattr(A, aug_type)
            transform_instance = transform_class(**processed_params)
            logger.debug("Created %s with %s", aug_type, processed_params)
            return transform_instance

        except TypeError as e:
            logger.error(
                "Parameter signature mismatch for %s: %s\n"
                "Provided params: %s\n"
                "This transform will be skipped.",
                aug_type, str(e), list(unified_params.keys())
            )
            return None

        except ValueError as e:
            logger.error(
                "Invalid parameter values for %s: %s\n"
                "This transform will be skipped.",
                aug_type, str(e)
            )
            return None

        except KeyError as e:
            logger.error(
                "Missing required parameter for %s: %s\n"
                "This transform will be skipped.",
                aug_type, str(e)
            )
            return None

        except Exception as e:
            logger.error(
                "Unexpected error instantiating %s: %s\n"
                "This transform will be skipped.",
                aug_type, str(e), exc_info=True
            )
            return None

    def __call__(
        self, image: np.ndarray, 
        masks: Optional[List[np.ndarray]] = None,
        keypoints: Optional[List[List[int]]] = None,
        bboxes: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Apply augmentation pipeline - EXACT SAME API as IndustrialAugmentationPipeline
        
        CRITICAL: This method must maintain 100% API compatibility
        
        Args:
            image: Input image (H, W, C) - numpy array
            masks: List of masks (optional) - (N, H, W), typical N = 1
            keypoints: List of keypoints (optional) - (N, 2), typical N = 5
            bboxes: List of bounding boxes (optional) - (N, 4), typical N = 1

        Returns:
            Dictionary with augmented data - SAME FORMAT as original:
            {
                "image": augmented_image,
                "masks": augmented_masks,  # List format can be []
                "keypoints": augmented_keypoints,  # List format can be []
                "bboxes": augmented_bboxes,  # List format can be []
            }
        """

        self._validate_input_data(image, masks, keypoints, bboxes)

        # Prepare augmentation inputs
        aug_input = {"image": image}

        has_masks = masks is not None and len(masks) > 0
        has_keypoints = keypoints is not None and len(keypoints) > 0
        has_bboxes = bboxes is not None and len(bboxes) > 0

        if has_masks:
            aug_input["masks"] = np.stack(masks, axis=0)
            logger.debug("Stacked %d masks for augmentation", len(masks))

        if has_keypoints:
            aug_input["keypoints"] = keypoints
            logger.debug("Added %d keypoints for augmentation", len(keypoints))

        if has_bboxes:
            aug_input["bboxes"] = bboxes
            logger.debug("Added %d bounding boxes for augmentation", len(bboxes))

        # Apply augmentations using albumentations pipeline
        # try:
        #     augmented = self.pipeline(**aug_input)
        #     logger.debug("Augmentation pipeline applied successfully")
        # except Exception as e:
        #     logger.error("Augmentation failed: %s", e)
        #     # Fallback: return original data
        #     return {
        #         "image": image,
        #         "masks": masks,
        #         "keypoints": keypoints or [],
        #         "bboxes": bboxes or []
        #     }

        augmented = self.pipeline(**aug_input)
        logger.debug("Augmentation pipeline applied successfully")


        # Build result dictionary
        result = {"image": augmented["image"]}

        # === MASKS PROCESSING ===
        if has_masks:
            if 'masks' not in augmented or augmented['masks'] is None:
                raise RuntimeError(
                    "Augmentation corrupted: provided masks but got none back."
                )
            if isinstance(augmented["masks"], np.ndarray) and augmented["masks"].ndim == 3:
                result["masks"] = [augmented["masks"][i] for i in range(augmented["masks"].shape[0])]
            else:
                result["masks"] = augmented["masks"]
            logger.debug("Processed %d masks", len(result["masks"]))
        else:
            result["masks"] = []

        # === KEYPOINTS PROCESSING ===
        if has_keypoints:
            if 'keypoints' not in augmented or augmented['keypoints'] is None:
                raise RuntimeError(
                    "Augmentation corrupted: provided keypoints but got none back."
                )
            kpts = augmented["keypoints"]
            if isinstance(kpts, np.ndarray):
                result["keypoints"] = kpts.tolist()
            else:
                result["keypoints"] = kpts
            logger.debug("Processed %d keypoints", len(result["keypoints"]))
        else:
            # No keypoints provided, return empty list
            result["keypoints"] = []

        # === BBOXES PROCESSING ===
        if has_bboxes:
            if 'bboxes' not in augmented or augmented['bboxes'] is None:
                raise RuntimeError(
                    "Augmentation corrupted: provided bboxes but got none back."
                )
            bboxes = augmented["bboxes"]
            if isinstance(bboxes, np.ndarray):
                result["bboxes"] = bboxes.tolist()
            else:
                result["bboxes"] = bboxes
            logger.debug("Processed %d bounding boxes", len(result["bboxes"]))
        else:
            # No bboxes provided, return empty list
            result["bboxes"] = []

        return result

    # def get_pipeline_summary(self) -> Dict[str, Any]:
    #     """
    #     Get summary information about the pipeline
    #     Useful for debugging and logging

    #     Returns:
    #         Pipeline summary information
    #     """
    #     transform_count = 0
    #     transform_types = []

    #     for transform_section in self.config.get("transforms", []):
    #         for aug_config in transform_section.get("augmentations", []):
    #             transform_count += 1
    #             transform_types.append(aug_config["type"])

    #     return {
    #         "intensity": self.intensity,
    #         "characteristics": self.characteristics,
    #         "environment": self.environment,
    #         "total_transforms": transform_count,
    #         "transform_types": transform_types,
    #         "config_sections": len(self.config.get("transforms", []))
    #     }
