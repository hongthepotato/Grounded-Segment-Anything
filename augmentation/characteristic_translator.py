"""
Characteristic Translator - Range-based parameter system
Converts user characteristics to agumentation configurations with proper range
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .parameter_system import AlbumentationsParameter, RangeParameter, NestedParameter

logger = logging.getLogger(__name__)



@dataclass
class AugmentationRule:
    """Rule for translating a characteristic to augmentations with ranges"""
    augmentations: List[str]
    reason: str
    intensity_ranges: Dict[str, Dict[str, AlbumentationsParameter]]

    def __post_init__(self):
        """Validate rule structure at initialization"""
        # Validate required intensity levels
        required_intensities = {"low", "medium", "high"}
        provided = set(self.intensity_ranges.keys())

        missing = required_intensities - provided
        if missing:
            raise ValueError(
                f"AugmentationRule missing intensity levels: {sorted(missing)}. "
                f"Required: {sorted(required_intensities)}"
            )

        # Validate at least one augmentation
        if not self.augmentations:
            raise ValueError("AugmentationRule must have at least one augmentation")

        # Validate reason is not empty
        if not self.reason or not self.reason.strip():
            raise ValueError("AugmentationRule must have a non-empty reason")


class CharacteristicTranslator:
    """
    Translates user characteristics to technical augmentation configurations
    Uses proper parameter ranges for realistic augmentation variation
    """

    # Valid intensity levels
    VALID_INTENSITIES = {"low", "medium", "high"}

    # Characteristic-based rules with proper ranges
    CHARACTERISTIC_RULES = {
        "changes_shape": AugmentationRule(
            augmentations=["ElasticTransform", "PiecewiseAffine"],
            reason="Object can deform and change shape",
            intensity_ranges={
                "low": {
                    "ElasticTransform": {
                        "alpha": RangeParameter(0.2, 0.8),
                        "sigma": RangeParameter(20, 40),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "PiecewiseAffine": {
                        "scale": RangeParameter(0.01, 0.03),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "ElasticTransform": {
                        "alpha": RangeParameter(0.5, 1.5),
                        "sigma": RangeParameter(30, 70),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "PiecewiseAffine": {
                        "scale": RangeParameter(0.02, 0.05),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "ElasticTransform": {
                        "alpha": RangeParameter(1.0, 3.0),
                        "sigma": RangeParameter(50, 100),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "PiecewiseAffine": {
                        "scale": RangeParameter(0.03, 0.08),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "changes_size": AugmentationRule(
            augmentations=["RandomScale", "RandomSizedBBoxSafeCrop"],
            reason="Object appears at different sizes in images",
            intensity_ranges={
                "low": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.1, 0.1),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "medium": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.2, 0.2),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.6)
                    }
                },
                "high": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.4, 0.4),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.8)
                    }
                }
            }
        ),

        "reflective_surface": AugmentationRule(
            augmentations=["RandomSunFlare", "RandomShadow", "ColorJitter"],
            reason="Reflective surfaces create lighting variations",
            intensity_ranges={
                "low": {
                    "RandomSunFlare": {
                        "flare_roi": (0, 0, 1, 1),
                        "src_radius": RangeParameter(200, 300),
                        "p": RangeParameter.scalar(0.05)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(1, 2),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "ColorJitter": {
                        "brightness": RangeParameter(0.9, 1.1),
                        "contrast": RangeParameter(0.9, 1.1),
                        "saturation": RangeParameter(0.9, 1.1),
                        "hue": RangeParameter(-0.3, 0.3),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomSunFlare": {
                        "flare_roi": (0, 0, 1, 1),
                        "src_radius": RangeParameter(300, 400),
                        "p": RangeParameter.scalar(0.15)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(2, 3),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "ColorJitter": {
                        "brightness": RangeParameter(0.8, 1.2),
                        "contrast": RangeParameter(0.8, 1.2),
                        "saturation": RangeParameter(0.8, 1.2),
                        "hue": RangeParameter(-0.5, 0.5),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomSunFlare": {
                        "flare_roi": (0, 0, 1, 1),
                        "src_radius": RangeParameter(400, 500),
                        "p": RangeParameter.scalar(0.3)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(3, 4),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "ColorJitter": {
                        "brightness": RangeParameter(0.7, 1.3),
                        "contrast": RangeParameter(0.7, 1.3),
                        "saturation": RangeParameter(0.7, 1.3),
                        "hue": RangeParameter(-0.7, 0.7),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "low_contrast": AugmentationRule(
            augmentations=["CLAHE", "RandomBrightnessContrast", "Sharpen"],
            reason="Enhances visibility of hard-to-see objects",
            intensity_ranges={
                "low": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(0.8, 2.0),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.15, 0.15),
                        "contrast_limit": RangeParameter(-0.15, 0.15),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.1, 0.25),
                        "lightness": RangeParameter(0.25, 0.5),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(1.0, 4.0),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.25, 0.25),
                        "contrast_limit": RangeParameter(-0.25, 0.25),
                        "p": RangeParameter.scalar(0.3)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.2, 0.5),
                        "lightness": RangeParameter(0.5, 1.0),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(2.0, 5.0),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.4, 0.4),
                        "contrast_limit": RangeParameter(-0.4, 0.4),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.3, 0.7),
                        "lightness": RangeParameter(0.8, 1.5),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "moves_or_vibrates": AugmentationRule(
            augmentations=["MotionBlur", "SafeRotate"],
            reason="Motion creates blur and orientation changes",
            intensity_ranges={
                "low": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(1, 5),
                        "p": RangeParameter.scalar(0.15)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-180, 180),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(3, 7),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-180, 180),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(5, 9),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-180, 180),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "semi_transparent": AugmentationRule(
            augmentations=["RandomFog", "GaussNoise", "Blur"],
            reason="Semi-transparent objects create varying transparency",
            intensity_ranges={
                "low": {
                    "RandomFog": {
                        "alpha_corf": RangeParameter.scalar(0.06),
                        "fog_coef_range": RangeParameter(0.02, 0.05),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.001, 0.005),
                        "noise_scale_factor": RangeParameter(0.4, 0.6),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Blur": {
                        "blur_limit": RangeParameter(1, 3),
                        "p": RangeParameter.scalar(0.15)
                    }
                },
                "medium": {
                    "RandomFog": {
                        "alpha_coef": RangeParameter.scalar(0.08),
                        "fog_coef_range": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.005, 0.01),
                        "noise_scale_factor": RangeParameter(0.6, 0.8),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Blur": {
                        "blur_limit": RangeParameter(3, 5),
                        "p": RangeParameter.scalar(0.25)
                    }
                },
                "high": {
                    "RandomFog": {
                        "alpha_coef": RangeParameter.scalar(0.12),
                        "fog_coef_range": RangeParameter(0.1, 0.2),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.01, 0.02),
                        "noise_scale_factor": RangeParameter(0.8, 1.0),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Blur": {
                        "blur_limit": RangeParameter(3, 7),
                        "p": RangeParameter.scalar(0.35)
                    }
                }
            }
        ),

        "similar_to_background": AugmentationRule(
            augmentations=["CLAHE", "Sharpen", "RandomGamma"],
            reason="Enhance object-background separation",
            intensity_ranges={
                "low": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(0.8, 2.0),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.1, 0.25),
                        "lightness": RangeParameter(0.25, 0.5),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(90, 110),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(1.0, 4.0),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.2, 0.5),
                        "lightness": RangeParameter(0.5, 1.0),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(80, 120),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(2.0, 5.0),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.3, 0.7),
                        "lightness": RangeParameter(0.8, 1.5),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(70, 130),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "multiple_objects": AugmentationRule(
            augmentations=["RandomSizedBBoxSafeCrop"],
            reason="Handles scenes with multiple objects by focusing on regions",
            intensity_ranges={
                "low": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "medium": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.4)
                    }
                }
            }
        ),

        "partially_hidden": AugmentationRule(
            augmentations=["CoarseDropout"],
            reason="Simulates occlusion and partial hiding of objects",
            intensity_ranges={
                "low": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 1),
                        "hole_height_range": RangeParameter(0.05, 0.1),
                        "hole_width_range": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.2)
                    },
                },
                "medium": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 2), # Moderate occlusions
                        "hole_height_range": RangeParameter(0.1, 0.2),     # Medium holes
                        "hole_width_range": RangeParameter(0.1, 0.2),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 3),         # Many occlusions
                        "hole_height_range": RangeParameter(0.15, 0.25),       # Large holes
                        "hole_width_range": RangeParameter(0.15, 0.25),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        )
    }


    # Environment rules with ranges
    ENVIRONMENT_RULES = {
        "variable_lighting": AugmentationRule(
            augmentations=["RandomBrightnessContrast", "RandomGamma", "RandomShadow"],
            reason="Compensates for changing lighting conditions",
            intensity_ranges={
                "low": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.15, 0.15),
                        "contrast_limit": RangeParameter(-0.15, 0.15),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(90, 110),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(1, 2),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.25, 0.25),
                        "contrast_limit": RangeParameter(-0.25, 0.25),
                        "p": RangeParameter.scalar(0.3)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(80, 120),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(2, 3),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.4, 0.4),
                        "contrast_limit": RangeParameter(-0.4, 0.4),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(70, 130),
                        "p": RangeParameter.scalar(0.8)
                    },
                    "RandomShadow": {
                        "shadow_roi": (0, 0, 1, 1),
                        "num_shadows_limit": RangeParameter.integer_range(3, 4),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "fixed_camera": AugmentationRule(
            augmentations=["SafeRotate", "Affine"],
            reason="Even fixed camera has minor positioning variations and mounting imperfections",
            intensity_ranges={
                "low": {
                    "SafeRotate": {
                        "limit": RangeParameter(-1, 1),  # Very tiny rotation
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.98, 1.02),  # Minimal scale
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.02, 0.02),  # 2% translation
                            "y": RangeParameter(-0.02, 0.02)
                        }),
                        "rotate": RangeParameter(-2, 2),  # Tiny rotation
                        "shear": RangeParameter(-2, 2),  # Minimal shear
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "SafeRotate": {
                        "limit": RangeParameter(-2, 2),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.97, 1.03),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.03, 0.03),
                            "y": RangeParameter(-0.03, 0.03)
                        }),
                        "rotate": RangeParameter(-3, 3),
                        "shear": RangeParameter(-3, 3),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "SafeRotate": {
                        "limit": RangeParameter(-5, 5),  # Still mild for "fixed"
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.95, 1.05),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.05, 0.05),
                            "y": RangeParameter(-0.05, 0.05)
                        }),
                        "rotate": RangeParameter(-5, 5),
                        "shear": RangeParameter(-5, 5),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "shaky_camera": AugmentationRule(
            augmentations=["MotionBlur", "GaussNoise", "SafeRotate"],
            reason="Simulates camera vibration and instability",
            intensity_ranges={
                "low": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(1, 5),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.001, 0.005),
                        "noise_scale_factor": RangeParameter(0.4, 0.6),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-2, 2),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(3, 7),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.005, 0.01),
                        "noise_scale_factor": RangeParameter(0.6, 0.8),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-5, 5),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "MotionBlur": {
                        "blur_limit": RangeParameter(5, 9),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.01, 0.02),
                        "noise_scale_factor": RangeParameter(0.8, 1.0),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "SafeRotate": {
                        "limit": RangeParameter(-10, 10),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "moving_camera": AugmentationRule(
            augmentations=["Affine", "Perspective"],
            reason="Simulates camera movement and angle changes",
            intensity_ranges={
                "low": {
                    "Affine": {
                        "scale": RangeParameter(0.95, 1.05),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.05, 0.05),
                            "y": RangeParameter(-0.05, 0.05)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-15, 15),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.02, 0.05),
                        "p": RangeParameter.scalar(0.2)
                    },
                },
                "medium": {
                    "Affine": {
                        "scale": RangeParameter(0.9, 1.1),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.1, 0.1),
                            "y": RangeParameter(-0.1, 0.1)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-30, 30),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.3)
                    },
                },
                "high": {
                    "Affine": {
                        "scale": RangeParameter(0.85, 1.15),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.15, 0.15),
                            "y": RangeParameter(-0.15, 0.15)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-45, 45),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.1, 0.12),
                        "p": RangeParameter.scalar(0.6)
                    },
                }
            }
        ),

        "busy_background": AugmentationRule(
            augmentations=["CoarseDropout", "GaussNoise"],
            reason="Helps model focus on object despite background distractions",
            intensity_ranges={
                "low": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 1),
                        "hole_height_range": RangeParameter(0.05, 0.1),
                        "hole_width_range": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.001, 0.005),
                        "noise_scale_factor": RangeParameter(0.4, 0.6),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 2),
                        "hole_height_range": RangeParameter(0.1, 0.2),
                        "hole_width_range": RangeParameter(0.1, 0.2),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.005, 0.01),
                        "noise_scale_factor": RangeParameter(0.6, 0.8),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "CoarseDropout": {
                        "num_holes_range": RangeParameter.integer_range(1, 3),
                        "hole_height_range": RangeParameter(0.15, 0.25),
                        "hole_width_range": RangeParameter(0.15, 0.25),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "GaussNoise": {
                        "std_range": RangeParameter(0.01, 0.02),
                        "noise_scale_factor": RangeParameter(0.8, 1.0),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "clean_background": AugmentationRule(
            augmentations=["RandomBrightnessContrast", "ColorJitter", "RandomGamma"],
            reason="Clean background still has subtle lighting and color variations to simulate",
            intensity_ranges={
                "low": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.05, 0.05),  # Very subtle
                        "contrast_limit": RangeParameter(-0.05, 0.05),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "HueSaturationValue": {
                        "hue_shift_limit": RangeParameter(-5, 5),  # Minimal hue shift
                        "sat_shift_limit": RangeParameter(-10, 10),  # Subtle saturation
                        "val_shift_limit": RangeParameter(-5, 5),  # Minimal value
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(95, 105),  # Very tight range
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.08, 0.08),
                        "contrast_limit": RangeParameter(-0.08, 0.08),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "HueSaturationValue": {
                        "hue_shift_limit": RangeParameter(-8, 8),
                        "sat_shift_limit": RangeParameter(-15, 15),
                        "val_shift_limit": RangeParameter(-8, 8),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(92, 108),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.1, 0.1),  # Still conservative
                        "contrast_limit": RangeParameter(-0.1, 0.1),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "HueSaturationValue": {
                        "hue_shift_limit": RangeParameter(-10, 10),
                        "sat_shift_limit": RangeParameter(-20, 20),
                        "val_shift_limit": RangeParameter(-10, 10),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(90, 110),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "fixed_distance": AugmentationRule(
            augmentations=["RandomScale", "Affine", "Perspective"],
            reason="Even at fixed distance, slight scale variations and perspective changes can occur",
            intensity_ranges={
                "low": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.03, 0.03),  # 3% scale variation
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.98, 1.02),  # Minimal scale
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.02, 0.02),
                            "y": RangeParameter(-0.02, 0.02)
                        }),
                        "rotate": RangeParameter(-2, 2),
                        "shear": RangeParameter(-2, 2),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.01, 0.02),  # Very subtle perspective
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.05, 0.05),  # 5% variation
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.97, 1.03),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.03, 0.03),
                            "y": RangeParameter(-0.03, 0.03)
                        }),
                        "rotate": RangeParameter(-3, 3),
                        "shear": RangeParameter(-3, 3),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.02, 0.03),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.08, 0.08),  # Still modest
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.95, 1.05),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.05, 0.05),
                            "y": RangeParameter(-0.05, 0.05)
                        }),
                        "rotate": RangeParameter(-5, 5),
                        "shear": RangeParameter(-5, 5),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.03, 0.05),
                        "p": RangeParameter.scalar(0.26)
                    }
                }
            }
        ),

        "changing_background": AugmentationRule(
            augmentations=["RandomBrightnessContrast", "HueSaturationValue", "RandomGamma"],
            reason="Adapts to background variations that affect object appearance",
            intensity_ranges={
                "low": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.15, 0.15),
                        "contrast_limit": RangeParameter(-0.15, 0.15),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "HueSaturationValue": {
                        "hue_shift_limit": RangeParameter(-10, 10),
                        "sat_shift_limit": RangeParameter(-20, 20),
                        "val_shift_limit": RangeParameter(-10, 10),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(90, 110),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.25, 0.25),
                        "contrast_limit": RangeParameter(-0.25, 0.25),
                        "p": RangeParameter.scalar(0.3)
                    },
                    "HueSaturationValue": {
                        "hue_shift_limit": RangeParameter(-20, 20),
                        "sat_shift_limit": RangeParameter(-30, 30),
                        "val_shift_limit": RangeParameter(-20, 20),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(80, 120),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.4, 0.4),
                        "contrast_limit": RangeParameter(-0.4, 0.4),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "HueSaturationValue": {
                    "hue_shift_limit": RangeParameter(-20, 20),
                        "sat_shift_limit": RangeParameter(-30, 30),
                        "val_shift_limit": RangeParameter(-20, 20),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(70, 130),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "variable_distance": AugmentationRule(
            augmentations=["RandomScale", "Perspective", "Affine"],
            reason="Simulates objects at different distances from camera",
            intensity_ranges={
                "low": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.1, 0.1),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.02, 0.05),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.95, 1.05),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.05, 0.05),
                            "y": RangeParameter(-0.05, 0.05)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-15, 15),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.2, 0.2),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.3)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.9, 1.1),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.1, 0.1),
                            "y": RangeParameter(-0.1, 0.1)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-30, 30),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomScale": {
                        "scale_limit": RangeParameter(-0.4, 0.4),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.1, 0.12),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Affine": {
                        "scale": RangeParameter(0.85, 1.15),
                        "translate_percent": NestedParameter({
                            "x": RangeParameter(-0.15, 0.15),
                            "y": RangeParameter(-0.15, 0.15)
                        }),
                        "rotate": RangeParameter(-360, 360),
                        "shear": RangeParameter(-45, 45),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "close_distance": AugmentationRule(
            augmentations=["RandomSizedBBoxSafeCrop", "Perspective"],
            reason="Handles close-up views where object fills most of the frame",
            intensity_ranges={
                "low": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.02, 0.05),
                        "p": RangeParameter.scalar(0.2)
                    }
                },
                "medium": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.05, 0.1),
                        "p": RangeParameter.scalar(0.3)
                    }
                },
                "high": {
                    "RandomSizedBBoxSafeCrop": {
                        "height": RangeParameter.scalar(1024),
                        "width": RangeParameter.scalar(1024),
                        "p": RangeParameter.scalar(0.8)
                    },
                    "Perspective": {
                        "scale": RangeParameter(0.1, 0.12),
                        "p": RangeParameter.scalar(0.6)
                    }
                }
            }
        ),

        "poor_lighting": AugmentationRule(
            augmentations=["CLAHE", "RandomGamma", "Sharpen", "RandomBrightnessContrast"],
            reason="Compensates for poor lighting conditions",
            intensity_ranges={
                "low": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(0.8, 2.0),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(90, 110),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.1, 0.25),
                        "lightness": RangeParameter(0.25, 0.5),
                        "p": RangeParameter.scalar(0.2)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.1, 0.2),
                        "contrast_limit": RangeParameter(-0.1, 0.2),
                        "p": RangeParameter.scalar(0.6)
                    }
                },
                "medium": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(1.0, 4.0),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(80, 120),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.2, 0.5),
                        "lightness": RangeParameter(0.5, 1.0),
                        "p": RangeParameter.scalar(0.4)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.15, 0.3),
                        "contrast_limit": RangeParameter(-0.15, 0.3),
                        "p": RangeParameter.scalar(0.7)
                    }
                },
                "high": {
                    "CLAHE": {
                        "clip_limit": RangeParameter(2.0, 5.0),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomGamma": {
                        "gamma_limit": RangeParameter(70, 130),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "Sharpen": {
                        "alpha": RangeParameter(0.3, 0.7),
                        "lightness": RangeParameter(0.8, 1.5),
                        "p": RangeParameter.scalar(0.6)
                    },
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.2, 0.4),
                        "contrast_limit": RangeParameter(-0.2, 0.4),
                        "p": RangeParameter.scalar(0.8)
                    }
                }
            }
        ),

        "stable_lighting": AugmentationRule(
            augmentations=["RandomBrightnessContrast"],
            reason="Minimal lighting variation for stable conditions",
            intensity_ranges={
                "low": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.05, 0.05),
                        "contrast_limit": RangeParameter(-0.05, 0.05),
                        "p": RangeParameter.scalar(0.3)
                    }
                },
                "medium": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.1, 0.1),
                        "contrast_limit": RangeParameter(-0.1, 0.1),
                        "p": RangeParameter.scalar(0.4)
                    }
                },
                "high": {
                    "RandomBrightnessContrast": {
                        "brightness_limit": RangeParameter(-0.15, 0.15),
                        "contrast_limit": RangeParameter(-0.15, 0.15),
                        "p": RangeParameter.scalar(0.5)
                    }
                }
            }
        )
    }

    def _validate_intensity(self, intensity: str) -> None:
        """
        Validate intensity parameter
        
        Args:
            intensity: Intensity level to validate
            
        Raises:
            ValueError: If intensity is not valid
        """
        if intensity not in self.VALID_INTENSITIES:
            raise ValueError(
                f"Invalid intensity '{intensity}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_INTENSITIES))}"
            )

    def _get_augmentation_config_unified(
        self, aug_type: str, unified_params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Generate configuration using unified parameter system"""
        return {
            "type": aug_type,
            "unified_params": unified_params,
            "reason": f"Generated for {aug_type}"
        }

    def _keep_higher_p(
        self, existing: Dict[str, Any], new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Keep parameters with higher probability value

        Args:
            existing: Existing parameter dict
            new: New parameter dict

        Returns:
            Parameter dict with higher probability
        """
        existing_p = existing["p"]
        new_p = new["p"]

        existing_prob = existing_p.min_val if hasattr(existing_p, "min_val") else existing_p
        new_prob = new_p.min_val if hasattr(new_p, "min_val") else new_p

        if new_prob > existing_prob:
            logger.debug(
                "Deduplication: Kept new with higher p=%.2f (vs %.2f)",
                new_prob, existing_prob
            )
            return new
        logger.debug(
            "Deduplication: Kept existing with higher p=%.2f (vs %.2f)",
            existing_prob, new_prob
        )
        return existing

    def translate_from_characteristics(
        self, characteristics: List[str],
        environment: Optional[Dict[str, str]] = None,
        intensity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Translate characteristics to augmentation configurations
        Returns dict-based config with clear separation of transforms and metadata
        
        Note: Empty characteristics and environment will create identity pipeline (no-op)

        Args:
            characteristics: List of object characteristics (can be empty)
            environment: Environment conditions (optional)
            intensity: Augmentation intensity level

        Returns:
            Configuration dictionary with structure:
            {
                "augmentations": {aug_type: params, ...},  # May be empty dict
                "characteristics": [...],
                "environment": {...},
                "intensity": "...",
                "metadata": {...}
            }
            
        Raises:
            ValueError: If characteristics or environment values are invalid
        """
        # Validate inputs early (fail fast)
        self._validate_intensity(intensity)

        # Allow empty inputs - will create identity pipeline (no-op)
        # This is useful for dynamic configurations or GUI scenarios
        merged_augmentations = {}
        applied_rules = []

        # Add characteristic-specific transforms
        for characteristic in characteristics:
            if characteristic in self.CHARACTERISTIC_RULES:
                rule = self.CHARACTERISTIC_RULES[characteristic] # changes_shape level
                params_dict = rule.intensity_ranges[intensity]
                # params_dict =
                # {
                #     "ElasticTransform": {
                #         "alpha": RangeParameter(0.5, 1.5),
                #         "sigma": RangeParameter(30, 70),
                #         "p": RangeParameter.scalar(0.4)
                #     },
                #     "PiecewiseAffine": {
                #         "scale": RangeParameter(0.02, 0.05),
                #         "p": RangeParameter.scalar(0.4)
                #     }
                # }
                applied_rules.append({
                    "type": "characteristic",
                    "name": characteristic,
                    "reason": rule.reason
                })

                # Merge augmentations inline
                for aug_type, params in params_dict.items():
                    if aug_type in merged_augmentations:
                        merged_augmentations[aug_type] = self._keep_higher_p(
                            merged_augmentations[aug_type], params
                        )
                    else:
                        merged_augmentations[aug_type] = params
            else:
                # Fail fast on unknown characteristic
                available = list(self.CHARACTERISTIC_RULES.keys())
                raise ValueError(
                    f"Unknown characteristic: '{characteristic}'. "
                    f"Available characteristics: {', '.join(sorted(available))}"
                )

        # Add environment-specific transforms
        if environment:
            for env_key, env_value in environment.items():
                env_rule_key = f"{env_value}_{env_key}"   # e.g., "variable_lighting"
                if env_rule_key in self.ENVIRONMENT_RULES:
                    rule = self.ENVIRONMENT_RULES[env_rule_key]
                    params_dict = rule.intensity_ranges[intensity]
                    applied_rules.append({
                        "type": "environment",
                        "name": env_rule_key,
                        "reason": rule.reason
                    })
                    # Merge augmentations inline
                    for aug_type, params in params_dict.items():
                        if aug_type in merged_augmentations:
                            merged_augmentations[aug_type] = self._keep_higher_p(
                                merged_augmentations[aug_type], params
                            )
                        else:
                            merged_augmentations[aug_type] = params
                else:
                    # Fail fast on unknown environment
                    available_envs = self.get_available_environments()
                    raise ValueError(
                        f"Unknown environment condition: '{env_rule_key}' "
                        f"(from {env_key}='{env_value}'). "
                        f"Available: {available_envs}"
                    )

        # Deduplicate augmentations
        logger.info(
            "Generated config with %d unique augmentations from %d rules",
            len(merged_augmentations), len(applied_rules)
        )

        return {
            "augmentations": merged_augmentations,
            "characteristics": characteristics,
            "environment": environment or {},
            "intensity": intensity,
            "metadata": {
                "applied_rules": applied_rules,
                "description": f"Auto-generated for {', '.join(characteristics)} objects"
            }
        }

    def get_available_characteristics(self) -> List[str]:
        """Get list of all available characteristics"""
        return list(self.CHARACTERISTIC_RULES.keys())

    def get_available_environments(self) -> Dict[str, List[str]]:
        """Get available environment options"""
        return {
            "lighting": ["stable", "variable", "poor"],
            "camera": ["fixed", "moving", "shaky"],
            "background": ["clean", "busy", "changing"],
            "distance": ["fixed", "variable", "close"]
        }

    def validate_characteristics(self, characteristics: List[str]) -> Dict[str, Any]:
        """
        Validate that characteristics are supported
        
        Note: Empty list is valid - will result in no characteristic-based augmentations

        Args:
            characteristics: List of characteristics to validate

        Returns:
            Validation result with supported/unsupported characteristics
        """
        # Empty list is valid - will create identity pipeline or rely on environment
        if not characteristics:
            return {
                "valid": True,
                "supported_characteristics": [],
                "unsupported_characteristics": [],
                "available_characteristics": list(self.CHARACTERISTIC_RULES.keys())
            }

        available = set(self.CHARACTERISTIC_RULES.keys())
        provided = set(characteristics)

        supported = provided.intersection(available)
        unsupported = provided.difference(available)

        return {
            "valid": len(unsupported) == 0,
            "supported_characteristics": list(supported),
            "unsupported_characteristics": list(unsupported),
            "available_characteristics": list(available)
        }

    def validate_environment(self, environment: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate environment conditions

        Args:
            environment: Environment conditions to validate

        Returns:
            Validation result
        """
        available_envs = self.get_available_environments()
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        for env_key, env_value in environment.items():
            if env_key not in available_envs:
                validation_result["errors"].append(f"Unknown environment key: {env_key}")
                validation_result["valid"] = False
            elif env_value not in available_envs[env_key]:
                validation_result["errors"].append(f"Unknown value '{env_value}' for environment '{env_key}'")
                validation_result["valid"] = False

        return validation_result
