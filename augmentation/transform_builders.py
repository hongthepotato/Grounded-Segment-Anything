"""
Create transforms from unified parameters
"""

import logging
import re
import difflib
from typing import Dict, Any, List, Union
from .parameter_system import AlbumentationsParameter

logger = logging.getLogger(__name__)

class TransformParameterBuilder:
    """Builds parameters for specific albumentations transforms"""

    def __init__(self):
        """Initialize the builder"""
        logger.debug("TransformParameterBuilder initialized")

    # ============================================
    # VALIDATION HELPER
    # ============================================

    def _validate_params(
        self,
        params: Dict[str, AlbumentationsParameter],
        required: List[str],
        transform_name: str
    ) -> None:
        """
        Validate that required parameters are present and correct type
        
        Args:
            params: Parameter dictionary to validate
            required: List of required parameter names
            transform_name: Name of transform (for error messages)
        
        Raises:
            ValueError: If required parameters are missing
            TypeError: If parameters are wrong type
        """
        # Check for missing parameters
        missing = [p for p in required if p not in params]
        if missing:
            # Try to detect typos by finding similar keys
            suggestions = []
            provided_keys = list(params.keys())
            for m in missing:
                close_matches = difflib.get_close_matches(m, provided_keys, n=1, cutoff=0.6)
                if close_matches:
                    suggestions.append(f"{close_matches[0]} is not valid. Did you mean: {m}?)")
                else:
                    suggestions.append(m)

            raise ValueError(
                f"{transform_name} missing required parameters: {suggestions}. "
                f"Provided: {provided_keys}"
            )

        # Check parameter types
        for param_name in required:
            if not isinstance(params[param_name], AlbumentationsParameter):
                raise TypeError(
                    f"{transform_name} parameter '{param_name}' must be AlbumentationsParameter, "
                    f"got {type(params[param_name]).__name__}"
                )

    # ============================================
    # GEOMETRIC TRANSFORMS
    # ============================================

    def build_elastic_transform_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Union[float, int]]:
        """Build parameters for ElasticTransform
        
        Args:
            params: Parameters for the ElasticTransform (alpha, sigma, p)

        Returns:
            Parameters for the ElasticTransform (alpha, sigma, p)
        """
        self._validate_params(
            params, ["alpha", "sigma", "p"],
            "ElasticTransform"
        )

        return {
            "alpha": params["alpha"].sample(),
            "sigma": params["sigma"].sample(),
            "p": params["p"].sample()
        }

    def build_piecewise_affine_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for PiecewiseAffine
        
        Args:
            params: Parameters for the PiecewiseAffine (scale, p)

        Returns:
            Parameters for the PiecewiseAffine (scale, p)
        """
        self._validate_params(params, ["scale", "p"], "PiecewiseAffine")

        return {
            "scale": params["scale"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_random_scale_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomScale
        
        Args:
            params: Parameters for the RandomScale (scale_limit, p)
        """
        self._validate_params(params, ["scale_limit", "p"], "RandomScale")
        return {
            "scale_limit": params["scale_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_affine_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for Affine
        
        Args:
            params: Parameters for the Affine (translate_percent, scale, rotate, shear, p)

        Returns:
            Parameters for the Affine (translate_percent, scale, rotate, shear, p)
        """
        self._validate_params(
            params, ["translate_percent", "scale", "rotate", "shear", "p"],
            "Affine"
        )

        return {
            "translate_percent": params["translate_percent"].to_albumentations_format(),
            "scale": params["scale"].to_albumentations_format(),
            "rotate": params["rotate"].to_albumentations_format(),
            "shear": params["shear"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_perspective_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for Perspective
        
        Args:
            params: Parameters for the Perspective (perspective_scale, perspective_p)

        Returns:
            Parameters for the Perspective (scale, p)
        """
        self._validate_params(params, ["scale", "p"], "Perspective")

        return {
            "scale": params["scale"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_safe_rotation_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for Rotate
        
        Args:
            params: Parameters for the Rotation (limit, p)

        Returns:
            Parameters for the Rotate (limit, p)
        """
        self._validate_params(params, ["limit", "p"], "SafeRotation")

        return {
            "limit": params["limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    # ============================================
    # COLOR AND BRIGHTNESS TRANSFORMS
    # ============================================

    def build_random_brightness_contrast_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomBrightnessContrast

        Args:
            params: Parameters for the RandomBrightnessContrast
                    (brightness_limit, contrast_limit, p)

        Returns:
            Parameters for the RandomBrightnessContrast (brightness_limit, contrast_limit, p)
        """
        self._validate_params(
            params,
            ["brightness_limit", "contrast_limit", "p"],
            "RandomBrightnessContrast"
        )

        return {
            "brightness_limit": params["brightness_limit"].to_albumentations_format(),
            "contrast_limit": params["contrast_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_color_jitter_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for ColorJitter
        
        Args:
            params: Parameters for the ColorJitter (brightness, contrast, saturation, hue, p)
        """
        self._validate_params(
            params,
            ["brightness", "contrast", "saturation", "hue", "p"],
            "ColorJitter"
        )

        return {
            "brightness": params["brightness"].to_albumentations_format(),
            "contrast": params["contrast"].to_albumentations_format(),
            "saturation": params["saturation"].to_albumentations_format(),
            "hue": params["hue"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_random_sized_b_box_safe_crop_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomSizedBBoxSafeCrop
        
        Args:
            params: Parameters for the RandomSizedBBoxSafeCrop (height, width, p)
        """
        self._validate_params(params, ["height", "width", "p"], "RandomSizedBBoxSafeCrop")
        return {
            "height": params["height"].to_albumentations_format(),
            "width": params["width"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_random_gamma_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomGamma
        
        Args:
            params: Parameters for the RandomGamma (gamma_limit, p)

        Returns:
            Parameters for the RandomGamma (gamma_limit, p)
        """
        self._validate_params(params, ["gamma_limit", "p"], "RandomGamma")

        return {
            "gamma_limit": params["gamma_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_hue_saturation_value_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for HueSaturationValue
        
        Args:
            params: Parameters for the HueSaturationValue
                    (hue_shift_limit, sat_shift_limit, val_shift_limit, p)

        Returns:
            Parameters for the HueSaturationValue 
                    (hue_shift_limit, sat_shift_limit, val_shift_limit, p)
        """
        self._validate_params(
            params,
            ["hue_shift_limit", "sat_shift_limit", "val_shift_limit", "p"],
            "HueSaturationValue"
        )

        return {
            "hue_shift_limit": params["hue_shift_limit"].to_albumentations_format(),
            "sat_shift_limit": params["sat_shift_limit"].to_albumentations_format(),
            "val_shift_limit": params["val_shift_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_clahe_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for CLAHE
        
        Args:
            params: Parameters for the CLAHE (clip_limit, p)

        Returns:
            Parameters for the CLAHE (clip_limit, tile_grid_size, p)
        """
        self._validate_params(params, ["clip_limit", "p"], "CLAHE")

        return {
            "clip_limit": params["clip_limit"].to_albumentations_format(),
            "tile_grid_size": (8, 8),  # Fixed tile size
            "p": params["p"].sample()
        }

    def build_sharpen_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for Sharpen
        
        Args:
            params: Parameters for the Sharpen (alpha lightness, p)

        Returns:
            Parameters for the Sharpen (alpha, lightness, p)
        """
        # Validate sharpen_p is required, alpha parameters are optional
        self._validate_params(params, ["p", "alpha", "lightness"], "Sharpen")
        return {
            "alpha": params["alpha"].to_albumentations_format(),  # Use the computed alpha variable
            "lightness": params["lightness"].to_albumentations_format(),  # Fixed lightness range
            "p": params["p"].sample()
        }

    # ============================================
    # NOISE AND BLUR TRANSFORMS
    # ============================================

    def build_gauss_noise_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for GaussNoise
        
        Args:
            params: Parameters for the GaussNoise (std_range, noise_scale_factor, p)

        Returns:
            Parameters for the GaussNoise (std_range, scale_factor, p) - v2.0.11+ uses std_range
        """
        self._validate_params(params, ["std_range", "noise_scale_factor", "p"], "GaussNoise")

        return {
            "std_range": params["std_range"].to_albumentations_format(),
            "noise_scale_factor": params["noise_scale_factor"].sample(),
            "p": params["p"].sample()
        }

    def build_motion_blur_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for MotionBlur
        
        Args:
            params: Parameters for the MotionBlur (blur_limit, p)

        Returns:
            Parameters for the MotionBlur (blur_limit, p)
        """
        self._validate_params(params, ["blur_limit", "p"], "MotionBlur")

        return {
            "blur_limit": params["blur_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    def build_blur_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for Blur
        
        Args:
            params: Parameters for the Blur (blur_limit, p)

        Returns:
            Parameters for the Blur (blur_limit, p)
        """
        self._validate_params(params, ["blur_limit", "p"], "Blur")

        return {
            "blur_limit": params["blur_limit"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    # ============================================
    # WEATHER AND ENVIRONMENT TRANSFORMS
    # ============================================

    def build_random_fog_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomFog
        
        Args:
            params: Parameters for the RandomFog (fog_coef_range, alpha_coef, p)
        
        Returns:
            Parameters for the RandomFog (fog_coef_range, alpha_coef, p)
        """
        self._validate_params(params, ["fog_coef_range", "alpha_coef", "p"], "RandomFog")

        return {
            "fog_coef_range": params["fog_coef_range"].to_albumentations_format(),
            "alpha_coef": params["alpha_coef"].sample(),  # Fixed alpha coefficient
            "p": params["p"].sample()
        }

    def build_random_shadow_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomShadow
        
        Args:
            params: Parameters for the RandomShadow (p, num_shadows_limit)
        
        Returns:
            Parameters for the RandomShadow (shadow_roi, shadow_dimension, p)
        """
        self._validate_params(params, ["p", "num_shadows_limit"], "RandomShadow")

        return {
            "shadow_roi": (0, 0, 1, 1),  # Fixed ROI
            "num_shadows_limit": params["num_shadows_limit"].to_albumentations_format(),
            "shadow_dimension": 5,  # Fixed dimension
            "p": params["p"].sample()
        }

    def build_random_sun_flare_params(
        self, params: Dict[str, AlbumentationsParameter]
    ) -> Dict[str, Any]:
        """Build parameters for RandomSunFlare
        
        Args:
            params: Parameters for the RandomSunFlare (p, src_radius)
        
        Returns:
            Parameters for the RandomSunFlare (flare_roi, src_radius, p)
        """
        self._validate_params(params, ["p", "src_radius"], "RandomSunFlare")

        return {
            "flare_roi": (0, 0, 1, 1),  # Fixed ROI
            "src_radius": params["src_radius"].sample(),
            "p": params["p"].sample()
        }

    # ============================================
    # OCCLUSION TRANSFORMS
    # ============================================

    def build_coarse_dropout_params(self, params: Dict[str, AlbumentationsParameter]) -> Dict[str, Any]:
        """
        Build parameters for CoarseDropout
        Required params: num_holes_range (integer range), hole_height_range, hole_width_range, p
        """
        self._validate_params(
            params,
            ["num_holes_range", "hole_height_range", "hole_width_range", "p"],
            "CoarseDropout"
        )

        return {
            "num_holes_range": params["num_holes_range"].to_albumentations_format(),
            "hole_height_range": params["hole_height_range"].to_albumentations_format(),
            "hole_width_range": params["hole_width_range"].to_albumentations_format(),
            "p": params["p"].sample()
        }

    # def build_random_erasing_params(self, params: Dict[str, AlbumentationsParameter]) -> Dict[str, Any]:
    #     """
    #     Build parameters for RandomErasing
    #     Required params: erasing_scale, erasing_p
    #     """
    #     self._validate_params(params, ["erasing_scale", "erasing_p"], "RandomErasing")

    #     return {
    #         "scale": params["erasing_scale"].to_albumentations_format(),
    #         "ratio": (0.3, 3.3),  # Fixed aspect ratio range
    #         "p": params["erasing_p"].sample()
    #     }

    # ============================================
    # GENERIC FALLBACK
    # ============================================

    def build_generic_params(self, params: Dict[str, AlbumentationsParameter]) -> Dict[str, Any]:
        """
        Generic parameter builder for transforms without specific builders
        Converts all AlbumentationsParameter objects to their albumentations format
        Handles probability parameters specially
        """
        processed = {}
        for key, param in params.items():
            if key.endswith('_p'):
                # Probability parameter - sample it
                processed_key = 'p'
                processed[processed_key] = param.sample()
            else:
                # Regular parameter - convert to albumentations format
                processed[key] = param.to_albumentations_format()

        logger.debug("Generic parameter processing: %s", list(processed.keys()))
        return processed

    def get_builder_method(self, transform_type: str):
        """
        Get the appropriate builder method for a transform type
        
        Args:
            transform_type: Albumentations transform name (e.g., "ElasticTransform")
        
        Returns:
            Builder method or None if not found
        """
        # Convert CamelCase to snake_case
        method_name = f"build_{self._snake_case(transform_type)}_params"

        if hasattr(self, method_name):
            return getattr(self, method_name)
        logger.warning("No specific builder for %s, using generic", transform_type)
        return self.build_generic_params

    def _snake_case(self, camel_case: str) -> str:
        """Convert CamelCase to snake_case, handling acronyms correctly."""
        # Add an underscore before a capital letter if it's preceded by a lowercase letter or digit.
        # e.g., "AbcDef" -> "Abc_Def", "myVar" -> "my_Var"
        s1 = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', camel_case)
        # Add an underscore before a capital letter if it's preceded by another capital
        # and followed by a lowercase letter. This handles acronyms.
        # e.g., "HTTPRequest" -> "HTTP_Request"
        s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
        return s2.lower()
    