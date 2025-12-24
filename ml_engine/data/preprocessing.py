"""
Multi-model preprocessing pipeline using OFFICIAL model implementations.

Architecture:
- Base class defines interface for model-specific preprocessors
- Each model uses its OFFICIAL preprocessing utilities (no reinventing!)
- Easy to extend: just add new model preprocessor class
- Maintainable: bugs fixed upstream by model authors

Models:
- SAM: Uses official ResizeLongestSide from segment_anything
- Grounding DINO: Uses official resize + Normalize from groundingdino
- YOLO: Uses official LetterBox from ultralytics
"""

from typing import Dict, Tuple, List, Optional, Any
from abc import ABC, abstractmethod
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from core.config import load_config
from groundingdino.datasets.transforms import resize
from groundingdino.datasets import transforms as T


class BaseModelPreprocessor(ABC):
    """
    Abstract base class for model-specific preprocessors.
    """
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Args:
            model_name: Name of the model (for logging/debugging)
            config: Model-specific configuration from preprocessing.yaml
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def preprocess(
        self,
        image: Image.Image,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess image and annotations.
        
        Args:
            image: PIL Image (RGB)
            boxes: Optional boxes in COCO format [x, y, w, h] (N, 4)
            masks: Optional masks (N, H, W)
        
        Returns:
            Tuple of (preprocessed_image_tensor, metadata_dict)
            
            metadata_dict must contain:
                - 'original_size': (height, width) of input image
                - 'final_size': (height, width) of output tensor
                - 'model_name': str
                - Any model-specific transformation info
        """
        pass

    @abstractmethod
    def transform_boxes(
        self,
        boxes: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Transform boxes from original image space to preprocessed image space.
        
        Args:
            boxes: Boxes in COCO format [x, y, w, h] (N, 4) in original space
            metadata: Metadata from preprocess() call
        
        Returns:
            Transformed boxes (N, 4) in preprocessed space
        """
        pass

    @abstractmethod
    def transform_masks(
        self,
        masks: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Transform masks from original image space to preprocessed image space.
        
        Args:
            masks: Masks (N, H, W) in original space
            metadata: Metadata from preprocess() call
        
        Returns:
            Transformed masks (N, H', W') in preprocessed space
        """
        pass


class SAMPreprocessor(BaseModelPreprocessor):
    """
    Preprocessor for SAM (Segment Anything Model).
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        from segment_anything.utils.transforms import ResizeLongestSide
        self.sam_transformer = ResizeLongestSide(
            target_length=config['input_size']['height']
        )

        # Normalization parameters
        norm_cfg = config['normalization']
        self.mean = torch.tensor(norm_cfg['mean']).view(3, 1, 1)
        self.std = torch.tensor(norm_cfg['std']).view(3, 1, 1)
        self.pad_value = config.get('padding_value', 0)

    def preprocess(
        self,
        image: Image.Image,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Preprocess using SAM's official ResizeLongestSide."""
        orig_width, orig_height = image.size

        # Use SAM's official resize (numpy input expected)
        image_np = np.array(image)
        resized_np = self.sam_transformer.apply_image(image_np)

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - self.mean) / (self.std + 1e-8)

        # Pad to square (SAM expects square input)
        image_tensor = self._pad_to_square(image_tensor)

        metadata = {
            'original_size': (orig_height, orig_width),   # (H, W)
            'final_size': tuple(image_tensor.shape[-2:]),  # (H, W)
            'model_name': self.model_name,
            'sam_transformer': self.sam_transformer  # Store for coordinate transforms!
        }

        return image_tensor, metadata

    def _pad_to_square(self, image: torch.Tensor) -> torch.Tensor:
        """
        Pad image to square (SAM requirement).
        
        Uses torchvision.transforms.functional.pad which is specifically
        designed for image padding operations.
        
        Args:
            image: Tensor of shape (C, H, W)
        
        Returns:
            Padded tensor of shape (C, target_size, target_size)
        """
        _, h, w = image.shape
        target_size = self.config['input_size']['height']

        pad_h = target_size - h
        pad_w = target_size - w

        # Padding format: (left, top, right, bottom)
        return TF.pad(
            image,
            padding=[0, 0, pad_w, pad_h],  # Pad right and bottom
            fill=self.pad_value,
            padding_mode='constant'
        )

    def transform_boxes(
        self,
        boxes: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Use SAM's official apply_boxes method.
        
        Returns boxes in xyxy format as expected by SAM's prompt encoder.
        """
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Convert COCO [x, y, w, h] to xyxy [x1, y1, x2, y2]
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h

        sam_transformer = metadata['sam_transformer']
        orig_h, orig_w = metadata['original_size']  # (H, W)
        transformed_xyxy = sam_transformer.apply_boxes(
            boxes_xyxy,
            original_size=(orig_h, orig_w)
        )

        return transformed_xyxy.astype(np.float32)

    def transform_masks(
        self,
        masks: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Transform masks using resize + padding."""
        import cv2

        if len(masks) == 0:
            h, w = metadata['final_size']
            return np.zeros((0, h, w), dtype=np.uint8)

        # Get transformation parameters from SAM
        sam_transformer = metadata['sam_transformer']
        orig_h, orig_w = metadata['original_size']  # (H, W)

        # Calculate target size after resize (before padding)
        scale = sam_transformer.target_length / max(orig_h, orig_w)
        new_h = int(orig_h * scale + 0.5)
        new_w = int(orig_w * scale + 0.5)

        # Transform each mask
        transformed_masks = []
        for mask in masks:
            # Resize mask
            resized = cv2.resize(
                mask.astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Pad to final size
            final_h, final_w = metadata['final_size']
            padded = np.zeros((final_h, final_w), dtype=np.uint8)
            padded[:new_h, :new_w] = resized

            transformed_masks.append(padded)

        return np.stack(transformed_masks, axis=0)


class GroundingDINOPreprocessor(BaseModelPreprocessor):
    """
    Preprocessor for Grounding DINO.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        self.dino_totensor = T.ToTensor()
        self.dino_normalize = T.Normalize(
            mean=config['normalization']['mean'],
            std=config['normalization']['std']
        )

        self.min_size = config['input_size']['min_size']
        self.max_size = config['input_size']['max_size']

    def preprocess(
        self,
        image: Image.Image,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Preprocess using Grounding DINO's official transforms."""

        orig_width, orig_height = image.size

        # Prepare target dict (DINO's format)
        target = {}
        if boxes is not None and len(boxes) > 0:
            # Convert COCO [x, y, w, h] to xyxy [x1, y1, x2, y2]
            boxes_xyxy = torch.from_numpy(boxes.copy()).float()
            boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
            boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]
            target['boxes'] = boxes_xyxy

        if masks is not None and len(masks) > 0:
            target['masks'] = torch.from_numpy(masks)

        image, target = resize(image, target, self.min_size, self.max_size)
        image, target = self.dino_totensor(image, target)
        image, target = self.dino_normalize(image, target)

        transformed_boxes = None
        transformed_masks = None

        if 'boxes' in target:
            boxes_normalized = target['boxes'].numpy()  # [N, 4] in [cx, cy, w, h], range [0, 1]
            transformed_boxes = boxes_normalized

        if 'masks' in target:
            transformed_masks = target['masks'].numpy()

        metadata = {
            'original_size': (orig_height, orig_width),  # (H, W)
            'final_size': tuple(image.shape[-2:]),       # (H, W)
            'model_name': self.model_name,
            'transformed_boxes': transformed_boxes,
            'transformed_masks': transformed_masks,      # (N, H, W)
        }

        return image, metadata

    def transform_boxes(
        self,
        boxes: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Boxes already transformed by DINO's official transforms.
        Just return from metadata.
        """
        if 'transformed_boxes' in metadata and metadata['transformed_boxes'] is not None:
            return metadata['transformed_boxes'].astype(np.float32)
        return np.zeros((0, 4), dtype=np.float32)

    def transform_masks(
        self,
        masks: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Masks already transformed by DINO's official transforms.
        Just return from metadata.
        """
        if 'transformed_masks' in metadata and metadata['transformed_masks'] is not None:
            return metadata['transformed_masks'].astype(np.uint8)

        h, w = metadata['final_size']
        return np.zeros((0, h, w), dtype=np.uint8)


class YOLOPreprocessor(BaseModelPreprocessor):
    """
    Preprocessor for YOLO (Ultralytics).
    
    Uses OFFICIAL implementation: ultralytics.data.augment.LetterBox
    
    Source: https://github.com/ultralytics/ultralytics
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        
        # Import YOLO's official LetterBox
        try:
            from ultralytics.data.augment import LetterBox
            target_size = config['input_size'].get('size', 640)
            self.letterbox = LetterBox(
                new_shape=(target_size, target_size),
                auto=False,  # Don't auto-adjust stride
                scaleFill=False,  # Maintain aspect ratio
                scaleup=True,
            )
        except ImportError:
            raise ImportError(
                "Ultralytics YOLO not installed! Run: pip install ultralytics"
            )
        
        # Normalization parameters
        norm_cfg = config['normalization']
        self.mean = torch.tensor(norm_cfg['mean']).view(3, 1, 1)
        self.std = torch.tensor(norm_cfg['std']).view(3, 1, 1)
        self.target_size = config['input_size'].get('size', 640)
    
    def preprocess(
        self, 
        image: Image.Image,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Preprocess using YOLO's official LetterBox."""
        orig_width, orig_height = image.size
        image_np = np.array(image)
        
        # Use YOLO's official LetterBox!
        transformed = self.letterbox(image=image_np)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - self.mean) / (self.std + 1e-8)
        
        # Calculate scale and padding (LetterBox doesn't expose this, so we compute)
        scale = self.target_size / max(orig_height, orig_width)
        new_h = int(orig_height * scale)
        new_w = int(orig_width * scale)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        metadata = {
            'original_size': (orig_height, orig_width),     # (H, W)
            'final_size': tuple(image_tensor.shape[-2:]),   # (H, W)
            'model_name': self.model_name,
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
        }
        
        return image_tensor, metadata
    
    def transform_boxes(
        self, 
        boxes: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Transform boxes using scale + padding."""
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        scale = metadata['scale']
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']
        
        transformed = boxes.copy().astype(np.float32)
        transformed[:, 0] = boxes[:, 0] * scale + pad_left  # x
        transformed[:, 1] = boxes[:, 1] * scale + pad_top   # y
        transformed[:, 2] = boxes[:, 2] * scale              # w
        transformed[:, 3] = boxes[:, 3] * scale              # h
        
        return transformed
    
    def transform_masks(
        self, 
        masks: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Transform masks using scale + padding."""
        import cv2
        
        if len(masks) == 0:
            h, w = metadata['final_size']
            return np.zeros((0, h, w), dtype=np.uint8)
        
        scale = metadata['scale']
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']
        final_h, final_w = metadata['final_size']
        
        transformed_masks = []
        for mask in masks:
            orig_h, orig_w = mask.shape
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            
            # Resize
            resized = cv2.resize(
                mask.astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Pad
            padded = np.zeros((final_h, final_w), dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            
            transformed_masks.append(padded)
        
        return np.stack(transformed_masks, axis=0)


class MultiModelPreprocessor:
    """
    Orchestrates preprocessing for multiple models.
    
    Architecture:
    1. Each model has its own preprocessor
    2. Single image is preprocessed for all active models
    3. Returns dict: model_name -> (tensor, metadata)
    
    Example:
        >>> preprocessor = MultiModelPreprocessor(
        >>>     active_models=['grounding_dino', 'sam'],
        >>>     config_path='configs/defaults/preprocessing.yaml'
        >>> )
        >>> results = preprocessor.preprocess_batch(image, boxes, masks)
        >>> # Returns: {
        >>> #   'grounding_dino': (tensor, metadata),
        >>> #   'sam': (tensor, metadata)
        >>> # }
    """
    PREPROCESSOR_REGISTRY = {
        'sam': SAMPreprocessor,
        'grounding_dino': GroundingDINOPreprocessor,
        'yolo': YOLOPreprocessor,
    }

    def __init__(self, active_models: List[str], config_path: str):
        """
        Args:
            active_models: List of model names to preprocess for
            config_path: Path to preprocessing.yaml
        """
        config = load_config(config_path)
        self.config = config['preprocessing']

        self.preprocessors: Dict[str, BaseModelPreprocessor] = {}

        for model_name in active_models:
            if model_name not in self.PREPROCESSOR_REGISTRY:
                raise ValueError(
                    f"Unknown model: {model_name}. "
                    f"Available: {list(self.PREPROCESSOR_REGISTRY.keys())}"
                )

            if model_name not in self.config:
                raise ValueError(
                    f"No config found for {model_name} in {config_path}"
                )

            preprocessor_class = self.PREPROCESSOR_REGISTRY[model_name]
            self.preprocessors[model_name] = preprocessor_class(
                model_name,
                self.config[model_name]
            )

    def preprocess_batch(
        self,
        image: Image.Image,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Dict[str, Tuple[torch.Tensor, Dict]]:
        """
        Preprocess image for all active models.
        
        Args:
            image: PIL Image (RGB)
            boxes: Optional boxes in COCO format [x, y, w, h] (N, 4)
            masks: Optional masks (N, H, W)
        
        Returns:
            Dict mapping model_name → (preprocessed_tensor, metadata)
        """
        results = {}
        for model_name, preprocessor in self.preprocessors.items():
            results[model_name] = preprocessor.preprocess(image, boxes, masks)
        return results

    def preprocess_for_model(
        self,
        image: Image.Image,
        model_name: str,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Preprocess for a specific model only."""
        if model_name not in self.preprocessors:
            raise ValueError(
                f"Model {model_name} not loaded. "
                f"Available: {list(self.preprocessors.keys())}"
            )
        return self.preprocessors[model_name].preprocess(image, boxes, masks)
    
    def get_preprocessor(self, model_name: str) -> BaseModelPreprocessor:
        """Get preprocessor for a specific model (for coordinate transforms)."""
        if model_name not in self.preprocessors:
            raise ValueError(
                f"Model {model_name} not loaded. "
                f"Available: {list(self.preprocessors.keys())}"
            )
        return self.preprocessors[model_name]
    
    @classmethod
    def register_preprocessor(
        cls, 
        model_name: str, 
        preprocessor_class: type
    ):
        """
        Register a new model preprocessor (for extensibility).
        
        Example:
            >>> class MyModelPreprocessor(BaseModelPreprocessor):
            >>>     ...
            >>> 
            >>> MultiModelPreprocessor.register_preprocessor(
            >>>     'my_model', MyModelPreprocessor
            >>> )
        """
        if not issubclass(preprocessor_class, BaseModelPreprocessor):
            raise TypeError(
                f"{preprocessor_class} must inherit from BaseModelPreprocessor"
            )
        cls.PREPROCESSOR_REGISTRY[model_name] = preprocessor_class


def create_preprocessor_from_models(
    model_names: List[str],
    config_path: Optional[str] = None
) -> MultiModelPreprocessor:
    """
    Helper function to create MultiModelPreprocessor from model names.
    
    This is a convenience function for the training pipeline.
    
    Args:
        model_names: List of model names (e.g., ['grounding_dino', 'sam'])
        config_path: Optional path to preprocessing config. 
                     Defaults to 'configs/defaults/preprocessing.yaml'
    
    Returns:
        MultiModelPreprocessor instance
    
    Example:
        >>> from ml_engine.data import create_preprocessor_from_models
        >>> preprocessor = create_preprocessor_from_models(['sam', 'grounding_dino'])
        >>> # Use in dataset
        >>> dataset = manager.create_pytorch_dataset(
        >>>     split='train',
        >>>     preprocessor=preprocessor
        >>> )
    """
    if config_path is None:
        from core.constants import DEFAULT_CONFIGS_DIR
        config_path = str(DEFAULT_CONFIGS_DIR / 'preprocessing.yaml')

    return MultiModelPreprocessor(
        active_models=model_names,
        config_path=config_path
    )
