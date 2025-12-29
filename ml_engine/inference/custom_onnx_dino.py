"""
Custom ONNX Grounding DINO inference.

Uses the ONNX model exported from our own export script (scripts/export_grounding_dino_onnx.py).
This model uses the same preprocessing as PyTorch, unlike the HuggingFace ONNX model.

Key design for variable image sizes:
- ONNX model is exported with fixed input size (e.g., 800x800)
- Preprocessing: resize with aspect ratio preservation + padding
- Model outputs: boxes in normalized [0,1] coordinates relative to padded input
- Postprocessing: scale boxes back to original image coordinates

Usage:
    detector = CustomONNXGroundingDINO(
        model_path="data/models/groundingdino_swint.onnx",
        bert_path="data/models/pretrained/bert-base-uncased"
    )
    detections = detector.predict_with_classes(image, ["cat", "dog"])
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of image preprocessing with metadata for postprocessing."""
    image: np.ndarray          # Preprocessed image [1, 3, H, W]
    original_size: Tuple[int, int]  # (height, width) of original image
    input_size: Tuple[int, int]     # (height, width) of model input (e.g., 800x800)
    scale: float               # Scale factor applied
    pad_w: int                 # Padding on width (right)
    pad_h: int                 # Padding on height (bottom)


class CustomONNXGroundingDINO:
    """
    ONNX Runtime-based Grounding DINO detector using custom export.
    
    Handles variable image sizes with proper preprocessing/postprocessing:
    - Resize with aspect ratio preservation
    - Pad to fixed input size
    - Scale boxes back to original coordinates
    """
    
    def __init__(
        self,
        model_path: str,
        bert_path: str = "data/models/pretrained/bert-base-uncased",
        device: str = "cuda",
        max_text_len: int = 256,
        input_size: Tuple[int, int] = (800, 800),
    ):
        """
        Initialize Custom ONNX Grounding DINO.
        
        Args:
            model_path: Path to custom ONNX model file
            bert_path: Path to BERT tokenizer
            device: "cuda" or "cpu"
            max_text_len: Maximum text length for tokenization
            input_size: Fixed input size (height, width) for ONNX model
        """
        self.model_path = Path(model_path)
        self.bert_path = bert_path
        self.device = device
        self.max_text_len = max_text_len
        self.input_size = input_size  # (H, W)
        
        self._session = None
        self._tokenizer = None
        self._loaded = False
        
        logger.info("CustomONNXGroundingDINO initialized (model: %s, device: %s, input: %s)", 
                   self.model_path.name, device, input_size)
    
    def load(self) -> None:
        """Load ONNX model and tokenizer."""
        if self._loaded:
            return
        
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        # Configure ONNX Runtime session
        providers = self._get_providers()
        logger.info("Loading custom ONNX model with providers: %s", providers)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Log which provider is actually being used
        actual_provider = self._session.get_providers()[0]
        logger.info("ONNX Runtime using: %s", actual_provider)
        
        # Get input/output names
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        logger.info("Model inputs: %s", self._input_names)
        logger.info("Model outputs: %s", self._output_names)
        
        # Load tokenizer
        logger.info("Loading tokenizer from: %s", self.bert_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        
        self._loaded = True
        logger.info("Custom ONNX model loaded successfully")
    
    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers based on device."""
        if self.device == "cuda":
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    
    def _preprocess_image(self, image: np.ndarray) -> PreprocessResult:
        """
        Preprocess image for ONNX model with fixed input size.
        
        Pipeline:
        1. Convert BGR to RGB
        2. Resize to fit in input_size while preserving aspect ratio
        3. Pad to exact input_size (bottom-right padding)
        4. Normalize with ImageNet mean/std
        
        Args:
            image: BGR image (OpenCV format) [H, W, C]
            
        Returns:
            PreprocessResult with preprocessed image and metadata
        """
        import cv2
        
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size
        
        # Step 1: Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Resize with aspect ratio preservation
        # Calculate scale to fit image in target size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Step 3: Pad to target size (bottom-right padding)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        # Create padded image (zero padding)
        image_padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
        image_padded[:new_h, :new_w, :] = image_resized.astype(np.float32) / 255.0
        
        # Step 4: ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_padded - mean) / std
        
        # HWC to CHW
        image_chw = image_norm.transpose(2, 0, 1)
        
        # Add batch dimension
        image_batch = image_chw[np.newaxis, ...].astype(np.float32)
        
        return PreprocessResult(
            image=image_batch,
            original_size=(orig_h, orig_w),
            input_size=(target_h, target_w),
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
        )
    
    def _tokenize(self, text_prompt: str, batch_size: int = 1) -> dict:
        """
        Tokenize text prompt.
        
        Args:
            text_prompt: Text prompt (e.g., "cat. dog.")
            batch_size: Batch size
            
        Returns:
            Dict with tokenized inputs
        """
        from groundingdino.models.GroundingDINO.bertwarper import (
            generate_masks_with_special_tokens_and_transfer_map
        )
        
        # Tokenize
        tokenized = self._tokenizer(
            [text_prompt] * batch_size,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        )
        
        # Get special tokens for mask generation
        specical_tokens = self._tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        
        # Generate masks
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, specical_tokens, self._tokenizer
        )
        
        # Truncate if needed
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]
            position_ids = position_ids[:, :self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :self.max_text_len]
        
        return {
            "input_ids": tokenized["input_ids"].numpy().astype(np.int64),
            "attention_mask": tokenized["attention_mask"].numpy().astype(np.int64),
            "token_type_ids": tokenized["token_type_ids"].numpy().astype(np.int64),
            "position_ids": position_ids.numpy().astype(np.int64),
            "text_token_mask": text_self_attention_masks.numpy().astype(np.bool_),
        }
    
    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run detection on a single image.
        
        Args:
            image: BGR image (OpenCV format)
            text_prompt: Text prompt (e.g., "cat. dog.")
            box_threshold: Confidence threshold for boxes
            text_threshold: Threshold for text matching
            
        Returns:
            Tuple of (boxes_xyxy, scores, class_ids) in original image coordinates
        """
        self.load()
        
        # Preprocess image (resize + pad + normalize)
        preprocess = self._preprocess_image(image)
        
        # Tokenize text
        text_inputs = self._tokenize(text_prompt)
        
        # Build ONNX inputs
        onnx_inputs = {
            "inputs": preprocess.image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "position_ids": text_inputs["position_ids"],
            "token_type_ids": text_inputs["token_type_ids"],
            "text_token_mask": text_inputs["text_token_mask"],
        }
        
        # Run inference
        outputs = self._session.run(None, onnx_inputs)
        
        # Parse outputs: pred_logits, pred_boxes
        pred_logits = outputs[0]  # [B, num_queries, max_text_len]
        pred_boxes = outputs[1]   # [B, num_queries, 4] in cxcywh normalized
        
        # Post-process with preprocessing metadata
        boxes, scores, class_ids = self._post_process(
            pred_logits=pred_logits[0],
            pred_boxes=pred_boxes[0],
            preprocess=preprocess,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        return boxes, scores, class_ids
    
    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
    ) -> 'Detections':
        """
        Run detection with class prompts (compatible with PyTorch interface).
        
        Args:
            image: BGR image (OpenCV format)
            classes: List of class names to detect
            box_threshold: Confidence threshold for boxes
            text_threshold: Threshold for text matching
            
        Returns:
            Detections object with xyxy, confidence, class_id attributes
        """
        import supervision as sv
        
        # Build text prompt from classes (lowercase with dots)
        text_prompt = ". ".join([c.lower() for c in classes]) + "."
        
        boxes, scores, class_ids = self.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=boxes if len(boxes) > 0 else np.empty((0, 4)),
            confidence=scores if len(scores) > 0 else np.empty(0),
            class_id=class_ids if len(class_ids) > 0 else np.empty(0, dtype=int)
        )
        
        return detections
    
    def _post_process(
        self,
        pred_logits: np.ndarray,
        pred_boxes: np.ndarray,
        preprocess: PreprocessResult,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process model outputs and scale boxes to original image coordinates.
        
        The key insight from TAO:
        - Model outputs boxes in normalized [0,1] coordinates relative to input tensor
        - We need to account for padding and scaling to get original coordinates
        
        Args:
            pred_logits: [num_queries, max_text_len]
            pred_boxes: [num_queries, 4] in cxcywh normalized to input tensor
            preprocess: Preprocessing metadata (scale, padding, sizes)
            box_threshold: Confidence threshold
            text_threshold: Text threshold
            
        Returns:
            Tuple of (boxes_xyxy, scores, class_ids) in original image coordinates
        """
        # Sigmoid on logits
        probs = self._sigmoid(pred_logits)  # [num_queries, max_text_len]
        
        # Get max probability per query
        max_probs = probs.max(axis=-1)  # [num_queries]
        max_indices = probs.argmax(axis=-1)  # [num_queries]
        
        # Filter by threshold
        keep_mask = max_probs > box_threshold
        keep_indices = np.where(keep_mask)[0]
        
        if len(keep_indices) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        
        # Get filtered results
        filtered_boxes = pred_boxes[keep_indices]
        filtered_scores = max_probs[keep_indices]
        filtered_class_ids = max_indices[keep_indices]
        
        # Convert boxes from normalized cxcywh to original xyxy coordinates
        boxes_xyxy = self._cxcywh_to_original_xyxy(filtered_boxes, preprocess)
        
        # Map token indices to class IDs (simplified: use token index directly)
        # In practice, you'd map these to actual class indices
        class_ids = filtered_class_ids.astype(np.int64)
        
        return boxes_xyxy, filtered_scores, class_ids
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _cxcywh_to_original_xyxy(
        self,
        boxes: np.ndarray,
        preprocess: PreprocessResult
    ) -> np.ndarray:
        """
        Convert boxes from normalized cxcywh to original image xyxy coordinates.
        
        The transformation pipeline:
        1. boxes are in [0,1] normalized to input tensor size (800x800)
        2. Convert to pixel coordinates in input tensor
        3. Remove padding effect (boxes in padded region will be clipped)
        4. Undo the resize scaling to get original coordinates
        
        Args:
            boxes: [N, 4] in cxcywh normalized [0,1]
            preprocess: Preprocessing metadata
            
        Returns:
            [N, 4] in xyxy original image coordinates
        """
        input_h, input_w = preprocess.input_size
        orig_h, orig_w = preprocess.original_size
        scale = preprocess.scale
        
        # Effective image size in input tensor (before padding)
        effective_w = input_w - preprocess.pad_w
        effective_h = input_h - preprocess.pad_h
        
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Step 1: Convert normalized to input tensor pixel coordinates
        x1_input = (cx - bw / 2) * input_w
        y1_input = (cy - bh / 2) * input_h
        x2_input = (cx + bw / 2) * input_w
        y2_input = (cy + bh / 2) * input_h
        
        # Step 2: Clip to effective area (remove padding region)
        x1_input = np.clip(x1_input, 0, effective_w)
        y1_input = np.clip(y1_input, 0, effective_h)
        x2_input = np.clip(x2_input, 0, effective_w)
        y2_input = np.clip(y2_input, 0, effective_h)
        
        # Step 3: Undo resize scaling to get original coordinates
        x1_orig = x1_input / scale
        y1_orig = y1_input / scale
        x2_orig = x2_input / scale
        y2_orig = y2_input / scale
        
        # Step 4: Final clip to original image bounds
        x1_orig = np.clip(x1_orig, 0, orig_w)
        y1_orig = np.clip(y1_orig, 0, orig_h)
        x2_orig = np.clip(x2_orig, 0, orig_w)
        y2_orig = np.clip(y2_orig, 0, orig_h)
        
        return np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

