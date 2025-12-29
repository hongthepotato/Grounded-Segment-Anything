"""
ONNX-based Grounding DINO inference.

This module provides an ONNX Runtime backend for Grounding DINO,
offering faster inference compared to PyTorch.

Uses the HuggingFace transformers processor for preprocessing/postprocessing
and ONNX Runtime for model inference.

Usage:
    detector = ONNXGroundingDINO(
        model_path="grounding-dino-tiny-ONNX/onnx/model_fp16.onnx",
        processor_path="grounding-dino-tiny-ONNX"
    )
    boxes, scores, labels = detector.predict(image, "cat. dog.")
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class ONNXGroundingDINO:
    """
    ONNX Runtime-based Grounding DINO detector.
    
    Provides the same interface as the PyTorch model but uses
    ONNX Runtime for inference with optional GPU acceleration.
    """
    
    def __init__(
        self,
        model_path: str,
        processor_path: str,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """
        Initialize ONNX Grounding DINO.
        
        Args:
            model_path: Path to ONNX model file (e.g., model_fp16.onnx)
            processor_path: Path to HuggingFace processor directory
            device: "cuda" or "cpu"
            use_fp16: Whether to use FP16 model (ignored if model_path already specifies)
        """
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path)
        self.device = device
        self.use_fp16 = use_fp16
        
        self._session = None
        self._processor = None
        self._loaded = False
        
        logger.info("ONNXGroundingDINO initialized (model: %s, device: %s)", 
                   self.model_path.name, device)
    
    def load(self) -> None:
        """Load ONNX model and processor."""
        if self._loaded:
            return
        
        import onnxruntime as ort
        from transformers import AutoProcessor
        
        # Configure ONNX Runtime session
        providers = self._get_providers()
        logger.info("Loading ONNX model with providers: %s", providers)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Try to load with preferred providers, fallback if needed
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            # INT8/quantized models may not work with CUDA, fallback to CPU
            model_name = self.model_path.name.lower()
            if self.device == "cuda" and ("int8" in model_name or "quantized" in model_name or "uint8" in model_name):
                logger.warning(
                    "Failed to load quantized model with CUDA: %s. "
                    "Falling back to CPU. Consider using fp16 variant for GPU.", e
                )
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
            else:
                raise
        
        # Log which provider is actually being used
        actual_provider = self._session.get_providers()[0]
        logger.info("ONNX Runtime using: %s", actual_provider)
        
        # Load HuggingFace processor for pre/post processing
        logger.info("Loading processor from: %s", self.processor_path)
        self._processor = AutoProcessor.from_pretrained(str(self.processor_path))
        
        self._loaded = True
        logger.info("ONNX model loaded successfully")
    
    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers based on device."""
        if self.device == "cuda":
            return [
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        return ['CPUExecutionProvider']
    
    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run detection on a single image.
        
        Args:
            image: BGR image (OpenCV format) or RGB image
            text_prompt: Text prompt for detection (e.g., "cat. dog.")
            box_threshold: Confidence threshold for boxes
            text_threshold: Threshold for text matching
            
        Returns:
            Tuple of (boxes_xyxy, scores, labels)
            - boxes_xyxy: (N, 4) array of boxes in xyxy format
            - scores: (N,) array of confidence scores
            - labels: List of N label strings
        """
        self.load()
        
        from PIL import Image
        import cv2
        
        # Convert BGR to RGB if needed (assume BGR input from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        h, w = image_rgb.shape[:2]
        
        # Convert to PIL for processor
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess with HuggingFace processor
        # Note: text should be lowercase and end with dots
        text_prompt_formatted = self._format_text_prompt(text_prompt)
        inputs = self._processor(
            images=pil_image,
            text=text_prompt_formatted,
            return_tensors="np"
        )
        
        # Get required input names from the model
        input_names = [inp.name for inp in self._session.get_inputs()]
        
        # Prepare ONNX inputs - include all inputs the model expects
        onnx_inputs = {}
        
        if "pixel_values" in input_names:
            onnx_inputs["pixel_values"] = inputs["pixel_values"].astype(np.float32)
        
        if "pixel_mask" in input_names:
            # pixel_mask should be provided by processor, or create ones
            if "pixel_mask" in inputs:
                onnx_inputs["pixel_mask"] = inputs["pixel_mask"].astype(np.int64)
            else:
                # Create mask of ones with same spatial dims as pixel_values
                # pixel_values shape: (1, 3, H, W)
                _, _, ph, pw = inputs["pixel_values"].shape
                onnx_inputs["pixel_mask"] = np.ones((1, ph, pw), dtype=np.int64)
        
        if "input_ids" in input_names:
            onnx_inputs["input_ids"] = inputs["input_ids"].astype(np.int64)
        
        if "attention_mask" in input_names:
            onnx_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
        
        if "token_type_ids" in input_names:
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                onnx_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"]).astype(np.int64)
        
        # Run inference
        outputs = self._session.run(None, onnx_inputs)
        
        # Parse outputs
        # Output order: logits, pred_boxes
        logits = outputs[0]  # (1, num_queries, num_tokens)
        pred_boxes = outputs[1]  # (1, num_queries, 4) in cxcywh normalized
        
        # Post-process
        boxes, scores, labels = self._post_process(
            logits=logits,
            pred_boxes=pred_boxes,
            input_ids=inputs["input_ids"],
            target_size=(h, w),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        return boxes, scores, labels
    
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
        
        # Build text prompt from classes
        text_prompt = ". ".join(classes) + "."
        
        boxes, scores, labels = self.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        # Map labels to class IDs
        class_ids = []
        for label in labels:
            label_lower = label.lower().strip()
            matched = False
            for idx, cls in enumerate(classes):
                if cls.lower() in label_lower or label_lower in cls.lower():
                    class_ids.append(idx)
                    matched = True
                    break
            if not matched:
                class_ids.append(0)
        
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=boxes if len(boxes) > 0 else np.empty((0, 4)),
            confidence=scores if len(scores) > 0 else np.empty(0),
            class_id=np.array(class_ids) if class_ids else np.empty(0, dtype=int)
        )
        
        return detections
    
    def _format_text_prompt(self, text: str) -> str:
        """Format text prompt for Grounding DINO (lowercase, with dots)."""
        # Ensure lowercase
        text = text.lower()
        # Ensure ends with dot
        if not text.endswith("."):
            text = text + "."
        return text
    
    def _post_process(
        self,
        logits: np.ndarray,
        pred_boxes: np.ndarray,
        input_ids: np.ndarray,
        target_size: Tuple[int, int],
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Post-process model outputs to get boxes, scores, and labels.
        
        Args:
            logits: (1, num_queries, num_tokens) logits
            pred_boxes: (1, num_queries, 4) boxes in cxcywh normalized
            input_ids: (1, seq_len) input token IDs
            target_size: (height, width) of original image
            box_threshold: Confidence threshold
            text_threshold: Text matching threshold
            
        Returns:
            Tuple of (boxes_xyxy, scores, labels)
        """
        # Sigmoid on logits
        probs = self._sigmoid(logits[0])  # (num_queries, num_tokens)
        
        # Get max probability per query (across all tokens)
        max_probs = probs.max(axis=-1)  # (num_queries,)
        
        # Filter by box threshold
        keep_mask = max_probs > box_threshold
        keep_indices = np.where(keep_mask)[0]
        
        if len(keep_indices) == 0:
            return np.empty((0, 4)), np.empty(0), []
        
        # Get filtered boxes and scores
        filtered_boxes = pred_boxes[0, keep_indices]  # (N, 4) cxcywh normalized
        filtered_scores = max_probs[keep_indices]  # (N,)
        filtered_probs = probs[keep_indices]  # (N, num_tokens)
        
        # Convert boxes from cxcywh to xyxy and denormalize
        h, w = target_size
        boxes_xyxy = self._cxcywh_to_xyxy(filtered_boxes, w, h)
        
        # Extract labels from token probabilities
        labels = self._extract_labels(
            probs=filtered_probs,
            input_ids=input_ids[0],
            text_threshold=text_threshold,
        )
        
        return boxes_xyxy, filtered_scores, labels
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _cxcywh_to_xyxy(
        self,
        boxes: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """Convert boxes from cxcywh normalized to xyxy pixel coordinates."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, width)
        y1 = np.clip(y1, 0, height)
        x2 = np.clip(x2, 0, width)
        y2 = np.clip(y2, 0, height)
        
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def _extract_labels(
        self,
        probs: np.ndarray,
        input_ids: np.ndarray,
        text_threshold: float,
    ) -> List[str]:
        """
        Extract label strings from token probabilities.
        
        Args:
            probs: (N, num_tokens) probabilities per detection
            input_ids: (seq_len,) input token IDs
            text_threshold: Threshold for token selection
            
        Returns:
            List of N label strings
        """
        labels = []
        
        # Get tokenizer from processor
        tokenizer = self._processor.tokenizer
        
        for det_probs in probs:
            # Find tokens above threshold
            token_mask = det_probs > text_threshold
            token_indices = np.where(token_mask)[0]
            
            if len(token_indices) == 0:
                # Fallback: use argmax token
                token_indices = [np.argmax(det_probs)]
            
            # Map token indices to input_ids indices
            # Filter to valid range
            valid_indices = [idx for idx in token_indices if idx < len(input_ids)]
            
            if valid_indices:
                # Get token IDs and decode
                token_ids = input_ids[valid_indices]
                label = tokenizer.decode(token_ids, skip_special_tokens=True)
                label = label.strip().replace(".", "").strip()
            else:
                label = "unknown"
            
            labels.append(label)
        
        return labels
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# Convenience function to create detector
def create_onnx_detector(
    model_dir: str = "grounding-dino-tiny-ONNX",
    model_variant: str = "fp16",
    device: str = "cuda",
) -> ONNXGroundingDINO:
    """
    Create an ONNX Grounding DINO detector.
    
    Args:
        model_dir: Path to the model directory
        model_variant: One of "fp32", "fp16", "int8", "q4"
        device: "cuda" or "cpu"
        
    Returns:
        Configured ONNXGroundingDINO instance
    """
    model_dir = Path(model_dir)
    
    # Map variant to filename
    variant_map = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "int8": "model_int8.onnx",
        "q4": "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
        "uint8": "model_uint8.onnx",
        "quantized": "model_quantized.onnx",
    }
    
    model_file = variant_map.get(model_variant, f"model_{model_variant}.onnx")
    model_path = model_dir / "onnx" / model_file
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return ONNXGroundingDINO(
        model_path=str(model_path),
        processor_path=str(model_dir),
        device=device,
    )

