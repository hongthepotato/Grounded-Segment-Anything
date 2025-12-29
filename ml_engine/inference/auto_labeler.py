"""
Auto-labeling service using Grounding DINO + MobileSAM.

This module provides automatic annotation generation for images using:
1. Grounding DINO: Text-prompted object detection (text -> boxes)
2. NMS: Non-Maximum Suppression to filter duplicate detections
3. MobileSAM: Box-prompted segmentation (boxes -> masks)

Output is in standard COCO format with both bounding boxes and segmentation masks.

Usage:
    from ml_engine.inference.auto_labeler import AutoLabeler
    
    labeler = AutoLabeler()
    coco_annotations = labeler.label_images(
        image_paths=['img1.jpg', 'img2.jpg'],
        class_prompts=['ear of bag', 'defect']
    )
"""
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.ops

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GroundingDINO"))
sys.path.insert(0, str(project_root / "deps" / "segment_anything"))
# EfficientSAM directory contains MobileSAM as a subdirectory
sys.path.insert(0, str(project_root / "EfficientSAM"))

# Grounding DINO - reuse existing inference module
from groundingdino.util.inference import Model as GroundingDINOModel

# MobileSAM - reuse existing setup from EfficientSAM/MobileSAM/
# Note: MobileSAM is bundled in this project under EfficientSAM/MobileSAM/
# No need to install from https://github.com/ChaoningZhang/MobileSAM
from MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from segment_anything import SamPredictor

# Config utilities
from core.config import save_json

logger = logging.getLogger(__name__)


# Output mode options
OUTPUT_BOXES_ONLY = "boxes"
OUTPUT_MASKS_ONLY = "masks"
OUTPUT_BOTH = "both"


class PipelineProfiler:
    """
    Comprehensive profiler for auto-labeling pipeline.
    
    Tracks timing for all steps including model loading, image I/O,
    preprocessing, inference, and post-processing.
    
    Usage:
        profiler = PipelineProfiler(enabled=True)
        
        with profiler.measure("step_name"):
            # code to profile
            
        profiler.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        self.total_images = 0
        self.total_detections = 0
        self.total_masks = 0
    
    def _sync_cuda(self):
        """Synchronize CUDA for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def start(self, name: str):
        """Start timing a step."""
        if self.enabled:
            self._sync_cuda()
            self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing a step and record the duration."""
        if self.enabled and name in self._start_times:
            self._sync_cuda()
            elapsed = (time.perf_counter() - self._start_times[name]) * 1000  # ms
            self.timings[name].append(elapsed)
            del self._start_times[name]
            return elapsed
        return 0.0
    
    class _TimingContext:
        """Context manager for timing blocks."""
        def __init__(self, profiler: 'PipelineProfiler', name: str):
            self.profiler = profiler
            self.name = name
        
        def __enter__(self):
            self.profiler.start(self.name)
            return self
        
        def __exit__(self, *args):
            self.profiler.stop(self.name)
    
    def measure(self, name: str) -> _TimingContext:
        """Context manager for timing a block of code."""
        return self._TimingContext(self, name)
    
    def record(self, name: str, elapsed_ms: float):
        """Manually record a timing."""
        if self.enabled:
            self.timings[name].append(elapsed_ms)
    
    def add_counts(self, images: int = 0, detections: int = 0, masks: int = 0):
        """Track counts for summary."""
        self.total_images += images
        self.total_detections += detections
        self.total_masks += masks
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timing step."""
        times = self.timings.get(name, [])
        if not times:
            return {"total": 0, "mean": 0, "min": 0, "max": 0, "count": 0}
        return {
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times)
        }
    
    def print_summary(self):
        """Print comprehensive timing summary."""
        if not self.enabled or not self.timings:
            return
        
        print("\n" + "=" * 80)
        print("                    PIPELINE PROFILING SUMMARY")
        print("=" * 80)
        
        # Overall stats
        print(f"\n📊 Overall Statistics:")
        print(f"   Images processed:    {self.total_images}")
        print(f"   Total detections:    {self.total_detections}")
        print(f"   Total masks:         {self.total_masks}")
        
        # Define step categories for organized output
        model_loading_steps = ["load_grounding_dino", "load_mobile_sam", "model_loading_total"]
        per_image_steps = [
            "image_read", "image_color_convert", 
            "dino_inference", "nms",
            "sam_set_image", "sam_predict_single_mask", "sam_predict_all_masks",
            "box_format_convert", "result_build",
            "per_image_total"
        ]
        viz_steps = ["visualization"]
        
        # Print model loading times (one-time costs)
        print(f"\n⏱️  Model Loading (one-time):")
        print("-" * 80)
        print(f"{'Step':<40} {'Time (ms)':>12}")
        print("-" * 80)
        
        for step in model_loading_steps:
            if step in self.timings:
                stats = self.get_stats(step)
                print(f"   {step:<37} {stats['total']:>12.2f}")
        
        # Print per-image inference times
        print(f"\n⏱️  Per-Image Inference:")
        print("-" * 80)
        print(f"{'Step':<40} {'Total(ms)':>10} {'Mean(ms)':>10} {'Min':>8} {'Max':>8}")
        print("-" * 80)
        
        inference_total = 0
        for step in per_image_steps:
            if step in self.timings:
                stats = self.get_stats(step)
                print(f"   {step:<37} {stats['total']:>10.2f} {stats['mean']:>10.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")
                if step != "per_image_total":
                    inference_total += stats['total']
        
        # Print visualization times
        if any(step in self.timings for step in viz_steps):
            print(f"\n⏱️  Visualization:")
            print("-" * 80)
            for step in viz_steps:
                if step in self.timings:
                    stats = self.get_stats(step)
                    print(f"   {step:<37} {stats['total']:>10.2f} {stats['mean']:>10.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")
        
        # Print any other steps not in the predefined categories
        all_known = set(model_loading_steps + per_image_steps + viz_steps)
        other_steps = [s for s in self.timings.keys() if s not in all_known]
        if other_steps:
            print(f"\n⏱️  Other:")
            print("-" * 80)
            for step in other_steps:
                stats = self.get_stats(step)
                print(f"   {step:<37} {stats['total']:>10.2f} {stats['mean']:>10.2f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("📈 SUMMARY")
        print("=" * 80)
        
        if self.total_images > 0:
            # Calculate key metrics
            per_image_stats = self.get_stats("per_image_total")
            avg_per_image = per_image_stats['mean'] if per_image_stats['count'] > 0 else 0
            
            if avg_per_image > 0:
                fps = 1000.0 / avg_per_image
                print(f"   Average time per image:  {avg_per_image:.2f} ms")
                print(f"   Throughput:              {fps:.2f} FPS")
            
            # Breakdown
            if inference_total > 0:
                print(f"\n   Time Breakdown (inference only):")
                breakdown_steps = ["dino_inference", "sam_set_image", "sam_predict_all_masks", "nms", "image_read"]
                for step in breakdown_steps:
                    if step in self.timings:
                        stats = self.get_stats(step)
                        pct = (stats['total'] / inference_total) * 100
                        bar_len = int(pct / 2)
                        bar = "█" * bar_len + "░" * (50 - bar_len)
                        print(f"   {step:<25} {bar} {pct:>5.1f}%")
        
        print("=" * 80 + "\n")


# Backend options
BACKEND_PYTORCH = "pytorch"
BACKEND_ONNX = "onnx"
BACKEND_CUSTOM_ONNX = "custom_onnx"


@dataclass
class AutoLabelerConfig:
    """Configuration for AutoLabeler."""
    # Backend selection: "pytorch", "onnx", or "custom_onnx"
    # - "pytorch": Original PyTorch model (default)
    # - "onnx": HuggingFace ONNX model with ONNX Runtime
    # - "custom_onnx": Our custom-exported ONNX model
    backend: str = BACKEND_PYTORCH
    
    # Model paths (PyTorch backend)
    grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "data/models/pretrained/groundingdino_swint_ogc.pth"
    mobile_sam_checkpoint: str = "data/models/pretrained/mobile_sam.pt"
    
    # ONNX model paths (onnx backend - HuggingFace)
    onnx_model_dir: str = "grounding-dino-tiny-ONNX"
    onnx_model_variant: str = "fp16"  # "fp32", "fp16", "int8", "q4"
    
    # Custom ONNX model path (custom_onnx backend)
    custom_onnx_path: str = "data/models/groundingdino_swint.onnx"
    bert_path: str = "data/models/pretrained/bert-base-uncased"
    custom_onnx_input_size: Tuple[int, int] = (800, 800)  # (H, W) for custom ONNX

    # Detection thresholds
    box_threshold: float = 0.5
    text_threshold: float = 0.5
    nms_threshold: float = 0.7

    # Output mode: "boxes", "masks", or "both"
    # - "boxes": Only bounding boxes (faster, no SAM needed, default)
    # - "masks": Only segmentation masks (boxes used internally but not in output)
    # - "both": Both boxes and masks
    output_mode: str = OUTPUT_BOXES_ONLY

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Profiling: Enable detailed timing analysis
    enable_profiling: bool = False


class AutoLabeler:
    """
    Auto-labeling service using Grounding DINO + MobileSAM.
    
    Pipeline:
    1. Grounding DINO: Detect objects based on text prompts
    2. NMS: Filter overlapping detections
    3. MobileSAM: Generate segmentation masks for each detection
    4. Export: Convert to COCO format
    
    Example:
        labeler = AutoLabeler()
        coco_json = labeler.label_images(
            image_paths=['img1.jpg', 'img2.jpg'],
            class_prompts=['ear of bag', 'defect']
        )
        # Returns COCO-format dict with images, annotations, categories
    """

    def __init__(self, config: Optional[AutoLabelerConfig] = None):
        """
        Initialize AutoLabeler with models.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or AutoLabelerConfig()
        self.device = torch.device(self.config.device)
        self.backend = self.config.backend

        # Models will be loaded lazily
        self._grounding_dino = None  # PyTorch model
        self._onnx_detector = None   # ONNX model
        self._sam_predictor = None
        self._models_loaded = False
        
        # Profiler for timing analysis
        self.profiler = PipelineProfiler(enabled=self.config.enable_profiling)

        logger.info("AutoLabeler initialized (backend: %s, device: %s, profiling: %s)", 
                    self.backend, self.device, self.config.enable_profiling)

    def _load_models(self) -> None:
        """
        Load Grounding DINO and MobileSAM models.
        
        Supports three backends:
        - PyTorch: Original model from grounded_sam_simple_demo.py
        - ONNX: HuggingFace ONNX model with ONNX Runtime
        - Custom ONNX: Our custom-exported ONNX model
        
        Note: MobileSAM is only loaded if output_mode requires masks.
        """
        if self._models_loaded:
            return

        self.profiler.start("model_loading_total")
        
        if self.backend == BACKEND_ONNX:
            # Load HuggingFace ONNX model
            logger.info("Loading HuggingFace ONNX Grounding DINO model...")
            with self.profiler.measure("load_grounding_dino"):
                from ml_engine.inference.onnx_grounding_dino import create_onnx_detector
                self._onnx_detector = create_onnx_detector(
                    model_dir=self.config.onnx_model_dir,
                    model_variant=self.config.onnx_model_variant,
                    device=str(self.device).split(":")[0],  # "cuda:0" -> "cuda"
                )
                self._onnx_detector.load()
        elif self.backend == BACKEND_CUSTOM_ONNX:
            # Load custom-exported ONNX model
            logger.info("Loading custom ONNX Grounding DINO model...")
            with self.profiler.measure("load_grounding_dino"):
                from ml_engine.inference.custom_onnx_dino import CustomONNXGroundingDINO
                self._onnx_detector = CustomONNXGroundingDINO(
                    model_path=self.config.custom_onnx_path,
                    bert_path=self.config.bert_path,
                    device=str(self.device).split(":")[0],
                    input_size=self.config.custom_onnx_input_size,
                )
                self._onnx_detector.load()
        else:
            # Load PyTorch model (default)
            logger.info("Loading PyTorch Grounding DINO model...")
            with self.profiler.measure("load_grounding_dino"):
                self._grounding_dino = GroundingDINOModel(
                    model_config_path=self.config.grounding_dino_config,
                    model_checkpoint_path=self.config.grounding_dino_checkpoint,
                    device=str(self.device)
                )

        # Only load SAM if we need masks
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            logger.info("Loading MobileSAM model...")
            with self.profiler.measure("load_mobile_sam"):
                mobile_sam = setup_mobile_sam()
                checkpoint = torch.load(self.config.mobile_sam_checkpoint, map_location="cpu")
                mobile_sam.load_state_dict(checkpoint, strict=True)
                mobile_sam.to(device=self.device)
                mobile_sam.eval()
                
                self._sam_predictor = SamPredictor(mobile_sam)
        else:
            logger.info("Skipping MobileSAM (output_mode='boxes')")
            self._sam_predictor = None
        
        self.profiler.stop("model_loading_total")
        self._models_loaded = True
        logger.info("Models loaded successfully (backend: %s)", self.backend)
    
    def _detect_objects(
        self,
        image: np.ndarray,
        class_prompts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects using Grounding DINO.
        
        Args:
            image: BGR image (OpenCV format)
            class_prompts: List of class names to detect
            
        Returns:
            Tuple of (boxes_xyxy, confidences, class_ids) after NMS
        """
        # Use appropriate backend
        with self.profiler.measure("dino_inference"):
            if self.backend in (BACKEND_ONNX, BACKEND_CUSTOM_ONNX):
                detections = self._onnx_detector.predict_with_classes(
                    image=image,
                    classes=class_prompts,
                    box_threshold=self.config.box_threshold,
                    text_threshold=self.config.text_threshold,
                )
            else:
                # PyTorch backend (default)
                detections = self._grounding_dino.predict_with_classes(
                    image=image,
                    classes=class_prompts,
                    box_threshold=self.config.box_threshold,
                    text_threshold=self.config.text_threshold
                )
        
        # Check if any detections
        if len(detections.xyxy) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Apply NMS (from all demo patterns)
        with self.profiler.measure("nms"):
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.config.nms_threshold
            ).numpy().tolist()
        
        boxes = detections.xyxy[nms_idx]
        confidences = detections.confidence[nms_idx]
        class_ids = detections.class_id[nms_idx]
        
        logger.debug(f"Detected {len(boxes)} objects after NMS")
        return boxes, confidences, class_ids
    
    def _segment_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate segmentation masks for detected boxes using MobileSAM.
        
        Args:
            image: RGB image
            boxes: Array of boxes in xyxy format
            
        Returns:
            List of binary masks (one per box)
        """
        if len(boxes) == 0:
            return []
        
        # Set image (encodes once, from grounded_mobile_sam.py pattern)
        with self.profiler.measure("sam_set_image"):
            self._sam_predictor.set_image(image)
        
        masks = []
        self.profiler.start("sam_predict_all_masks")
        for box in boxes:
            # Predict mask for this box
            self.profiler.start("sam_predict_single_mask")
            mask_predictions, scores, _ = self._sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            self.profiler.stop("sam_predict_single_mask")
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            masks.append(mask_predictions[best_idx])
        self.profiler.stop("sam_predict_all_masks")
        
        self.profiler.add_counts(masks=len(masks))
        return masks
    
    def label_single_image(
        self,
        image_path: str,
        class_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate annotations for a single image.
        
        Args:
            image_path: Path to image file
            class_prompts: List of class names to detect
            
        Returns:
            Dict with (depending on output_mode):
                - boxes: List of [x, y, w, h] in COCO format (if output_mode is 'boxes' or 'both')
                - masks: List of binary masks (if output_mode is 'masks' or 'both')
                - class_ids: List of category IDs
                - scores: List of confidence scores
                - image_info: Dict with file_name, width, height
        """
        self._load_models()
        
        # Start per-image timing
        self.profiler.start("per_image_total")

        # Load image from disk
        with self.profiler.measure("image_read"):
            image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        with self.profiler.measure("image_color_convert"):
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]

        # Detect objects (profiling inside _detect_objects)
        boxes_xyxy, confidences, class_ids = self._detect_objects(image_bgr, class_prompts)
        
        # Convert boxes to COCO format [x, y, width, height]
        with self.profiler.measure("box_format_convert"):
            boxes_coco = []
            for box in boxes_xyxy:
                x1, y1, x2, y2 = box
                boxes_coco.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        
        # Generate masks if needed (profiling inside _segment_boxes)
        masks = []
        if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
            masks = self._segment_boxes(image_rgb, boxes_xyxy)
        
        # Build result based on output_mode
        with self.profiler.measure("result_build"):
            result = {
                'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
                'scores': confidences.tolist() if len(confidences) > 0 else [],
                'image_info': {
                    'file_name': os.path.basename(image_path),
                    'width': width,
                    'height': height
                }
            }
            
            # Include boxes if requested
            if self.config.output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH):
                result['boxes'] = boxes_coco
            
            # Include masks if requested
            if self.config.output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH):
                result['masks'] = masks
        
        # Stop per-image timing and record counts
        self.profiler.stop("per_image_total")
        self.profiler.add_counts(images=1, detections=len(boxes_xyxy))
        
        return result
    
    def label_images(
        self,
        image_paths: List[str],
        class_prompts: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate COCO-format annotations for multiple images.
        
        Args:
            image_paths: List of paths to image files
            class_prompts: List of class names to detect
            output_path: Optional path to save COCO JSON
            
        Returns:
            COCO-format dictionary with images, annotations, categories.
            Annotation content depends on output_mode:
            - "boxes": bbox only, no segmentation
            - "masks": segmentation only, bbox auto-generated from mask
            - "both": both bbox and segmentation
        """
        self._load_models()
        
        # Initialize COCO structure
        coco_output = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': i, 'name': name}
                for i, name in enumerate(class_prompts)
            ]
        }
        
        annotation_id = 1
        output_mode = self.config.output_mode
        
        for image_id, image_path in enumerate(image_paths, start=1):
            logger.info(f"Processing image {image_id}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.label_single_image(image_path, class_prompts)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
            
            # Add image info
            coco_output['images'].append({
                'id': image_id,
                'file_name': result['image_info']['file_name'],
                'width': result['image_info']['width'],
                'height': result['image_info']['height']
            })
            
            # Get data based on output mode
            boxes = result.get('boxes', [])
            masks = result.get('masks', [])
            num_detections = len(result['class_ids'])
            
            # Add annotations
            for i in range(num_detections):
                class_id = result['class_ids'][i]
                score = result['scores'][i]
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(class_id) if class_id is not None else 0,
                    'iscrowd': 0,
                    'score': float(score)
                }
                
                # Add bbox if available
                if output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH) and i < len(boxes):
                    box = boxes[i]
                    annotation['bbox'] = box
                    annotation['area'] = box[2] * box[3]  # width * height
                
                # Add segmentation if available
                if output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH) and i < len(masks):
                    mask = masks[i]
                    segmentation = self._mask_to_polygon(mask)
                    annotation['segmentation'] = segmentation
                    annotation['area'] = float(mask.sum()) if mask is not None and mask.sum() > 0 else annotation.get('area', 0)
                    
                    # For masks-only mode, generate bbox from mask
                    if output_mode == OUTPUT_MASKS_ONLY and 'bbox' not in annotation:
                        annotation['bbox'] = self._bbox_from_mask(mask)
                
                coco_output['annotations'].append(annotation)
                annotation_id += 1
        
        # Save if output path provided
        if output_path:
            save_json(coco_output, output_path)
            logger.info(f"Saved COCO annotations to: {output_path}")
        
        logger.info(
            f"Auto-labeling complete: {len(coco_output['images'])} images, "
            f"{len(coco_output['annotations'])} annotations (mode: {output_mode})"
        )
        
        return coco_output
    
    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> List[float]:
        """
        Generate bounding box from mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Bounding box [x, y, width, height] in COCO format
        """
        if mask is None or mask.sum() == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    
    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to COCO polygon format.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of polygons, each polygon is [x1, y1, x2, y2, ...]
        """
        if mask is None or mask.sum() == 0:
            return []
        
        # Ensure mask is uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # Skip small contours
            if len(contour) < 3:
                continue
            
            # Flatten to [x1, y1, x2, y2, ...]
            polygon = contour.flatten().tolist()
            
            # COCO requires at least 6 points (3 vertices)
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons


def visualize_detections(
    image_path: str,
    result: Dict[str, Any],
    class_prompts: List[str],
    output_path: str,
    show_boxes: bool = True,
    show_masks: bool = True,
    show_labels: bool = True,
    show_scores: bool = True
) -> None:
    """
    Visualize auto-labeling results on an image.
    
    Draws bounding boxes and/or masks with class labels and confidence scores.
    
    Args:
        image_path: Path to original image
        result: Detection result from label_single_image()
        class_prompts: List of class names
        output_path: Path to save visualized image
        show_boxes: Whether to draw bounding boxes
        show_masks: Whether to overlay masks
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not load image for visualization: {image_path}")
        return
    
    # Color palette (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    boxes = result.get('boxes', [])
    masks = result.get('masks', [])
    class_ids = result.get('class_ids', [])
    scores = result.get('scores', [])
    
    # Draw masks first (so boxes appear on top)
    if show_masks and masks:
        mask_overlay = image.copy()
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            color = colors[class_ids[i] % len(colors)] if i < len(class_ids) else colors[0]
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            # Blend with original
            mask_overlay = cv2.addWeighted(
                mask_overlay, 1.0,
                colored_mask, 0.4,
                0
            )
            # Draw mask contour
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(mask_overlay, contours, -1, color, 2)
        image = mask_overlay
    
    # Draw boxes
    if show_boxes and boxes:
        for i, box in enumerate(boxes):
            class_id = class_ids[i] if i < len(class_ids) else 0
            score = scores[i] if i < len(scores) else 0.0
            color = colors[class_id % len(colors)]
            
            # COCO format [x, y, w, h] -> xyxy
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Build label text
            if show_labels or show_scores:
                label_parts = []
                if show_labels:
                    class_name = class_prompts[class_id] if class_id < len(class_prompts) else f"cls_{class_id}"
                    label_parts.append(class_name)
                if show_scores:
                    label_parts.append(f"{score:.2f}")
                label_text = " ".join(label_parts)
                
                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_h - 8),
                    (x1 + text_w + 4, y1),
                    color,
                    -1
                )
                # Draw label text
                cv2.putText(
                    image, label_text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
    
    # Save visualization
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.debug(f"Saved visualization: {output_path}")


def visualize_batch(
    image_paths: List[str],
    results: List[Dict[str, Any]],
    class_prompts: List[str],
    output_dir: str,
    output_mode: str = OUTPUT_BOXES_ONLY
) -> int:
    """
    Visualize auto-labeling results for multiple images.
    
    Args:
        image_paths: List of paths to original images
        results: List of detection results from label_single_image()
        class_prompts: List of class names
        output_dir: Directory to save visualizations
        output_mode: "boxes", "masks", or "both"
        
    Returns:
        Number of images visualized
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    show_boxes = output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH)
    show_masks = output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH)
    
    count = 0
    for image_path, result in zip(image_paths, results):
        filename = Path(image_path).stem + "_viz.jpg"
        save_path = str(output_path / filename)
        
        visualize_detections(
            image_path=image_path,
            result=result,
            class_prompts=class_prompts,
            output_path=save_path,
            show_boxes=show_boxes,
            show_masks=show_masks
        )
        count += 1
    
    logger.info(f"Saved {count} visualizations to: {output_dir}")
    return count


def export_to_coco(
    detections_list: List[Dict[str, Any]],
    class_prompts: List[str],
    output_mode: str = OUTPUT_BOTH
) -> Dict[str, Any]:
    """
    Convert a list of detection results to COCO format.
    
    This is a standalone function for flexibility.
    
    Args:
        detections_list: List of detection dicts (from label_single_image)
        class_prompts: List of class names
        output_mode: "boxes", "masks", or "both" (default: "both")
        
    Returns:
        COCO-format dictionary
    """
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i, 'name': name}
            for i, name in enumerate(class_prompts)
        ]
    }
    
    annotation_id = 1
    
    for image_id, detection in enumerate(detections_list, start=1):
        # Add image info
        coco_output['images'].append({
            'id': image_id,
            'file_name': detection['image_info']['file_name'],
            'width': detection['image_info']['width'],
            'height': detection['image_info']['height']
        })
        
        # Get data based on what's available
        boxes = detection.get('boxes', [])
        masks = detection.get('masks', [])
        num_detections = len(detection['class_ids'])
        
        # Add annotations
        for i in range(num_detections):
            class_id = detection['class_ids'][i]
            score = detection['scores'][i]
            
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(class_id) if class_id is not None else 0,
                'iscrowd': 0,
                'score': float(score)
            }
            
            # Add bbox if available
            if output_mode in (OUTPUT_BOXES_ONLY, OUTPUT_BOTH) and i < len(boxes):
                box = boxes[i]
                annotation['bbox'] = box
                annotation['area'] = box[2] * box[3]
            
            # Add segmentation if available
            if output_mode in (OUTPUT_MASKS_ONLY, OUTPUT_BOTH) and i < len(masks):
                mask = masks[i]
                segmentation = AutoLabeler._mask_to_polygon(mask)
                annotation['segmentation'] = segmentation
                annotation['area'] = float(mask.sum()) if mask is not None and mask.sum() > 0 else annotation.get('area', 0)
                
                # For masks-only mode, generate bbox from mask
                if output_mode == OUTPUT_MASKS_ONLY and 'bbox' not in annotation:
                    annotation['bbox'] = AutoLabeler._bbox_from_mask(mask)
            
            coco_output['annotations'].append(annotation)
            annotation_id += 1
    
    return coco_output
