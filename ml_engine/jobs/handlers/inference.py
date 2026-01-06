"""
Inference job handler.

Handles the inference job type for running fine-tuned models on new images.
"""

import logging
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Dict, Any, List

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError


class InferenceHandler(JobHandler):
    """
    Handler for inference jobs using fine-tuned models.
    
    Loads LoRA adapters from a completed training job and runs
    inference on new images, producing visualizations and metrics.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        """
        Execute inference job.
        
        Args:
            job_config: Configuration containing:
                - training_job_id: ID of completed training job
                - image_paths: List of image paths to process
                - output_mode: "boxes", "masks", or "both"
                - box_threshold: Detection confidence threshold (default: 0.5)
                - nms_threshold: NMS threshold (default: 0.7)
            output_dir: Directory for inference outputs
            progress_queue: Queue for progress updates
            cancel_event: Cancellation signal
        """
        # Late imports - these load in subprocess, not parent
        import cv2
        import numpy as np
        import torch
        import torchvision.ops
        from ml_engine.inference.visualizer import visualize_detections
        from ml_engine.jobs import get_job_manager
        from ml_engine.jobs.models import JobStatus
        from core.config import save_json
        from core.constants import transform_image_path

        sub_logger = logging.getLogger(__name__)

        # Extract and validate config
        training_job_id = job_config.get("training_job_id")
        if not training_job_id:
            raise ValueError("training_job_id required in job config")

        raw_image_paths = job_config.get("image_paths", [])
        if not raw_image_paths:
            raise ValueError("image_paths required in job config")

        # Get training job info
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        job_manager = get_job_manager(redis_url)
        training_job = job_manager.get_job(training_job_id)

        if training_job is None:
            raise ValueError(f"Training job not found: {training_job_id}")

        if training_job.status != JobStatus.COMPLETED:
            raise ValueError(f"Training job not completed: {training_job.status.value}")

        if not training_job.output_dir:
            raise ValueError("Training job has no output_dir")

        # Get class names from training job
        class_names = training_job.config.get("class_names", [])
        if not class_names:
            # Try to load from evaluation report
            eval_report_path = Path(training_job.output_dir) / "evaluation" / "grounding_dino_report.json"
            if eval_report_path.exists():
                from core.config import load_json
                report = load_json(str(eval_report_path))
                class_names = report.get("class_names", [])

        if not class_names:
            raise ValueError("Could not determine class names from training job")

        sub_logger.info("Using class names from training job: %s", class_names)

        # Transform image paths
        image_paths = []
        for raw_path in raw_image_paths:
            actual_path = transform_image_path(raw_path)
            if not Path(actual_path).exists():
                raise ValueError(f"Image path not found: {raw_path} -> {actual_path}")
            image_paths.append(actual_path)

        sub_logger.info("Transformed %d image paths", len(image_paths))

        # Optional config with defaults
        output_mode = job_config.get("output_mode", "both")
        box_threshold = job_config.get("box_threshold", 0.5)
        nms_threshold = job_config.get("nms_threshold", 0.7)

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        sub_logger.info("Running inference on %d images with fine-tuned model from %s",
                       len(image_paths), training_job_id[:8])

        # Load fine-tuned model
        training_output = Path(training_job.output_dir)
        dino_lora_path = training_output / "teachers" / "grounding_dino_lora_adapters"
        sam_lora_path = training_output / "teachers" / "sam_lora_adapters"

        if not dino_lora_path.exists():
            raise ValueError(f"GroundingDINO LoRA adapters not found: {dino_lora_path}")

        # Determine if we need SAM
        needs_masks = output_mode in ("masks", "both")

        # Load models
        sub_logger.info("Loading fine-tuned GroundingDINO from %s", dino_lora_path)
        from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
        from core.constants import DEFAULT_MODELS_DIR

        dino_base = str(DEFAULT_MODELS_DIR / "pretrained" / "groundingdino_swint_ogc.pth")
        grounding_dino = load_grounding_dino_with_lora(
            base_checkpoint=dino_base,
            lora_adapter_path=str(dino_lora_path),
            merge=True  # Merge for faster inference
        )
        grounding_dino.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grounding_dino = grounding_dino.to(device)

        # Load SAM if needed
        sam_model = None
        sam_predictor = None
        if needs_masks:
            if sam_lora_path.exists():
                sub_logger.info("Loading fine-tuned SAM from %s", sam_lora_path)
                from ml_engine.models.teacher.sam_lora import load_sam_hq_with_lora
                sam_base = str(DEFAULT_MODELS_DIR / "pretrained" / "sam_hq_vit_h.pth")
                sam_model = load_sam_hq_with_lora(
                    base_checkpoint=sam_base,
                    lora_adapter_path=str(sam_lora_path),
                    merge=True
                )
            else:
                sub_logger.info("SAM LoRA not found, using pretrained MobileSAM")
                from ml_engine.inference.segmenters.mobile_sam import MobileSAMSegmenter
                sam_predictor = MobileSAMSegmenter(device=str(device))

        # Track metrics
        show_boxes = output_mode in ("boxes", "both")
        show_masks = output_mode in ("masks", "both")
        total_images = len(image_paths)
        start_time = time.time()
        
        # Metrics accumulators
        all_results = []
        total_detections = 0
        all_confidences = []
        per_class_counts = {name: 0 for name in class_names}
        per_class_confidences = {name: [] for name in class_names}

        # Progress callback
        def send_progress(current: int, message: str):
            if cancel_event.is_set():
                raise TrainingCancelledError("Inference cancelled by user")
            try:
                progress_queue.put_nowait({
                    "current_step": current,
                    "total_steps": total_images,
                    "current_epoch": 0,
                    "total_epochs": 1,
                    "message": message,
                    "metrics": {
                        "images_processed": current,
                        "detections_found": total_detections,
                    }
                })
            except queue.Full:
                pass

        # Process images sequentially
        for i, image_path in enumerate(image_paths):
            send_progress(i, f"Processing {Path(image_path).name}")

            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                sub_logger.warning("Could not load image: %s", image_path)
                all_results.append(self._empty_result(image_path, output_mode))
                continue

            height, width = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Run detection with fine-tuned model
            try:
                # Build text prompt
                caption = ". ".join(class_names) + "."
                
                # Use grounding_dino inference
                from groundingdino.util.inference import predict
                
                boxes, logits, phrases = predict(
                    model=grounding_dino,
                    image=image_bgr,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=box_threshold
                )

                # Map phrases to class indices
                class_ids = []
                for phrase in phrases:
                    phrase_lower = phrase.lower().strip()
                    matched_idx = -1
                    for idx, class_name in enumerate(class_names):
                        if class_name.lower() in phrase_lower or phrase_lower in class_name.lower():
                            matched_idx = idx
                            break
                    class_ids.append(matched_idx if matched_idx >= 0 else 0)

                # Convert boxes to pixel coordinates [x1, y1, x2, y2]
                boxes_xyxy = boxes.clone()
                boxes_xyxy[:, 0] *= width   # x_center -> x1
                boxes_xyxy[:, 1] *= height  # y_center -> y1
                boxes_xyxy[:, 2] *= width   # w -> x2
                boxes_xyxy[:, 3] *= height  # h -> y2
                # Convert from cx,cy,w,h to x1,y1,x2,y2
                boxes_xyxy_final = torch.zeros_like(boxes_xyxy)
                boxes_xyxy_final[:, 0] = boxes_xyxy[:, 0] - boxes_xyxy[:, 2] / 2
                boxes_xyxy_final[:, 1] = boxes_xyxy[:, 1] - boxes_xyxy[:, 3] / 2
                boxes_xyxy_final[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2] / 2
                boxes_xyxy_final[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3] / 2

                # Apply NMS
                if len(boxes_xyxy_final) > 0:
                    nms_idx = torchvision.ops.nms(
                        boxes_xyxy_final,
                        logits,
                        nms_threshold
                    ).cpu().numpy().tolist()
                    
                    boxes_xyxy_final = boxes_xyxy_final[nms_idx].cpu().numpy()
                    logits = logits[nms_idx].cpu().numpy()
                    class_ids = [class_ids[i] for i in nms_idx]
                else:
                    boxes_xyxy_final = np.array([])
                    logits = np.array([])
                    class_ids = []

            except Exception as e:
                sub_logger.warning("Detection failed for %s: %s", image_path, e)
                all_results.append(self._empty_result(image_path, output_mode))
                continue

            # Convert to COCO format [x, y, w, h]
            boxes_coco = []
            for box in boxes_xyxy_final:
                x1, y1, x2, y2 = box
                boxes_coco.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            # Generate masks if needed
            masks = []
            if needs_masks and len(boxes_xyxy_final) > 0:
                if sam_predictor is not None:
                    # Use MobileSAM
                    masks = sam_predictor.segment(image_rgb, boxes_xyxy_final)
                elif sam_model is not None:
                    # Use fine-tuned SAM (TODO: implement proper interface)
                    sub_logger.warning("Fine-tuned SAM inference not yet implemented, skipping masks")

            # Build result
            result = {
                'class_ids': class_ids,
                'scores': logits.tolist() if len(logits) > 0 else [],
                'image_info': {
                    'file_name': os.path.basename(image_path),
                    'width': width,
                    'height': height
                }
            }

            if show_boxes:
                result['boxes'] = boxes_coco
            if show_masks:
                result['masks'] = masks

            all_results.append(result)

            # Update metrics
            num_detections = len(class_ids)
            total_detections += num_detections
            
            for cls_idx, score in zip(class_ids, logits if len(logits) > 0 else []):
                all_confidences.append(float(score))
                if 0 <= cls_idx < len(class_names):
                    class_name = class_names[cls_idx]
                    per_class_counts[class_name] += 1
                    per_class_confidences[class_name].append(float(score))

        # Final progress
        send_progress(total_images, "Generating visualizations...")

        # Generate visualizations
        for image_path, result in zip(image_paths, all_results):
            try:
                viz_filename = Path(image_path).stem + "_viz.jpg"
                viz_path = str(viz_dir / viz_filename)
                visualize_detections(
                    image_path=image_path,
                    result=result,
                    class_prompts=class_names,
                    output_path=viz_path,
                    show_boxes=show_boxes,
                    show_masks=show_masks
                )
            except Exception as viz_e:
                sub_logger.warning("Failed to visualize %s: %s", image_path, viz_e)

        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time

        metrics = {
            "summary": {
                "total_images": total_images,
                "total_detections": total_detections,
                "avg_detections_per_image": total_detections / max(total_images, 1),
                "avg_confidence": float(np.mean(all_confidences)) if all_confidences else 0.0
            },
            "per_class": {},
            "processing": {
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_image": round(total_time / max(total_images, 1), 3)
            },
            "config": {
                "training_job_id": training_job_id,
                "box_threshold": box_threshold,
                "nms_threshold": nms_threshold,
                "output_mode": output_mode
            }
        }

        for class_name in class_names:
            count = per_class_counts[class_name]
            confidences = per_class_confidences[class_name]
            metrics["per_class"][class_name] = {
                "count": count,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0
            }

        # Save metrics
        metrics_path = output_path / "metrics.json"
        save_json(metrics, str(metrics_path))

        sub_logger.info("Inference complete: %d images, %d detections",
                       total_images, total_detections)
        sub_logger.info("Results saved to: %s", output_dir)

    def _empty_result(self, image_path: str, output_mode: str) -> Dict[str, Any]:
        """Create empty result for failed image."""
        result = {
            'class_ids': [],
            'scores': [],
            'image_info': {
                'file_name': os.path.basename(image_path),
                'width': 0,
                'height': 0
            }
        }
        if output_mode in ("boxes", "both"):
            result['boxes'] = []
        if output_mode in ("masks", "both"):
            result['masks'] = []
        return result





