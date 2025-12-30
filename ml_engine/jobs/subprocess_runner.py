"""
Training Subprocess Runner for isolated job execution.

This module provides process isolation for training jobs:
- Each training job runs in a separate subprocess
- Process termination guarantees complete resource cleanup (GPU, memory, files)
- No manual cleanup code needed - OS handles everything

Why subprocess isolation?
1. GPU memory: CUDA driver releases all GPU memory when process exits
2. DataLoader workers: All child processes terminated with parent
3. File handles: OS closes all file descriptors on process exit
4. Reliability: 100% cleanup guaranteed by OS, not by cleanup code

Usage:
    runner = TrainingSubprocess(job, config, gpu_id)
    runner.start()
    
    # Monitor progress
    while runner.is_alive():
        progress = runner.get_progress()
        if progress:
            forward_to_redis(progress)
    
    # Cancel if needed
    runner.cancel()  # Kills process, all resources freed automatically
"""

import sys
import logging
import multiprocessing as mp
import os
import signal
import queue
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Exit codes for subprocess
EXIT_SUCCESS = 0
EXIT_CANCELLED = 10
EXIT_FAILED = 1


@dataclass
class SubprocessResult:
    """Result from training subprocess."""
    success: bool
    cancelled: bool
    error_message: Optional[str] = None
    output_dir: Optional[str] = None


class TrainingSubprocess:
    """
    Runs training in an isolated subprocess.
    
    Process isolation guarantees:
    - All GPU memory freed on process exit
    - All DataLoader workers terminated
    - All file handles closed
    - No manual cleanup required
    
    Example:
        >>> runner = TrainingSubprocess(job, config, gpu_id=0)
        >>> runner.start()
        >>> 
        >>> while runner.is_alive():
        >>>     if progress := runner.get_progress():
        >>>         report_progress(progress)
        >>>     time.sleep(0.1)
        >>> 
        >>> result = runner.get_result()
        >>> if result.success:
        >>>     mark_completed()
        >>> elif result.cancelled:
        >>>     mark_cancelled()
        >>> else:
        >>>     mark_failed(result.error_message)
    """

    # Timeouts for graceful shutdown sequence
    GRACEFUL_TIMEOUT = 5.0   # Wait for cancel_event to be checked
    SIGTERM_TIMEOUT = 2.0    # Wait after SIGTERM before SIGKILL

    def __init__(
        self,
        job_id: str,
        job_type: str,
        job_config: Dict[str, Any],
        output_dir: str,
        gpu_id: int = 0
    ):
        """
        Initialize training subprocess wrapper.
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job (teacher_training, student_distillation)
            job_config: Job configuration dictionary
            output_dir: Output directory for training artifacts
            gpu_id: GPU device ID for this training
        """
        self.job_id = job_id
        self.job_type = job_type
        self.job_config = job_config
        self.output_dir = output_dir
        self.gpu_id = gpu_id

        # IPC primitives (created fresh each time, not shared)
        self._process: Optional[mp.Process] = None
        self._progress_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._cancel_event: Optional[mp.Event] = None

        # Cached result
        self._result: Optional[SubprocessResult] = None

    def start(self):
        """
        Spawn training subprocess.
        
        The subprocess will:
        1. Set CUDA_VISIBLE_DEVICES to gpu_id
        2. Import and run TeacherTrainer
        3. Report progress via queue
        4. Check cancel_event periodically
        """
        # Create IPC primitives
        ctx = mp.get_context('spawn')  # Must use spawn for CUDA
        self._progress_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._cancel_event = ctx.Event()

        # Spawn subprocess
        self._process = ctx.Process(
            target=_training_entry_point,
            args=(
                self.job_id,
                self.job_type,
                self.job_config,
                self.output_dir,
                self.gpu_id,
                self._progress_queue,
                self._result_queue,
                self._cancel_event,
            ),
            daemon=False  # Not daemon - we want to wait for it
        )
        self._process.start()

        logger.info(
            "Spawned training subprocess (pid=%d, gpu=%d) for job %s",
            self._process.pid, self.gpu_id, self.job_id[:8]
        )

    def is_alive(self) -> bool:
        """Check if subprocess is still running."""
        return self._process is not None and self._process.is_alive()

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get progress update from subprocess (non-blocking).
        
        Returns:
            Progress dict or None if no update available
        """
        if self._progress_queue is None:
            return None

        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None

    def cancel(self) -> bool:
        """
        Cancel training subprocess.
        
        Shutdown sequence:
        1. Set cancel_event (graceful - wait 5s for trainer to check)
        2. Send SIGTERM (allow cleanup handlers)
        3. Send SIGKILL (force kill)
        
        Returns:
            True if process was terminated, False if not running
        """
        if self._process is None or not self._process.is_alive():
            logger.info("Cancel called but process not running")
            return False

        pid = self._process.pid
        logger.info("Cancelling training subprocess (pid=%d)", pid)

        # Step 1: Set cancel event (graceful)
        if self._cancel_event:
            self._cancel_event.set()

        # Wait for graceful exit
        self._process.join(timeout=self.GRACEFUL_TIMEOUT)
        if not self._process.is_alive():
            logger.info("Process exited gracefully after cancel event")
            return True

        # Step 2: SIGTERM
        logger.warning("Process did not exit gracefully, sending SIGTERM")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.info("Process already terminated")
            return True

        self._process.join(timeout=self.SIGTERM_TIMEOUT)
        if not self._process.is_alive():
            logger.info("Process exited after SIGTERM")
            return True

        # Step 3: SIGKILL (nuclear option)
        logger.warning("Process did not respond to SIGTERM, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info("Process already terminated")
            return True

        self._process.join(timeout=1.0)

        if self._process.is_alive():
            logger.error("Process still alive after SIGKILL!")
            return False

        logger.info("Process terminated after SIGKILL")
        return True

    def get_result(self) -> SubprocessResult:
        """
        Get the result of the training subprocess.
        
        Must be called after process has exited.
        
        Returns:
            SubprocessResult with success/cancelled/error status
        """
        if self._result is not None:
            return self._result

        if self._process is None:
            return SubprocessResult(
                success=False,
                cancelled=False,
                error_message="Process never started"
            )

        # Make sure process has finished
        if self._process.is_alive():
            self._process.join(timeout=1.0)

        exit_code = self._process.exitcode

        # Try to get result from queue
        result_from_queue = None
        if self._result_queue:
            try:
                result_from_queue = self._result_queue.get_nowait()
            except queue.Empty:
                pass

        # Determine result based on exit code and queue
        if exit_code == EXIT_SUCCESS:
            self._result = SubprocessResult(
                success=True,
                cancelled=False,
                output_dir=result_from_queue.get('output_dir') if result_from_queue else self.output_dir
            )
        elif exit_code == EXIT_CANCELLED:
            self._result = SubprocessResult(
                success=False,
                cancelled=True
            )
        elif exit_code is None:
            # Process was killed externally
            self._result = SubprocessResult(
                success=False,
                cancelled=True,
                error_message="Process was killed"
            )
        else:
            error_msg = "Unknown error"
            if result_from_queue and 'error' in result_from_queue:
                error_msg = result_from_queue['error']
            elif exit_code < 0:
                # Negative exit code = killed by signal
                error_msg = f"Process killed by signal {-exit_code}"
            else:
                error_msg = f"Process exited with code {exit_code}"

            self._result = SubprocessResult(
                success=False,
                cancelled=False,
                error_message=error_msg
            )

        return self._result

    def cleanup(self):
        """
        Clean up IPC resources.
        
        Should be called after process has exited.
        """
        # Drain and close queues
        for q in [self._progress_queue, self._result_queue]:
            if q:
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self._progress_queue = None
        self._result_queue = None
        self._cancel_event = None
        self._process = None


def _training_entry_point(
    job_id: str,
    job_type: str,
    job_config: Dict[str, Any],
    output_dir: str,
    gpu_id: int,
    progress_queue: mp.Queue,
    result_queue: mp.Queue,
    cancel_event: mp.Event,
):
    """
    Entry point for training subprocess.
    
    This function runs in an isolated process. All resources allocated here
    (GPU memory, file handles, child processes) are automatically freed
    when this process exits.
    
    Args:
        job_id: Job identifier
        job_type: Type of training job
        job_config: Training configuration
        output_dir: Output directory
        gpu_id: GPU device ID
        progress_queue: Queue to send progress updates
        result_queue: Queue to send final result
        cancel_event: Event to check for cancellation
    """
    # CRITICAL: Set up sys.path BEFORE any imports
    # Subprocess with 'spawn' starts fresh, doesn't inherit sys.path
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    deps_segment_anything = project_root / "deps" / "segment_anything"
    deps_groundingdino = project_root / "GroundingDINO"
    deps_efficientsam = project_root / "EfficientSAM"

    for path in [str(project_root), str(deps_segment_anything), str(deps_groundingdino), str(deps_efficientsam)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE any torch imports
    # This ensures PyTorch only sees the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup logging for subprocess
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [Subprocess-{job_id[:8]}] - %(levelname)s - %(message)s"
    )
    sub_logger = logging.getLogger(__name__)

    sub_logger.info("Training subprocess started (pid=%d, gpu=%d)", os.getpid(), gpu_id)
    sub_logger.info("CUDA_VISIBLE_DEVICES set to: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    # Verify GPU mapping after torch import
    import torch
    if torch.cuda.is_available():
        sub_logger.info("PyTorch sees %d GPU(s)", torch.cuda.device_count())
        sub_logger.info("PyTorch cuda:0 maps to physical GPU %d (%s)",
                       gpu_id, torch.cuda.get_device_name(0))
    else:
        sub_logger.warning("CUDA not available in subprocess!")

    try:
        _run_training_job(
            job_type=job_type,
            job_config=job_config,
            output_dir=output_dir,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
        )

        # Success
        result_queue.put({'success': True, 'output_dir': output_dir})
        sub_logger.info("Training completed successfully")
        sys.exit(EXIT_SUCCESS)

    except TrainingCancelledError:
        result_queue.put({'cancelled': True})
        sub_logger.info("Training cancelled by user")
        sys.exit(EXIT_CANCELLED)

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        result_queue.put({'error': error_msg, 'traceback': traceback.format_exc()})
        sub_logger.error("Training failed: %s", error_msg)
        sub_logger.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(EXIT_FAILED)


class TrainingCancelledError(Exception):
    """Raised when training is cancelled via cancel_event."""
    pass


def _run_training_job(
    job_type: str,
    job_config: Dict[str, Any],
    output_dir: str,
    progress_queue: mp.Queue,
    cancel_event: mp.Event,
):
    """
    Run the actual training job.
    
    This is separated from entry_point to allow different job types.
    """
    from ml_engine.training.teacher_trainer import TeacherTrainer, TrainingCancelledException
    from ml_engine.data.manager import DataManager
    from core.constants import transform_image_path
    from pathlib import Path

    if job_type == "teacher_training":
        # Extract paths from config
        data_path_raw = job_config.get("data_path")
        data_path = transform_image_path(data_path_raw) if data_path_raw else None
        image_paths = job_config.get("image_paths", [])

        if not data_path:
            raise ValueError("data_path required in job config")
        if not image_paths:
            raise ValueError("image_paths required in job config")

        # Create DataManager
        # Note: Normalization (bbox from masks, etc.) is always applied during loading
        split_config = job_config.get("split_config", {"train": 0.7, "val": 0.15, "test": 0.15})
        data_manager = DataManager(
            data_path=data_path,
            image_paths=image_paths,
            split_config=split_config
        )

        # Build config
        config = _build_teacher_config(data_manager, job_config)

        # Progress callback that sends to queue
        def progress_callback(progress_info: Dict[str, Any]):
            try:
                progress_queue.put_nowait(progress_info)
            except queue.Full:
                pass  # Drop if queue is full

        # Cancel check that reads event
        def cancel_check() -> bool:
            return cancel_event.is_set()

        # Create and run trainer
        trainer = TeacherTrainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )

        try:
            trainer.train()
        except TrainingCancelledException as e:
            raise TrainingCancelledError("Training cancelled by user") from e

    elif job_type == "student_distillation":
        raise NotImplementedError("Student distillation not yet implemented")

    elif job_type == "auto_label":
        image_paths = job_config.get("image_paths", [])
        if not image_paths:
            raise ValueError("image_paths required in job config")

        actual_paths = []
        for image_path in image_paths:
            actual_path = transform_image_path(image_path)
            if not Path(actual_path).exists():
                raise ValueError(f"Image path not found: {image_path}")
            actual_paths.append(actual_path)
        job_config['image_paths'] = actual_paths

        classes = job_config.get("classes", [])
        if not classes:
            raise ValueError("classes required in job config")

        _run_auto_label_job(job_config, output_dir, progress_queue, cancel_event)

    else:
        raise ValueError(f"Unknown job type: {job_type}")


def _build_teacher_config(
    data_manager,
    job_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build complete teacher training config from defaults + job overrides.
    
    Mirrors the config building logic in worker.py.
    """
    from core.config import load_config, merge_configs
    from core.constants import DEFAULT_CONFIGS_DIR

    logger = logging.getLogger(__name__)

    # Load shared training defaults
    shared_config_path = DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'
    shared_config = load_config(str(shared_config_path))
    logger.info("Loaded shared training config from %s", shared_config_path)

    # Load model-specific configs based on dataset
    dataset_info = data_manager.get_dataset_info()
    required_models = data_manager.get_required_models()
    logger.info("Required teacher models: %s", required_models)

    model_configs = {}
    if 'grounding_dino' in required_models:
        dino_config_path = DEFAULT_CONFIGS_DIR / 'teacher_grounding_dino_lora.yaml'
        model_configs['grounding_dino'] = load_config(str(dino_config_path))
        logger.info("Loaded Grounding DINO config")

    if 'sam' in required_models:
        sam_config_path = DEFAULT_CONFIGS_DIR / 'teacher_sam_lora.yaml'
        model_configs['sam'] = load_config(str(sam_config_path))
        logger.info("Loaded SAM config")

    if not model_configs:
        raise ValueError("No models to train! Dataset has no valid annotations.")

    # Build base config
    config = {
        **shared_config['training'],
        'num_classes': dataset_info['num_classes'],
        'class_names': list(dataset_info['class_mapping'].values()),
        'class_mapping': dataset_info['class_mapping'],
        'augmentation': shared_config.get('augmentation'),
        'evaluation': shared_config.get('evaluation'),
        'checkpointing': shared_config.get('checkpointing'),
        'models': model_configs
    }

    # Merge user overrides
    user_overrides = job_config.get("training", {})
    if user_overrides:
        config = merge_configs(config, user_overrides)
        logger.info("Applied user config overrides")

    return config


def _run_auto_label_job(
    job_config: Dict[str, Any],
    output_dir: str,
    progress_queue: mp.Queue,
    cancel_event: mp.Event,
):
    """
    Run auto-labeling job using Grounding DINO + MobileSAM.
    
    Uses batch inference for detection to improve throughput (2-4x speedup).
    
    Args:
        job_config: Job config containing:
            - image_paths: List of image paths to process
            - classes: List of class names to detect
            - output_mode: "boxes", "masks", or "both"
            - box_threshold: Detection confidence threshold (default: 0.5)
            - text_threshold: Text threshold (default: 0.5)
            - nms_threshold: NMS threshold (default: 0.7)
            - batch_size: Batch size for inference (default: 8)
        output_dir: Output directory for results
        progress_queue: Queue to send progress updates
        cancel_event: Event to check for cancellation
    """
    from pathlib import Path
    from ml_engine.inference.auto_labeler import (
        AutoLabeler,
        AutoLabelerConfig,
        visualize_detections
    )
    from core.config import save_json

    sub_logger = logging.getLogger(__name__)

    # Extract required config
    image_paths = job_config.get("image_paths", [])
    classes = job_config.get("classes", [])

    # Optional config with defaults
    output_mode = job_config.get("output_mode", "boxes")
    box_threshold = job_config.get("box_threshold", 0.5)
    text_threshold = job_config.get("text_threshold", 0.5)
    nms_threshold = job_config.get("nms_threshold", 0.7)
    batch_size = job_config.get("batch_size", 1)  # Default=1 for best performance

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    sub_logger.info("Auto-labeling %d images with classes: %s (batch_size=%d)", 
                   len(image_paths), classes, batch_size)

    # Create AutoLabeler config
    labeler_config = AutoLabelerConfig(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        output_mode=output_mode,
        device="cuda"
    )

    # Create labeler
    labeler = AutoLabeler(labeler_config)

    # Process images in batches with progress reporting
    results = []
    annotation_count = 0
    show_boxes = output_mode in ("boxes", "both")
    show_masks = output_mode in ("masks", "both")
    total_images = len(image_paths)

    for batch_start in range(0, total_images, batch_size):
        # Check for cancellation between batches
        if cancel_event.is_set():
            raise TrainingCancelledError("Auto-labeling cancelled by user")

        batch_end = min(batch_start + batch_size, total_images)
        batch_paths = image_paths[batch_start:batch_end]

        try:
            # Batch process detection (GroundingDINO batched, SAM per-image)
            batch_results = labeler.label_batch_images(
                image_paths=batch_paths,
                class_prompts=classes,
                batch_size=len(batch_paths)  # Process this chunk in one batch
            )

            # Process each result in the batch
            for i, (image_path, result) in enumerate(zip(batch_paths, batch_results)):
                results.append(result)
                annotation_count += len(result.get('class_ids', []))

                # Generate visualization
                try:
                    viz_filename = Path(image_path).stem + "_viz.jpg"
                    viz_path = str(viz_dir / viz_filename)
                    visualize_detections(
                        image_path=image_path,
                        result=result,
                        class_prompts=classes,
                        output_path=viz_path,
                        show_boxes=show_boxes,
                        show_masks=show_masks
                    )
                except Exception as viz_e:
                    sub_logger.warning("Failed to visualize %s: %s", image_path, viz_e)

                # Report progress for each image
                current_image = batch_start + i + 1
                try:
                    progress_queue.put_nowait({
                        "current_step": current_image,
                        "total_steps": total_images,
                        "current_epoch": 0,
                        "total_epochs": 1,
                        "message": f"Processing {Path(image_path).name}",
                        "metrics": {
                            "images_processed": current_image,
                            "annotations_found": annotation_count,
                            "batch_size": batch_size
                        }
                    })
                except queue.Full:
                    pass  # Drop if queue is full

        except TrainingCancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            sub_logger.warning("Failed to process batch starting at %d: %s", batch_start, e)
            # Continue with next batch instead of failing completely
            continue

    # Build COCO output
    coco_output = _build_coco_output(results, classes, output_mode, labeler)

    # Save annotations
    annotations_path = output_path / "annotations.json"
    save_json(coco_output, str(annotations_path))

    sub_logger.info("Auto-labeling complete: %d images, %d annotations",
                   len(coco_output['images']), len(coco_output['annotations']))
    sub_logger.info("Results saved to: %s", output_dir)


def _build_coco_output(
    results: list,
    classes: list,
    output_mode: str,
    labeler
) -> Dict[str, Any]:
    """
    Build COCO-format annotations from labeling results.
    
    Args:
        results: List of results from label_single_image
        classes: List of class names
        output_mode: "boxes", "masks", or "both"
        labeler: AutoLabeler instance for utility methods
        
    Returns:
        COCO-format dictionary
    """
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': i, 'name': name}
            for i, name in enumerate(classes)
        ]
    }

    annotation_id = 1

    for image_id, result in enumerate(results, start=1):
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
        num_detections = len(result.get('class_ids', []))

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
            if output_mode in ("boxes", "both") and i < len(boxes):
                box = boxes[i]
                annotation['bbox'] = box
                annotation['area'] = box[2] * box[3]

            # Add segmentation if available
            if output_mode in ("masks", "both") and i < len(masks):
                from ml_engine.inference.auto_labeler import AutoLabeler
                mask = masks[i]
                segmentation = AutoLabeler._mask_to_polygon(mask)
                annotation['segmentation'] = segmentation
                if mask is not None and hasattr(mask, 'sum'):
                    annotation['area'] = float(mask.sum())

                # For masks-only mode, generate bbox from mask
                if output_mode == "masks" and 'bbox' not in annotation:
                    annotation['bbox'] = AutoLabeler._bbox_from_mask(mask)

            coco_output['annotations'].append(annotation)
            annotation_id += 1

    return coco_output
