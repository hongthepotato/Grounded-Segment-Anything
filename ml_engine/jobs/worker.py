"""
Training Worker for job execution.

This module provides the TrainingWorker class that:
- Polls Redis queue for pending jobs
- Executes training jobs via TeacherTrainer
- Reports progress to Redis with pub/sub
- Handles cancellation and errors gracefully

Usage:
    # Start a worker
    worker = TrainingWorker(
        redis_url="redis://localhost:6379",
        gpu_id=0
    )
    worker.run()  # Blocks until shutdown
"""

import logging
import os
import signal
import socket
import traceback
import uuid
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Any, Optional, List

from ml_engine.jobs.models import Job, JobStatus, JobProgress, WorkerInfo, JobType
from ml_engine.jobs.redis_store import RedisJobStore
from ml_engine.training.teacher_trainer import TeacherTrainer, TrainingCancelledException
from ml_engine.data.manager import DataManager
from ml_engine.inference.auto_labeler import (
    AutoLabeler,
    AutoLabelerConfig,
    visualize_detections
)
from core.config import load_config, merge_configs, save_json
from core.constants import DEFAULT_CONFIGS_DIR

logger = logging.getLogger(__name__)


class TrainingWorker:
    """
    Worker that polls Redis queue and executes training jobs.
    
    Features:
    - Blocking poll on Redis queue
    - Progress reporting to Redis
    - Graceful cancellation via cancel_check
    - Heartbeat for worker health monitoring
    - Handles multiple job types (teacher training, distillation)
    
    Example:
        >>> worker = TrainingWorker(redis_url="redis://localhost:6379")
        >>> worker.run()  # Blocks until SIGTERM/SIGINT
    """

    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL = 10
    # Queue poll timeout in seconds
    POLL_TIMEOUT = 5

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        gpu_id: int = 0,
        worker_id: Optional[str] = None
    ):
        """
        Initialize worker.

        Args:
            redis_url: Redis connection URL
            gpu_id: GPU device ID for this worker
            worker_id: Optional worker ID (auto-generated if not provided)
        """
        self.redis_url = redis_url
        self.gpu_id = gpu_id
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

        # Redis store
        self.store = RedisJobStore(redis_url)

        # Current job state
        self.current_job: Optional[Job] = None
        self._cancel_requested = False
        self._shutdown_requested = False

        # Worker info
        self.worker_info = WorkerInfo(
            id=self.worker_id,
            gpu_id=gpu_id,
            hostname=socket.gethostname(),
            status="idle"
        )

        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Worker %s initialized (GPU %d)", self.worker_id, gpu_id)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %s, initiating shutdown...", signum)
        self._shutdown_requested = True

        # If running a job, request cancellation
        if self.current_job:
            self._cancel_requested = True

    def run(self):
        """
        Main worker loop.
        
        Polls Redis queue for jobs and executes them.
        Blocks until shutdown signal received.
        """
        logger.info("Worker %s starting main loop", self.worker_id)

        # Register worker
        self.store.register_worker(self.worker_info)

        try:
            while not self._shutdown_requested:
                # Update heartbeat
                self.store.update_worker_heartbeat(self.worker_id)

                # Poll for job
                job_id = self.store.dequeue_job(timeout=self.POLL_TIMEOUT)

                if job_id is None:
                    # No job available, continue polling
                    continue

                # Load job details
                job = self.store.get_job(job_id)
                if job is None:
                    logger.warning("Job %s not found in store", job_id[:8])
                    continue

                # Check if job was already cancelled
                if job.status == JobStatus.CANCELLED:
                    logger.info("Skipping cancelled job %s", job_id[:8])
                    continue

                # Execute job
                self._execute_job(job)

        finally:
            # Unregister worker
            self.store.update_worker_status(self.worker_id, "offline")
            self.store.unregister_worker(self.worker_id)
            self.store.close()
            logger.info("Worker %s shut down", self.worker_id)

    def _execute_job(self, job: Job):
        """
        Execute a training job.

        Args:
            job: Job to execute
        """
        self.current_job = job
        self._cancel_requested = False

        logger.info("=" * 60)
        logger.info("Starting job %s (%s)", job.id[:8], job.type)
        logger.info("=" * 60)

        # Update job status to RUNNING
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.worker_id = self.worker_id

        self.store.update_job(
            job.id,
            status=JobStatus.RUNNING,
            started_at=job.started_at,
            worker_id=self.worker_id
        )

        # Update worker status
        self.store.update_worker_status(self.worker_id, "busy", job.id)

        # Publish job started event
        self.store.publish_event(job.id, {
            "type": "job_started",
            "job_id": job.id,
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Route to appropriate trainer
            if job.type == JobType.TEACHER_TRAINING.value:
                self._run_teacher_training(job)
            elif job.type == JobType.STUDENT_DISTILLATION.value:
                self._run_student_distillation(job)
            elif job.type == JobType.AUTO_LABEL.value:
                self._run_auto_label(job)
            else:
                raise ValueError(f"Unknown job type: {job.type}")

            # Job completed successfully
            self._complete_job(job)

        except TrainingCancelledException:
            self._cancel_job(job)

        except Exception as e:
            self._fail_job(job, e)

        finally:
            self.current_job = None
            self.store.update_worker_status(self.worker_id, "idle")

    def _run_teacher_training(self, job: Job):
        """
        Run teacher model training.
        
        Builds complete config by:
        1. Loading shared defaults from teacher_training.yaml
        2. Loading model-specific configs based on dataset
        3. Merging with user-provided overrides from job config
        
        Args:
            job: Job with teacher training config
        """
        job_config = job.config

        # Extract paths from config
        data_path = job_config.get("data_path")
        image_dir = job_config.get("image_dir")
        output_dir = job.output_dir or f"experiments/{job.id[:8]}"

        if not data_path:
            raise ValueError("data_path required in job config")

        # Create DataManager
        split_config = job_config.get("split_config", {"train": 0.7, "val": 0.15, "test": 0.15})
        data_manager = DataManager(
            data_path=data_path,
            image_dir=image_dir,
            split_config=split_config,
            auto_preprocess=True
        )

        # Build complete config like CLI does
        config = self._build_teacher_config(data_manager, job_config)

        # Create trainer with callbacks
        trainer = TeacherTrainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=config,
            progress_callback=lambda p: self._on_progress(job.id, p),
            cancel_check=lambda: self._cancel_requested
        )

        # Run training
        trainer.train()

    def _build_teacher_config(
        self,
        data_manager: DataManager,
        job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete teacher training config from defaults + job overrides.
        
        Mirrors the config building logic in cli/train_teacher.py to ensure
        consistent behavior between CLI and API.
        
        Args:
            data_manager: DataManager with dataset info
            job_config: User-provided job configuration
            
        Returns:
            Complete training config with all required keys
        """
        # Step 1: Load shared training defaults
        shared_config_path = DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'
        shared_config = load_config(str(shared_config_path))
        logger.info("Loaded shared training config from %s", shared_config_path)

        # Step 2: Load model-specific configs based on dataset
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

        # Step 3: Build base config (same structure as CLI)
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

        # Step 4: Merge user overrides from job config
        user_overrides = job_config.get("training", {})
        if user_overrides:
            config = merge_configs(config, user_overrides)
            logger.info("Applied user config overrides")

        return config

    def _run_student_distillation(self, job: Job):
        """
        Run student model distillation.
        
        Args:
            job: Job with distillation config
        """
        # TODO: Implement when StudentDistiller is ready
        raise NotImplementedError("Student distillation not yet implemented")

    def _run_auto_label(self, job: Job):
        """
        Run auto-labeling job using Grounding DINO + MobileSAM.
        
        Args:
            job: Job with auto-labeling config containing:
                - image_dir: Path to images directory
                - classes: List of class names to detect
                - output_mode: "boxes", "masks", or "both"
                - box_threshold: Detection confidence threshold (default: 0.5)
                - text_threshold: Text threshold (default: 0.5)
                - nms_threshold: NMS threshold (default: 0.7)
        """
        config = job.config

        # Extract required config
        image_dir = config.get("image_dir")
        classes = config.get("classes", [])

        if not image_dir:
            raise ValueError("image_dir required in job config")
        if not classes:
            raise ValueError("classes required in job config")

        # Optional config with defaults
        output_mode = config.get("output_mode", "boxes")
        box_threshold = config.get("box_threshold", 0.5)
        text_threshold = config.get("text_threshold", 0.5)
        nms_threshold = config.get("nms_threshold", 0.7)

        # Setup output directory
        output_dir = Path(job.output_dir or f"data/autolabel/{job.id[:8]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Collect image paths
        image_paths = self._collect_image_paths(image_dir)
        if not image_paths:
            raise ValueError(f"No images found in: {image_dir}")

        logger.info("Auto-labeling %d images with classes: %s", len(image_paths), classes)

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

        # Process images with progress reporting
        results = []
        annotation_count = 0

        for i, image_path in enumerate(image_paths):
            # Check for cancellation
            if self._cancel_requested:
                raise TrainingCancelledException("Auto-labeling cancelled")

            # Process single image
            try:
                result = labeler.label_single_image(image_path, classes)
                results.append(result)
                annotation_count += len(result.get('class_ids', []))

                # Generate visualization
                show_boxes = output_mode in ("boxes", "both")
                show_masks = output_mode in ("masks", "both")
                
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
                
            except Exception as e:
                logger.warning("Failed to process %s: %s", image_path, e)
                continue
            
            # Report progress
            self._on_progress(job.id, {
                "current_step": i + 1,
                "total_steps": len(image_paths),
                "current_epoch": 0,
                "total_epochs": 1,
                "message": f"Processing {Path(image_path).name}",
                "metrics": {
                    "images_processed": i + 1,
                    "annotations_found": annotation_count
                }
            })
        
        # Build COCO output
        coco_output = self._build_coco_output(results, classes, output_mode)
        
        # Save annotations
        annotations_path = output_dir / "annotations.json"
        save_json(coco_output, str(annotations_path))
        
        logger.info("Auto-labeling complete: %d images, %d annotations",
                   len(coco_output['images']), len(coco_output['annotations']))
        logger.info("Results saved to: %s", output_dir)

    def _collect_image_paths(
        self,
        image_dir: str,
        extensions: List[str] = None
    ) -> List[str]:
        """
        Collect image paths from directory.
        
        Args:
            image_dir: Path to images directory
            extensions: List of valid extensions (default: jpg, jpeg, png, bmp)
            
        Returns:
            Sorted list of image file paths
        """
        if extensions is None:
            extensions = ["jpg", "jpeg", "png", "bmp"]

        path = Path(image_dir)
        if not path.exists():
            return []

        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(str(path / f"*.{ext}")))
            image_paths.extend(glob(str(path / f"*.{ext.upper()}")))

        return sorted(set(image_paths))

    def _build_coco_output(
        self,
        results: List[Dict[str, Any]],
        classes: List[str],
        output_mode: str
    ) -> Dict[str, Any]:
        """
        Build COCO-format annotations from labeling results.
        
        Args:
            results: List of results from label_single_image
            classes: List of class names
            output_mode: "boxes", "masks", or "both"
            
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

    def _on_progress(self, job_id: str, progress_info: Dict[str, Any]):
        """
        Progress callback - updates Redis and publishes event.
        
        Args:
            job_id: Job ID
            progress_info: Progress information from trainer
        """
        # Check if cancellation was requested via Redis
        job = self.store.get_job(job_id)
        if job and job.status == JobStatus.CANCELLING:
            self._cancel_requested = True

        # Create progress object
        progress = JobProgress(
            current_epoch=progress_info.get("current_epoch", 0),
            total_epochs=progress_info.get("total_epochs", 0),
            current_step=progress_info.get("current_step", 0),
            total_steps=progress_info.get("total_steps", 0),
            metrics=progress_info.get("metrics", progress_info.get("train_metrics", {})),
            message=progress_info.get("message", "")
        )

        # Update job progress in Redis
        self.store.update_job(job_id, progress=progress)

        # Publish progress event
        self.store.publish_event(job_id, {
            "type": "progress",
            "job_id": job_id,
            "progress": progress.to_dict(),
            "timestamp": datetime.now().isoformat()
        })

        # Update heartbeat
        self.store.update_worker_heartbeat(self.worker_id)

        logger.debug("Progress: epoch %d/%d, step %d/%d",
                    progress.current_epoch, progress.total_epochs,
                    progress.current_step, progress.total_steps)

    def _complete_job(self, job: Job):
        """Mark job as completed."""
        logger.info("Job %s completed successfully", job.id[:8])

        self.store.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            finished_at=datetime.now()
        )

        self.store.publish_event(job.id, {
            "type": "job_completed",
            "job_id": job.id,
            "output_dir": job.output_dir,
            "timestamp": datetime.now().isoformat()
        })

    def _cancel_job(self, job: Job):
        """Mark job as cancelled."""
        logger.info("Job %s cancelled", job.id[:8])

        self.store.update_job(
            job.id,
            status=JobStatus.CANCELLED,
            finished_at=datetime.now()
        )

        self.store.publish_event(job.id, {
            "type": "job_cancelled",
            "job_id": job.id,
            "timestamp": datetime.now().isoformat()
        })

    def _fail_job(self, job: Job, error: Exception):
        """Mark job as failed with error message."""
        error_msg = f"{type(error).__name__}: {str(error)}"
        logger.error("Job %s failed: %s", job.id[:8], error_msg)
        logger.debug("Traceback:\n%s", traceback.format_exc())

        self.store.update_job(
            job.id,
            status=JobStatus.FAILED,
            finished_at=datetime.now(),
            error_message=error_msg
        )

        self.store.publish_event(job.id, {
            "type": "job_failed",
            "job_id": job.id,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        })


def main():
    """Entry point for worker process."""
    import argparse

    parser = argparse.ArgumentParser(description="Training Worker")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                       help="Redis connection URL")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--worker-id", default=None,
                       help="Worker ID (auto-generated if not provided)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run worker
    worker = TrainingWorker(
        redis_url=args.redis_url,
        gpu_id=args.gpu,
        worker_id=args.worker_id
    )
    worker.run()


if __name__ == "__main__":
    main()
