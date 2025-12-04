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
from pathlib import Path
from typing import Dict, Any, Optional

from ml_engine.jobs.models import Job, JobStatus, JobProgress, WorkerInfo, JobType
from ml_engine.jobs.redis_store import RedisJobStore
from ml_engine.training.teacher_trainer import TeacherTrainer, TrainingCancelledException
from ml_engine.data.manager import DataManager

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
        job.started_at = datetime.utcnow()
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
            "timestamp": datetime.utcnow().isoformat()
        })
        
        try:
            # Route to appropriate trainer
            if job.type == JobType.TEACHER_TRAINING.value:
                self._run_teacher_training(job)
            elif job.type == JobType.STUDENT_DISTILLATION.value:
                self._run_student_distillation(job)
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
        
        Args:
            job: Job with teacher training config
        """
        config = job.config
        
        # Extract paths from config
        data_path = config.get("data_path")
        image_dir = config.get("image_dir")
        output_dir = job.output_dir or f"experiments/{job.id[:8]}"
        
        if not data_path:
            raise ValueError("data_path required in job config")
        
        # Create DataManager
        split_config = config.get("split_config", {"train": 0.7, "val": 0.15, "test": 0.15})
        data_manager = DataManager(
            data_path=data_path,
            image_dir=image_dir,
            split_config=split_config,
            auto_preprocess=True
        )
        
        # Training config
        training_config = config.get("training", {})
        
        # Create trainer with callbacks
        trainer = TeacherTrainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=training_config,
            progress_callback=lambda p: self._on_progress(job.id, p),
            cancel_check=lambda: self._cancel_requested
        )
        
        # Run training
        trainer.train()
    
    def _run_student_distillation(self, job: Job):
        """
        Run student model distillation.
        
        Args:
            job: Job with distillation config
        """
        # TODO: Implement when StudentDistiller is ready
        raise NotImplementedError("Student distillation not yet implemented")
    
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
            "timestamp": datetime.utcnow().isoformat()
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
            finished_at=datetime.utcnow()
        )
        
        self.store.publish_event(job.id, {
            "type": "job_completed",
            "job_id": job.id,
            "output_dir": job.output_dir,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _cancel_job(self, job: Job):
        """Mark job as cancelled."""
        logger.info("Job %s cancelled", job.id[:8])
        
        self.store.update_job(
            job.id,
            status=JobStatus.CANCELLED,
            finished_at=datetime.utcnow()
        )
        
        self.store.publish_event(job.id, {
            "type": "job_cancelled",
            "job_id": job.id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _fail_job(self, job: Job, error: Exception):
        """Mark job as failed with error message."""
        error_msg = f"{type(error).__name__}: {str(error)}"
        logger.error("Job %s failed: %s", job.id[:8], error_msg)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        
        self.store.update_job(
            job.id,
            status=JobStatus.FAILED,
            finished_at=datetime.utcnow(),
            error_message=error_msg
        )
        
        self.store.publish_event(job.id, {
            "type": "job_failed",
            "job_id": job.id,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
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





