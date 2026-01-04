"""
Training Worker for job execution.

This module provides the TrainingWorker class that:
- Polls Redis queue for pending jobs
- Executes training jobs in ISOLATED SUBPROCESSES
- Reports progress to Redis with pub/sub
- Handles cancellation by killing subprocess (guaranteed resource cleanup)

Architecture:
    Worker Process (long-lived, lightweight)
    └── Training Subprocess (short-lived, isolated)
        └── TeacherTrainer (GPU, DataLoaders, etc.)

Why subprocess isolation?
- Process death = OS automatically frees ALL resources
- GPU memory released by CUDA driver on process exit
- DataLoader workers terminated with parent
- No manual cleanup code needed

Usage:
    worker = TrainingWorker(
        redis_url="redis://localhost:6379",
        gpu_id=0
    )
    worker.run()  # Blocks until shutdown
"""

import logging
import signal
import socket
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ml_engine.jobs.models import Job, JobStatus, JobProgress, WorkerInfo
from ml_engine.jobs.redis_store import RedisJobStore
from ml_engine.jobs.subprocess_runner import TrainingSubprocess

logger = logging.getLogger(__name__)


class TrainingWorker:
    """
    Worker that polls Redis queue and executes training jobs in subprocesses.
    
    Key Features:
    - Training runs in isolated subprocess (not in worker process)
    - Cancel = kill subprocess = 100% reliable resource cleanup
    - Worker stays lightweight (only scheduling logic)
    - Heartbeat for worker health monitoring
    
    Example:
        >>> worker = TrainingWorker(redis_url="redis://localhost:6379")
        >>> worker.run()  # Blocks until SIGTERM/SIGINT
    """

    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL = 10
    # Queue poll timeout in seconds
    POLL_TIMEOUT = 5
    # Progress poll interval when job is running
    PROGRESS_POLL_INTERVAL = 0.5
    # Cancel check interval when job is running
    CANCEL_CHECK_INTERVAL = 1.0

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
        self.current_subprocess: Optional[TrainingSubprocess] = None
        self._shutdown_requested = False

        # Worker info
        self.worker_info = WorkerInfo(
            id=self.worker_id,
            gpu_id=gpu_id,
            hostname=socket.gethostname(),
            status="idle"
        )

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Worker %s initialized (GPU %d)", self.worker_id, gpu_id)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %s, initiating shutdown...", signum)
        self._shutdown_requested = True

        # If running a job, cancel the subprocess
        if self.current_subprocess and self.current_subprocess.is_alive():
            logger.info("Cancelling current subprocess due to shutdown signal")
            self.current_subprocess.cancel()

    def run(self):
        """
        Main worker loop.
        
        Polls Redis queue for jobs and executes them in subprocesses.
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

                # Execute job in subprocess
                self._execute_job(job)

        finally:
            # Unregister worker
            self.store.update_worker_status(self.worker_id, "offline")
            self.store.unregister_worker(self.worker_id)
            self.store.close()
            logger.info("Worker %s shut down", self.worker_id)

    def _execute_job(self, job: Job):
        """
        Execute a training job in an isolated subprocess.

        The subprocess:
        1. Allocates all GPU resources
        2. Runs training
        3. Reports progress via IPC queue
        4. On cancel: gets killed, OS frees all resources

        Args:
            job: Job to execute
        """
        self.current_job = job

        logger.info("=" * 60)
        logger.info("Starting job %s (%s) in subprocess", job.id[:8], job.type)
        logger.info("=" * 60)

        # Build output directory
        job_subdir = f"{job.type}_{job.id[:8]}"
        base_dir = job.output_dir or "experiments"
        job.output_dir = f"{base_dir}/{job_subdir}"

        # Update job status to RUNNING
        self.store.update_job(
            job.id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(),
            worker_id=self.worker_id,
            output_dir=job.output_dir
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

        # Create and start subprocess
        subprocess_runner = TrainingSubprocess(
            job_id=job.id,
            job_type=job.type,
            job_config=job.config,
            output_dir=job.output_dir,
            gpu_id=self.gpu_id
        )
        self.current_subprocess = subprocess_runner

        try:
            subprocess_runner.start()
            # Monitor subprocess: forward progress, check for cancellation
            self._monitor_subprocess(job, subprocess_runner)
            # Get result
            result = subprocess_runner.get_result()
            if result.success:
                self._complete_job(job, result.output_dir)
            elif result.cancelled:
                self._cancel_job(job)
            else:
                self._fail_job(job, result.error_message or "Unknown error")

        except Exception as e:
            logger.error("Exception during job execution: %s", e)
            logger.debug("Traceback:\n%s", traceback.format_exc())

            # Make sure subprocess is killed
            if subprocess_runner.is_alive():
                subprocess_runner.cancel()

            self._fail_job(job, str(e))

        finally:
            # Cleanup IPC resources
            subprocess_runner.cleanup()
            self.current_subprocess = None
            self.current_job = None
            self.store.update_worker_status(self.worker_id, "idle")

    def _monitor_subprocess(self, job: Job, subprocess_runner: TrainingSubprocess):
        """
        Monitor running subprocess.
        
        - Forward progress updates to Redis
        - Check for cancellation requests
        - Update heartbeat
        
        Args:
            job: The job being executed
            subprocess_runner: The subprocess wrapper
        """
        last_cancel_check = time.time()
        last_heartbeat = time.time()

        while subprocess_runner.is_alive():
            current_time = time.time()

            # Forward all available progress updates
            while True:
                progress_info = subprocess_runner.get_progress()
                if progress_info is None:
                    break
                self._forward_progress(job.id, progress_info)

            # Check for cancellation request from Redis
            if current_time - last_cancel_check >= self.CANCEL_CHECK_INTERVAL:
                last_cancel_check = current_time
                # Check if shutdown was requested
                if self._shutdown_requested:
                    logger.info("Shutdown requested, cancelling subprocess")
                    subprocess_runner.cancel()
                    break
                # Check Redis for cancel request
                updated_job = self.store.get_job(job.id)
                if updated_job and updated_job.status == JobStatus.CANCELLING:
                    logger.info("Cancel requested for job %s, killing subprocess", job.id[:8])
                    subprocess_runner.cancel()
                    break

            # Update heartbeat
            if current_time - last_heartbeat >= self.HEARTBEAT_INTERVAL:
                last_heartbeat = current_time
                self.store.update_worker_heartbeat(self.worker_id)

            # Brief sleep to avoid busy loop
            time.sleep(self.PROGRESS_POLL_INTERVAL)

        # Drain any remaining progress updates
        while True:
            progress_info = subprocess_runner.get_progress()
            if progress_info is None:
                break
            self._forward_progress(job.id, progress_info)

    def _forward_progress(self, job_id: str, progress_info: Dict[str, Any]):
        """
        Forward progress update from subprocess to Redis.
        
        Args:
            job_id: Job ID
            progress_info: Progress information from subprocess
        """
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

        logger.debug("Progress: epoch %d/%d, step %d/%d",
                    progress.current_epoch, progress.total_epochs,
                    progress.current_step, progress.total_steps)

    def _complete_job(self, job: Job, output_dir: Optional[str] = None):
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
            "output_dir": output_dir or job.output_dir,
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

    def _fail_job(self, job: Job, error_message: str):
        """Mark job as failed with error message."""
        logger.error("Job %s failed: %s", job.id[:8], error_message)

        self.store.update_job(
            job.id,
            status=JobStatus.FAILED,
            finished_at=datetime.now(),
            error_message=error_message
        )

        self.store.publish_event(job.id, {
            "type": "job_failed",
            "job_id": job.id,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })


def main():
    """Entry point for worker process."""
    import argparse
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

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
