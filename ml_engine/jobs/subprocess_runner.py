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
        # Use handler registry to dispatch job
        from ml_engine.jobs.registry import get_handler
        from ml_engine.jobs.handlers.base import TrainingCancelledError

        handler = get_handler(job_type)
        handler.run(
            job_config=job_config,
            output_dir=output_dir,
            progress_queue=progress_queue,
            cancel_event=cancel_event,
        )

        # Success
        result_queue.put({'success': True, 'output_dir': output_dir})
        sub_logger.info("Job completed successfully")
        sys.exit(EXIT_SUCCESS)

    except TrainingCancelledError:
        result_queue.put({'cancelled': True})
        sub_logger.info("Job cancelled by user")
        sys.exit(EXIT_CANCELLED)

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        result_queue.put({'error': error_msg, 'traceback': traceback.format_exc()})
        sub_logger.error("Job failed: %s", error_msg)
        sub_logger.debug("Traceback:\n%s", traceback.format_exc())
        sys.exit(EXIT_FAILED)
