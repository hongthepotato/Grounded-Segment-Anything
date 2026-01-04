"""
Base class for job handlers.

Each job type (teacher_training, auto_label, etc.) implements a handler
that inherits from JobHandler and provides the run() method.
"""

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, Any


class TrainingCancelledError(Exception):
    """Raised when a job is cancelled via cancel_event."""
    pass


class JobHandler(ABC):
    """
    Abstract base class for job handlers.
    
    Each job type implements this interface to run in an isolated subprocess.
    Heavy dependencies (torch, models, etc.) should be imported inside run()
    to ensure they're loaded in the subprocess, not the parent process.
    
    Example:
        class TeacherTrainingHandler(JobHandler):
            def run(self, config, output_dir, progress_queue, cancel_event):
                # Import heavy deps here, inside subprocess
                import torch
                from ml_engine.training.teacher_trainer import TeacherTrainer
                
                # Run training...
    """

    @abstractmethod
    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        """
        Execute the job.
        
        This method runs in an isolated subprocess. All resources allocated here
        (GPU memory, file handles, child processes) are automatically freed
        when the process exits.
        
        Args:
            job_config: Job configuration dictionary
            output_dir: Output directory for job artifacts
            progress_queue: Queue to send progress updates (non-blocking put)
            cancel_event: Event to check for cancellation requests
            
        Raises:
            TrainingCancelledError: If job was cancelled via cancel_event
            ValueError: If job_config is invalid
            Exception: Any other errors during execution
        """
        pass
