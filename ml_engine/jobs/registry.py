"""
Job handler registry.

Maps job type strings to their handler classes.
Adding a new job type only requires adding an entry here.
"""

from typing import Dict, Type

from ml_engine.jobs.handlers.base import JobHandler
from ml_engine.jobs.handlers.teacher import TeacherTrainingHandler
from ml_engine.jobs.handlers.auto_label import AutoLabelHandler


# Registry of job type -> handler class
JOB_HANDLERS: Dict[str, Type[JobHandler]] = {
    "teacher_training": TeacherTrainingHandler,
    "auto_label": AutoLabelHandler,
}


def get_handler(job_type: str) -> JobHandler:
    """
    Get a handler instance for the given job type.
    
    Args:
        job_type: Job type string (e.g., "teacher_training", "auto_label")
        
    Returns:
        Instantiated handler for the job type
        
    Raises:
        ValueError: If job_type is not registered
    """
    handler_cls = JOB_HANDLERS.get(job_type)
    if handler_cls is None:
        available = ", ".join(JOB_HANDLERS.keys())
        raise ValueError(f"Unknown job type: {job_type}. Available: {available}")
    return handler_cls()
