"""
Job data models for training pipeline.

This module defines:
- JobStatus: Enum for job lifecycle states
- JobProgress: Progress tracking during training
- Job: Main job dataclass with serialization
- JobType: Supported job types
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import json
import uuid


class JobStatus(str, Enum):
    """Job lifecycle states."""
    PENDING = "pending"          # In queue, waiting to be picked up
    RUNNING = "running"          # Currently executing on a worker
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Error occurred during execution
    CANCELLED = "cancelled"      # User cancelled, job stopped
    CANCELLING = "cancelling"    # Cancel requested, waiting for graceful stop


class JobType(str, Enum):
    """Supported job types."""
    TEACHER_TRAINING = "teacher_training"
    STUDENT_DISTILLATION = "student_distillation"
    MODEL_OPTIMIZATION = "model_optimization"
    EVALUATION = "evaluation"
    AUTO_LABEL = "auto_label"


@dataclass
class JobProgress:
    """
    Training progress information.

    Attributes:
        current_epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        current_step: Current step within epoch
        total_steps: Total steps per epoch
        metrics: Latest training/validation metrics
        message: Optional status message
    """
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "metrics": self.metrics,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobProgress":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            current_epoch=data.get("current_epoch", 0),
            total_epochs=data.get("total_epochs", 0),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            metrics=data.get("metrics", {}),
            message=data.get("message", "")
        )

    @property
    def epoch_progress(self) -> float:
        """Progress within current epoch (0.0 to 1.0)."""
        if self.total_steps == 0:
            return 0.0
        return self.current_step / self.total_steps

    @property
    def overall_progress(self) -> float:
        """Overall training progress (0.0 to 1.0)."""
        if self.total_epochs == 0:
            return 0.0
        epoch_fraction = 1.0 / self.total_epochs
        completed_epochs = self.current_epoch * epoch_fraction
        current_epoch_progress = self.epoch_progress * epoch_fraction
        return completed_epochs + current_epoch_progress


@dataclass
class Job:
    """
    Training job representation.

    Attributes:
        id: Unique job identifier (UUID)
        type: Job type (teacher_training, distillation, etc.)
        status: Current job status
        config: Training configuration dictionary
        progress: Current training progress
        worker_id: ID of worker executing this job (if running)
        created_at: Job creation timestamp
        started_at: Job start timestamp (when picked up by worker)
        finished_at: Job completion timestamp
        error_message: Error message if job failed
        output_dir: Directory for job outputs (checkpoints, logs)
        priority: Job priority (higher = more urgent)
        tags: Optional tags for filtering/grouping
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = JobType.TEACHER_TRAINING.value
    status: JobStatus = JobStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[JobProgress] = None
    worker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.output_dir is None:
            short_id = self.id[:8]
            self.output_dir = f"experiments/{self.type}_{short_id}"
        # Convert status string to enum if needed
        if isinstance(self.status, str):
            self.status = JobStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for Redis storage.
        
        Handles datetime serialization and nested objects.
        """
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "config": json.dumps(self.config),
            "progress": json.dumps(self.progress.to_dict() if self.progress else None),
            "worker_id": self.worker_id or "",
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "started_at": self.started_at.isoformat() if self.started_at else "",
            "finished_at": self.finished_at.isoformat() if self.finished_at else "",
            "error_message": self.error_message or "",
            "output_dir": self.output_dir or "",
            "priority": str(self.priority),
            "tags": json.dumps(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """
        Create Job from dictionary (Redis hash).
        
        Handles deserialization of JSON fields and datetimes.
        """
        # Handle bytes from Redis
        if data and isinstance(list(data.values())[0], bytes):
            data = {k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()}

        # Parse JSON fields
        config = data.get("config", "{}")
        if isinstance(config, str):
            config = json.loads(config) if config else {}

        progress_str = data.get("progress", "null")
        if isinstance(progress_str, str):
            progress_data = json.loads(progress_str) if progress_str and progress_str != "null" else None
        else:
            progress_data = progress_str
        progress = JobProgress.from_dict(progress_data) if progress_data else None

        tags_str = data.get("tags", "[]")
        if isinstance(tags_str, str):
            tags = json.loads(tags_str) if tags_str else []
        else:
            tags = tags_str or []

        # Parse datetimes
        def parse_datetime(value: str) -> Optional[datetime]:
            if not value or value == "":
                return None
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None

        # Parse priority
        priority_str = data.get("priority", "0")
        try:
            priority = int(priority_str) if priority_str else 0
        except (ValueError, TypeError):
            priority = 0

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", JobType.TEACHER_TRAINING.value),
            status=JobStatus(data.get("status", JobStatus.PENDING.value)),
            config=config,
            progress=progress,
            worker_id=data.get("worker_id") or None,
            created_at=parse_datetime(data.get("created_at", "")),
            started_at=parse_datetime(data.get("started_at", "")),
            finished_at=parse_datetime(data.get("finished_at", "")),
            error_message=data.get("error_message") or None,
            output_dir=data.get("output_dir") or None,
            priority=priority,
            tags=tags,
        )

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (won't change)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

    @property
    def is_active(self) -> bool:
        """Check if job is currently active (running or cancelling)."""
        return self.status in [JobStatus.RUNNING, JobStatus.CANCELLING]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds (if started)."""
        if not self.started_at:
            return None
        end_time = self.finished_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def __repr__(self) -> str:
        return f"Job(id={self.id[:8]}..., type={self.type}, status={self.status.value})"


@dataclass
class WorkerInfo:
    """
    Worker registration information.

    Attributes:
        id: Unique worker identifier
        gpu_id: GPU device ID this worker uses
        hostname: Machine hostname
        status: Worker status (idle, busy, offline)
        current_job_id: ID of job being executed (if any)
        last_heartbeat: Last heartbeat timestamp
        started_at: When worker started
    """
    id: str
    gpu_id: int = 0
    hostname: str = ""
    status: str = "idle"  # idle, busy, offline
    current_job_id: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    started_at: Optional[datetime] = None

    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
        if self.started_at is None:
            self.started_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "id": self.id,
            "gpu_id": str(self.gpu_id),
            "hostname": self.hostname,
            "status": self.status,
            "current_job_id": self.current_job_id or "",
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else "",
            "started_at": self.started_at.isoformat() if self.started_at else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """Create from dictionary."""
        # Handle bytes from Redis
        if data and isinstance(list(data.values())[0], bytes):
            data = {k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in data.items()}

        def parse_datetime(value: str) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None

        return cls(
            id=data.get("id", ""),
            gpu_id=int(data.get("gpu_id", 0)),
            hostname=data.get("hostname", ""),
            status=data.get("status", "idle"),
            current_job_id=data.get("current_job_id") or None,
            last_heartbeat=parse_datetime(data.get("last_heartbeat", "")),
            started_at=parse_datetime(data.get("started_at", "")),
        )
