"""
Teacher training job handler.

Handles the teacher_training job type for fine-tuning GroundingDINO and SAM models.
"""

import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Dict, Any

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError


class TeacherTrainingHandler(JobHandler):
    """
    Handler for teacher model training jobs.
    
    Trains GroundingDINO and/or SAM models with LoRA adapters on custom datasets.
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        """
        Execute teacher training job.
        
        Args:
            job_config: Configuration containing:
                - data_path: Path to dataset
                - image_paths: List of image paths
                - split_config: Train/val/test split ratios (optional)
                - training: Training hyperparameter overrides (optional)
            output_dir: Directory for checkpoints and logs
            progress_queue: Queue for progress updates
            cancel_event: Cancellation signal
        """
        # Late imports - these load in subprocess, not parent
        from ml_engine.training import Trainer, TrainingCancelledException
        from ml_engine.data.manager import DataManager
        from core.constants import transform_image_path

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
        data_manager = DataManager.from_file(
            data_path=data_path,
            image_paths=image_paths,
            split_config=split_config
        )

        # Build config
        from ml_engine.training.config import build_teacher_training_config
        config = build_teacher_training_config(data_manager, job_config.get("training"))

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
        trainer = Trainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )

        started_at = time.monotonic()
        try:
            val_metrics = trainer.train()
        except TrainingCancelledException as e:
            raise TrainingCancelledError("Training cancelled by user") from e

        wall_time = time.monotonic() - started_at

        # Collect artifact paths produced by this job
        out = Path(output_dir)
        artifacts = []
        for pattern in ("evaluation/*.json", "**/*.pth", "export/**/*"):
            artifacts.extend(str(p) for p in out.glob(pattern) if p.is_file())

        # Write outcome.json -- read by worker and included in job_completed event
        outcome = {
            "status": "completed",
            "metrics": val_metrics,
            "artifacts": artifacts,
            "wall_time_seconds": wall_time,
            "error_message": None,
        }
        outcome_path = out / "outcome.json"
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        outcome_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
