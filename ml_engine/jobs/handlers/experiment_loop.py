"""
ExperimentLoopHandler -- JobHandler wrapper for the AutoResearch experiment loop.

Reads budget/mutable_keys/immutable_keys from job_config["experiment"] (with
fallback to configs/defaults/experiment_loop.yaml), builds a ConfigGuard and
SimpleMutator, and runs ExperimentLoop.

At Stage 4 the Executor agent will drive ExperimentLoop directly (bypassing this
handler), passing its own LLM-guided propose_fn. This handler is the standalone
path that works today.
"""

import json
import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Dict

from ml_engine.jobs.handlers.base import JobHandler, TrainingCancelledError

logger = logging.getLogger(__name__)


class ExperimentLoopHandler(JobHandler):
    """
    Runs AutoResearch as a standalone job type.

    job_config fields::

        {
            "data_path": "/data/dataset.json",
            "image_paths": [...],
            "split_config": {"train": 0.7, "val": 0.15, "test": 0.15},  # optional
            "experiment": {
                "max_trials": 20,            # override defaults
                "epochs_per_trial": 5,
                "metric_name": "val_mAP50",
                "metric_mode": "max",
                "max_wall_time_seconds": null,
            }
        }
    """

    def run(
        self,
        job_config: Dict[str, Any],
        output_dir: str,
        progress_queue: mp.Queue,
        cancel_event: mp.Event,
    ) -> None:
        # Late imports -- these load in subprocess, not parent
        from core.config import load_config
        from core.constants import DEFAULT_CONFIGS_DIR, transform_image_path
        from ml_engine.data.manager import DataManager
        from ml_engine.experiment import (
            ConfigGuard,
            ExperimentBudget,
            ExperimentLoop,
            SimpleMutator,
        )

        # --- Data setup ---
        data_path_raw = job_config.get("data_path")
        data_path = transform_image_path(data_path_raw) if data_path_raw else None
        image_paths = job_config.get("image_paths", [])

        if not data_path:
            raise ValueError("data_path required in job_config")
        if not image_paths:
            raise ValueError("image_paths required in job_config")

        split_config = job_config.get("split_config", {"train": 0.7, "val": 0.15, "test": 0.15})
        data_manager = DataManager.from_file(
            data_path=data_path,
            image_paths=image_paths,
            split_config=split_config,
        )

        # --- Budget config ---
        defaults = load_config(str(DEFAULT_CONFIGS_DIR / "experiment_loop.yaml"))
        exp_defaults = defaults.get("experiment", {})
        exp_overrides = job_config.get("experiment", {})
        exp_cfg = {**exp_defaults, **exp_overrides}

        budget = ExperimentBudget(
            max_trials=int(exp_cfg.get("max_trials", 20)),
            epochs_per_trial=int(exp_cfg.get("epochs_per_trial", 5)),
            max_wall_time_seconds=exp_cfg.get("max_wall_time_seconds"),
            metric_name=exp_cfg.get("metric_name", "val_mAP50"),
            metric_mode=exp_cfg.get("metric_mode", "max"),
        )

        # --- Guard setup ---
        mutable_keys = defaults.get("mutable_keys", {})
        immutable_keys = defaults.get("immutable_keys", [])
        guard = ConfigGuard(mutable_keys=mutable_keys, immutable_keys=immutable_keys)

        # --- Mutator ---
        mutator = SimpleMutator(mutable_keys=mutable_keys)

        # --- Progress forwarding ---
        def _progress_cb(info: Dict) -> None:
            try:
                progress_queue.put_nowait(info)
            except Exception:
                pass

        def _cancel_check() -> bool:
            return cancel_event.is_set()

        # --- Run ---
        loop = ExperimentLoop(guard=guard)
        result = loop.run(
            data_manager=data_manager,
            output_dir=output_dir,
            budget=budget,
            propose_fn=mutator.propose,
            progress_callback=_progress_cb,
            cancel_check=_cancel_check,
        )

        logger.info(
            "ExperimentLoop done: best_metric=%.4f (%s trials, best_trial=%s)",
            result.best_metric or 0.0,
            result.trials_completed,
            result.best_trial_id,
        )

        # Write outcome.json so worker can publish it in job_completed event
        outcome = {
            "status": "completed",
            "metrics": {
                "best_metric": result.best_metric,
                "trials_completed": result.trials_completed,
            },
            "artifacts": [
                str(Path(output_dir) / "experiment_log.json"),
                str(Path(output_dir) / "best_config.yaml"),
                str(Path(output_dir) / "feedback.json"),
            ],
            "wall_time_seconds": result.wall_time_seconds,
            "error_message": None,
            "experiment_result": {
                "run_id": result.run_id,
                "best_trial_id": result.best_trial_id,
                "best_metric": result.best_metric,
                "trials_completed": result.trials_completed,
            },
        }
        (Path(output_dir) / "outcome.json").write_text(
            json.dumps(outcome, indent=2), encoding="utf-8"
        )
