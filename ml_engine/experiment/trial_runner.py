"""
TrialRunner -- runs a single training trial with config overrides.

Each trial spawns its own subprocess (mp.get_context('spawn') + mp.Queue)
for GPU memory isolation. Reuses the existing Trainer pipeline exactly.
"""

import json
import logging
import multiprocessing as mp
import queue
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    trial_id: str
    config: Dict[str, Any]          # full merged config used
    overrides: Dict[str, Any]       # only the overrides applied
    metrics: Dict[str, float]       # all val metrics
    primary_metric: Optional[float] # the single scalar for comparison (mAP50)
    wall_time_seconds: float
    status: str                      # "completed", "crashed", "oom"
    error_message: Optional[str] = None
    output_dir: str = ""


_PRIMARY_METRIC_KEY = "val_mAP50"


def _trial_subprocess(
    data_manager,
    config: Dict[str, Any],
    output_dir: str,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    cancel_event: mp.Event,
    primary_metric_key: str = _PRIMARY_METRIC_KEY,
) -> None:
    """Entry point for per-trial subprocess."""
    from pathlib import Path as _Path
    import sys as _sys

    # Same sys.path setup as main subprocess_runner
    project_root = str(_Path(__file__).parent.parent.parent)
    if project_root not in _sys.path:
        _sys.path.insert(0, project_root)

    from ml_engine.training import Trainer, TrainingCancelledException

    def _progress_cb(info):
        try:
            progress_queue.put_nowait(info)
        except Exception:
            pass

    def _cancel_check():
        return cancel_event.is_set()

    try:
        trainer = Trainer(
            data_manager=data_manager,
            output_dir=output_dir,
            config=config,
            progress_callback=_progress_cb,
            cancel_check=_cancel_check,
        )
        val_metrics = trainer.train()

        # Try to read mAP50 from evaluation report (more reliable than val_metrics)
        primary = _extract_primary_metric(output_dir, val_metrics, primary_metric_key)
        result_queue.put({
            "status": "completed",
            "metrics": val_metrics,
            "primary_metric": primary,
        })
    except TrainingCancelledException:
        result_queue.put({"status": "cancelled"})
    except MemoryError as e:
        result_queue.put({"status": "oom", "error": str(e)})
    except Exception as e:
        import traceback
        result_queue.put({"status": "crashed", "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})


def _extract_primary_metric(
    output_dir: str,
    val_metrics: Dict[str, float],
    primary_metric_key: str = _PRIMARY_METRIC_KEY,
) -> Optional[float]:
    """
    Extract the primary metric from evaluation report JSON (preferred) or fall back to val_metrics.

    TrialRunner uses best-checkpoint evaluation (written by _evaluate_on_test_set),
    not the proxy loss from _validate_epoch.
    """
    # Derive the bare metric name for report lookup (strip 'val_' prefix)
    bare_key = primary_metric_key.removeprefix("val_")
    eval_dir = Path(output_dir) / "evaluation"
    for report_path in eval_dir.glob("*_report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            # Report structure: {"metrics": {"mAP50": 0.72, ...}} or {"mAP50": 0.72}
            metrics = report.get("metrics", report)
            if bare_key in metrics:
                return float(metrics[bare_key])
        except Exception:
            pass

    # Fallback: look for the metric in training val_metrics
    return val_metrics.get(primary_metric_key)


class TrialRunner:
    """
    Runs a single training trial in an isolated subprocess.

    Reuses build_teacher_training_config and Trainer exactly as
    TeacherTrainingHandler does. The only difference is that overrides
    come from the experiment loop rather than job_config["training"].
    """

    # Kill a hung subprocess after this many seconds without exit.
    # 3 epochs × ~10 min/epoch × 3 safety margin = ~90 min upper bound.
    # Override per-call via max_wall_time_seconds if needed.
    DEFAULT_TIMEOUT_SECONDS = 5400  # 90 minutes

    def run(
        self,
        data_manager,
        overrides: Dict[str, Any],
        base_output_dir: str,
        trial_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        max_wall_time_seconds: Optional[int] = None,
        primary_metric_key: str = _PRIMARY_METRIC_KEY,
    ) -> TrialResult:
        """
        Run one trial.

        1. Build full config via build_teacher_training_config(overrides)
        2. Spawn subprocess, run Trainer
        3. Read evaluation report for primary_metric
        4. Return TrialResult

        Args:
            data_manager: DataManager instance.
            overrides: Config overrides (dotted or nested dict), validated by ExperimentLoop.
            base_output_dir: Parent dir; trial output goes in {base_output_dir}/{trial_id}/.
            trial_id: Optional explicit ID; auto-generated if None.
            progress_callback: Called with progress dicts from the subprocess.
            cancel_check: Returns True if the loop should abort.
            primary_metric_key: Metric key to extract as the primary scalar (e.g. 'val_mAP50').
        """
        from ml_engine.training.config import build_teacher_training_config

        trial_id = trial_id or f"trial_{uuid.uuid4().hex[:8]}"
        trial_output_dir = str(Path(base_output_dir) / trial_id)

        logger.info("Trial %s starting (overrides=%s)", trial_id, list(overrides.keys()))
        t0 = time.monotonic()

        # Build full config
        config = build_teacher_training_config(data_manager, overrides)

        # IPC
        ctx = mp.get_context("spawn")
        result_q: mp.Queue = ctx.Queue()
        progress_q: mp.Queue = ctx.Queue()
        cancel_ev: mp.Event = ctx.Event()

        proc = ctx.Process(
            target=_trial_subprocess,
            args=(data_manager, config, trial_output_dir, result_q, progress_q, cancel_ev,
                  primary_metric_key),
            daemon=False,
        )
        proc.start()

        timeout = max_wall_time_seconds or self.DEFAULT_TIMEOUT_SECONDS

        # Monitor
        result_payload: Optional[Dict] = None
        timed_out = False
        while proc.is_alive():
            # Forward progress
            while True:
                try:
                    info = progress_q.get_nowait()
                    if progress_callback:
                        progress_callback(info)
                except queue.Empty:
                    break

            if cancel_check and cancel_check():
                cancel_ev.set()

            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                logger.warning("Trial %s exceeded wall-time limit (%.0fs), killing.", trial_id, timeout)
                proc.kill()
                timed_out = True
                break

            time.sleep(0.5)

        # Ensure subprocess is fully reaped before reading the result queue.
        # join() waits for the OS to reclaim the process and for Queue feeder
        # threads to finish flushing — prevents the result being silently lost.
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()

        # Drain progress
        while True:
            try:
                info = progress_q.get_nowait()
                if progress_callback:
                    progress_callback(info)
            except queue.Empty:
                break

        # Get result
        try:
            result_payload = result_q.get_nowait()
        except queue.Empty:
            result_payload = None

        if timed_out and result_payload is None:
            result_payload = {"status": "crashed", "error": f"Trial exceeded wall-time limit ({timeout}s)"}

        # Cleanup queues
        for q in (result_q, progress_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        wall_time = time.monotonic() - t0
        status = result_payload.get("status", "crashed") if result_payload else "crashed"
        error_msg = result_payload.get("error") if result_payload else f"Subprocess exited with code {proc.exitcode}"

        return TrialResult(
            trial_id=trial_id,
            config=config,
            overrides=overrides,
            metrics=result_payload.get("metrics", {}) if result_payload else {},
            primary_metric=result_payload.get("primary_metric") if result_payload else None,
            wall_time_seconds=wall_time,
            status=status,
            error_message=error_msg if status != "completed" else None,
            output_dir=trial_output_dir,
        )
