"""
ExperimentLoop -- orchestrates HPO trials within a budget.

Config decisions are delegated to a propose_fn callback:
- Today (standalone job): SimpleMutator.propose()
- Stage 4 (Executor agent): LLM-guided proposal via Executor.propose()

The loop itself is policy-free: it runs trials, tracks results, and respects
the budget. All "what to try next" intelligence lives in propose_fn.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ml_engine.experiment.config_guard import ConfigGuard
from ml_engine.experiment.config_snapshot import ConfigSnapshot
from ml_engine.experiment.trial_log import TrialLog, TrialRecord
from ml_engine.experiment.trial_runner import TrialResult, TrialRunner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentBudget:
    max_trials: int = 20
    epochs_per_trial: int = 5       # validated for LoRA convergence
    max_wall_time_seconds: Optional[int] = None
    metric_name: str = "val_mAP50"  # direct detection quality, not proxy loss
    metric_mode: str = "max"        # "min" or "max"


@dataclass
class ExperimentResult:
    run_id: str
    best_metric: Optional[float]
    best_config: Optional[Dict[str, Any]]
    best_trial_id: Optional[str]
    trials_completed: int
    wall_time_seconds: float
    output_dir: str


# Signature: propose_fn(trial_log) -> overrides dict
ProposeFn = Callable[[TrialLog], Dict[str, Any]]


class ExperimentLoop:
    """
    Orchestrates the experiment loop.

    Usage::

        from ml_engine.experiment.mutators import SimpleMutator

        mutator = SimpleMutator(mutable_keys=mutable_keys)
        loop = ExperimentLoop(guard=guard)
        result = loop.run(
            data_manager=dm,
            output_dir="experiments/exp_001",
            budget=ExperimentBudget(max_trials=20),
            propose_fn=mutator.propose,
        )
    """

    def __init__(self, guard: ConfigGuard):
        self._guard = guard

    def run(
        self,
        data_manager,
        output_dir: str,
        budget: ExperimentBudget,
        propose_fn: ProposeFn,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ExperimentResult:
        """
        Run the experiment loop.

        Steps:
        1. Run baseline (no overrides) as trial_001.
        2. For each remaining trial in budget:
           a. Call propose_fn(trial_log) -> overrides
           b. Validate via ConfigGuard (skip if invalid, still counts)
           c. TrialRunner.run(overrides)
           d. Update TrialLog and ConfigSnapshot
        3. Return ExperimentResult with best config.
        """
        run_id = f"exp_{uuid.uuid4().hex[:12]}"
        runner = TrialRunner()
        snapshot = ConfigSnapshot(output_dir)

        trial_log = TrialLog(
            run_id=run_id,
            output_dir=output_dir,
            budget_summary={
                "max_trials": budget.max_trials,
                "epochs_per_trial": budget.epochs_per_trial,
                "metric_name": budget.metric_name,
                "metric_mode": budget.metric_mode,
            },
        )

        t_start = time.monotonic()
        trials_run = 0

        for trial_num in range(budget.max_trials):
            if cancel_check and cancel_check():
                logger.info("ExperimentLoop cancelled at trial %d", trial_num)
                break

            if budget.max_wall_time_seconds is not None:
                elapsed = time.monotonic() - t_start
                if elapsed > budget.max_wall_time_seconds:
                    logger.info("ExperimentLoop wall-time budget exhausted (%.0fs)", elapsed)
                    break

            trial_id = f"trial_{trial_num + 1:03d}"

            # Baseline: no overrides on first trial
            if trial_num == 0:
                overrides: Dict[str, Any] = {}
                description = "baseline — no overrides"
            else:
                overrides = propose_fn(trial_log)
                description = f"overrides: {list(overrides.keys())}"

                # Validate before running
                if overrides:
                    guard_result = self._guard.validate(overrides)
                    if not guard_result:
                        logger.warning(
                            "Trial %s skipped (guard rejected): %s",
                            trial_id, guard_result.errors,
                        )
                        trial_log.append(TrialRecord(
                            trial_id=trial_id,
                            overrides=overrides,
                            primary_metric=None,
                            all_metrics={},
                            status="skip",
                            description=f"guard_rejected: {guard_result.errors}",
                        ))
                        trials_run += 1
                        continue

            logger.info("Running %s: %s", trial_id, description)

            # Inject epochs_per_trial override
            effective_overrides = {
                "training.epochs": budget.epochs_per_trial,
                **overrides,
            }

            result: TrialResult = runner.run(
                data_manager=data_manager,
                overrides=effective_overrides,
                base_output_dir=output_dir,
                trial_id=trial_id,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            trials_run += 1

            # Determine if this is an improvement
            is_best = False
            if result.primary_metric is not None and result.status == "completed":
                current_best = trial_log.best_metric
                if current_best is None:
                    is_best = True
                elif budget.metric_mode == "max" and result.primary_metric > current_best:
                    is_best = True
                elif budget.metric_mode == "min" and result.primary_metric < current_best:
                    is_best = True

            status = "keep" if (result.status == "completed") else result.status
            trial_log.append(TrialRecord(
                trial_id=trial_id,
                overrides=overrides,
                primary_metric=result.primary_metric,
                all_metrics=result.metrics,
                status=status,
                description=description,
                wall_time_seconds=result.wall_time_seconds,
                error_message=result.error_message,
            ))

            if result.status == "completed":
                snap_id = snapshot.capture(result.config)
                if is_best:
                    snapshot.mark_best(snap_id, result.primary_metric)
                    logger.info(
                        "Trial %s is new best: %s=%.4f",
                        trial_id, budget.metric_name, result.primary_metric,
                    )

        wall_time = time.monotonic() - t_start
        best_config = snapshot.get_best()
        best_trial = trial_log.get_best()

        # Write feedback.json for MemoryStore at Stage 4
        import json
        from pathlib import Path
        feedback_path = Path(output_dir) / "feedback.json"
        feedback_path.write_text(
            json.dumps(trial_log.to_feedback_record(), indent=2), encoding="utf-8"
        )

        return ExperimentResult(
            run_id=run_id,
            best_metric=trial_log.best_metric,
            best_config=best_config,
            best_trial_id=best_trial.trial_id if best_trial else None,
            trials_completed=trials_run,
            wall_time_seconds=wall_time,
            output_dir=output_dir,
        )
