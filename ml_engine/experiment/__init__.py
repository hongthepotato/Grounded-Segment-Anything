"""
AutoResearch -- bounded HPO primitives for teacher training experiments.

Public API::

    from ml_engine.experiment import (
        ConfigSnapshot,
        ConfigGuard, GuardResult,
        TrialRunner, TrialResult,
        TrialLog, TrialRecord,
        ExperimentLoop, ExperimentBudget, ExperimentResult,
        SimpleMutator,
    )
"""

from ml_engine.experiment.config_snapshot import ConfigSnapshot
from ml_engine.experiment.config_guard import ConfigGuard, GuardResult
from ml_engine.experiment.trial_runner import TrialRunner, TrialResult
from ml_engine.experiment.trial_log import TrialLog, TrialRecord
from ml_engine.experiment.loop import ExperimentLoop, ExperimentBudget, ExperimentResult
from ml_engine.experiment.mutators import SimpleMutator

__all__ = [
    "ConfigSnapshot",
    "ConfigGuard",
    "GuardResult",
    "TrialRunner",
    "TrialResult",
    "TrialLog",
    "TrialRecord",
    "ExperimentLoop",
    "ExperimentBudget",
    "ExperimentResult",
    "SimpleMutator",
]
