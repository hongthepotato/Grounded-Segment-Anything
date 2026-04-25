"""
AutoResearch -- bounded HPO primitives for teacher training experiments.

Public API::

    from ml_engine.experiment import (
        ConfigGuard, GuardResult,
        TrialRunner, TrialResult,
        TrialLog, TrialRecord,
        ExperimentLoop, ExperimentBudget, ExperimentResult,
        SimpleMutator,
    )
"""

from ml_engine.experiment.config_guard import ConfigGuard, GuardResult
from ml_engine.experiment.llm_propose import LLMProposeFn
from ml_engine.experiment.loop import ExperimentBudget, ExperimentLoop, ExperimentResult
from ml_engine.experiment.mutators import SimpleMutator
from ml_engine.experiment.trial_log import TrialLog, TrialRecord
from ml_engine.experiment.trial_runner import TrialResult, TrialRunner

__all__ = [
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
    "LLMProposeFn",
]
