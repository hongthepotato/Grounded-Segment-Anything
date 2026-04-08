"""
SimpleMutator -- built-in propose_fn for standalone ExperimentLoop.

At Stage 4 the Executor agent replaces this with LLM-guided proposals.
Until then, SimpleMutator provides a reasonable baseline: random perturbation
with basic heuristics informed by trial history.
"""

import logging
import math
import random
from typing import Any, Dict, Optional

from ml_engine.experiment.trial_log import TrialLog

logger = logging.getLogger(__name__)


class SimpleMutator:
    """
    Proposes the next config overrides based on trial history.

    Strategy (in order):
    1. If no trials yet: return {} (baseline, no overrides).
    2. Apply one random perturbation to the best config seen so far.
    3. Basic heuristics: if best metric hasn't improved in last 3 trials,
       perturb a different key than the previous trial tried.

    This is intentionally simple. The value of AutoResearch comes from the
    LLM Executor at Stage 4; SimpleMutator just makes the standalone job
    useful for sanity-checking the pipeline.
    """

    def __init__(self, mutable_keys: Dict[str, Dict[str, Any]], seed: Optional[int] = None):
        """
        Args:
            mutable_keys: Same schema as ConfigGuard.mutable_keys.
            seed: Optional random seed for reproducible experiments.
        """
        self._mutable = mutable_keys
        self._rng = random.Random(seed)
        self._last_key: Optional[str] = None

    def propose(self, trial_log: TrialLog) -> Dict[str, Any]:
        """
        Propose next overrides given trial history.

        Returns a flat dict of dotted_key -> value.
        Returns {} to run the baseline (no overrides) on the first call.
        """
        trials = trial_log.trials
        if not trials:
            logger.debug("SimpleMutator: no trials yet, returning baseline (no overrides)")
            return {}

        # Pick a key to mutate (avoid last key if stuck)
        candidates = list(self._mutable.keys())
        if not candidates:
            raise ValueError("SimpleMutator: mutable_keys is empty — nothing to explore. Check experiment_loop.yaml.")
        if len(candidates) > 1 and self._last_key is not None:
            # If no improvement in last 3 non-baseline trials, try a different key
            recent = [t for t in trials[-3:] if t.overrides and t.primary_metric is not None]
            baseline = recent[0].primary_metric if recent else None
            stagnant = len(recent) == 3 and baseline is not None and all(
                math.isclose(t.primary_metric, baseline, rel_tol=1e-6) for t in recent
            )
            if stagnant:
                candidates = [k for k in candidates if k != self._last_key] or candidates

        key = self._rng.choice(candidates)
        self._last_key = key
        value = self._sample_value(key)
        logger.debug("SimpleMutator: proposing %s=%r", key, value)
        return {key: value}

    def _sample_value(self, key: str) -> Any:
        schema = self._mutable[key]
        vtype = schema.get("type", "any")

        if vtype == "int":
            lo, hi = schema.get("min", 1), schema.get("max", 100)
            return self._rng.randint(lo, hi)

        elif vtype == "float":
            lo, hi = schema.get("min", 0.0), schema.get("max", 1.0)
            # Log-uniform sampling for learning rates and similar
            if schema.get("log_scale", False) and lo > 0 and hi > 0:
                import math
                log_lo, log_hi = math.log(lo), math.log(hi)
                return math.exp(self._rng.uniform(log_lo, log_hi))
            return self._rng.uniform(lo, hi)

        elif vtype == "bool":
            return self._rng.choice([True, False])

        elif vtype == "choice":
            return self._rng.choice(schema.get("choices", [None]))

        elif vtype == "list":
            items = schema.get("items", [])
            k = self._rng.randint(1, max(1, len(items)))
            return self._rng.sample(items, k)

        return None
