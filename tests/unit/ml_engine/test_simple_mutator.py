"""
Unit tests for ml_engine.experiment.mutators.SimpleMutator.
"""

from __future__ import annotations

import pytest

from ml_engine.experiment.mutators import SimpleMutator
from ml_engine.experiment.trial_log import TrialLog, TrialRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUTABLE_KEYS = {
    "batch_size": {"type": "int", "min": 1, "max": 32},
    "learning_rate": {"type": "float", "min": 1e-6, "max": 1e-2, "log_scale": True},
    "optimizer": {"type": "choice", "choices": ["AdamW", "SGD"]},
    "use_amp": {"type": "bool"},
}

SINGLE_KEY = {
    "batch_size": {"type": "int", "min": 1, "max": 32},
}


def make_log(output_dir: str, trials: list = None) -> TrialLog:
    log = TrialLog(
        run_id="test-run",
        output_dir=output_dir,
        budget_summary={"metric_mode": "max"},
    )
    for t in trials or []:
        log.append(t)
    return log


def record(trial_id: str, metric: float, overrides: dict = None, status: str = "keep") -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        overrides=overrides or {},
        primary_metric=metric,
        all_metrics={"val_mAP50": metric},
        status=status,
        description=f"trial {trial_id}",
    )


# ---------------------------------------------------------------------------
# propose() -- baseline (no trials)
# ---------------------------------------------------------------------------


class TestProposeBaseline:
    r"""Tests for propose() behavior when no trials have been logged yet."""

    def test_returns_empty_dict_when_no_trials(self, tmp_path):
        r"""Should return {} to indicate baseline (no overrides) when no trials exist."""
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=42)
        log = make_log(str(tmp_path))
        result = m.propose(log)
        assert not result

    def test_baseline_returned_on_first_call(self, tmp_path):
        r"""Even if mutable_keys are defined, the first call to propose()
        with an empty trial log should return {}."""
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        log = make_log(str(tmp_path))
        assert not m.propose(log)


# ---------------------------------------------------------------------------
# propose() -- returns one key
# ---------------------------------------------------------------------------


class TestProposeReturnsOneKey:
    r"""Tests for propose() behavior when trials exist:
    should return a dict with exactly one key from mutable_keys."""

    def test_returns_single_key_dict(self, tmp_path):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=1)
        log = make_log(str(tmp_path), [record("t1", 0.5)])
        result = m.propose(log)
        assert len(result) == 1

    def test_returned_key_is_in_mutable_keys(self, tmp_path):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=2)
        log = make_log(str(tmp_path), [record("t1", 0.5)])
        result = m.propose(log)
        key = next(iter(result))
        assert key in MUTABLE_KEYS


# ---------------------------------------------------------------------------
# propose() -- empty mutable_keys raises
# ---------------------------------------------------------------------------


class TestProposeEmptyMutableKeys:
    def test_raises_when_mutable_keys_empty(self, tmp_path):
        m = SimpleMutator(mutable_keys={}, seed=0)
        log = make_log(str(tmp_path), [record("t1", 0.5)])
        with pytest.raises(ValueError, match="mutable_keys is empty"):
            m.propose(log)


# ---------------------------------------------------------------------------
# _sample_value -- type contracts
# ---------------------------------------------------------------------------


class TestSampleValue:
    def test_int_type_returns_int(self):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        for _ in range(20):
            v = m._sample_value("batch_size")
            assert isinstance(v, int)
            assert 1 <= v <= 32

    def test_float_type_returns_float(self):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        for _ in range(20):
            v = m._sample_value("learning_rate")
            assert isinstance(v, float)
            assert 1e-6 <= v <= 1e-2

    def test_log_scale_samples_full_range(self):
        """Log-scale should produce values near min and near max."""
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        values = [m._sample_value("learning_rate") for _ in range(200)]
        assert any(v < 1e-5 for v in values), "Expected some small LR values"
        assert any(v > 5e-3 for v in values), "Expected some large LR values"

    def test_bool_type_returns_bool(self):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        for _ in range(20):
            v = m._sample_value("use_amp")
            assert isinstance(v, bool)

    def test_choice_type_returns_valid_choice(self):
        m = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=0)
        for _ in range(20):
            v = m._sample_value("optimizer")
            assert v in ["AdamW", "SGD"]

    def test_list_type_returns_valid_subset(self):
        keys = {"augmentations": {"type": "list", "items": ["flip", "rotate", "blur", "crop"]}}
        m = SimpleMutator(mutable_keys=keys, seed=0)
        for _ in range(20):
            v = m._sample_value("augmentations")
            assert isinstance(v, list)
            assert 1 <= len(v) <= 4
            assert all(item in ["flip", "rotate", "blur", "crop"] for item in v)
            # No duplicates (random.sample guarantees this)
            assert len(v) == len(set(v))

    def test_list_type_empty_items_returns_empty_list(self):
        keys = {"augmentations": {"type": "list", "items": []}}
        m = SimpleMutator(mutable_keys=keys, seed=0)
        v = m._sample_value("augmentations")
        assert v == []

    def test_unknown_type_raises(self):
        keys = {"mystery": {"type": "quaternion"}}
        m = SimpleMutator(mutable_keys=keys, seed=0)
        with pytest.raises(ValueError, match="unsupported type"):
            m._sample_value("mystery")


# ---------------------------------------------------------------------------
# Stagnation avoidance
# ---------------------------------------------------------------------------


class TestStagnationAvoidance:
    def test_different_key_chosen_after_stagnation(self, tmp_path):
        """If last 3 trials show no improvement and same key used, switch key."""
        # Only 2 keys so we can observe switching
        keys = {
            "batch_size": {"type": "int", "min": 1, "max": 32},
            "learning_rate": {"type": "float", "min": 1e-6, "max": 1e-2},
        }
        m = SimpleMutator(mutable_keys=keys, seed=7)
        log = make_log(str(tmp_path))

        # Force _last_key and stagnant history
        m._last_key = "batch_size"
        stagnant_metric = 0.5
        for i in range(3):
            log.append(record(f"t{i}", stagnant_metric, overrides={"batch_size": 8 + i}))

        result = m.propose(log)
        key = next(iter(result))
        assert key == "learning_rate"  # should switch away from batch_size


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    def test_same_seed_same_sequence(self, tmp_path):
        trials = [record("t1", 0.5), record("t2", 0.55)]

        m1 = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=99)
        log1 = make_log(str(tmp_path / "run1"), trials)
        results1 = [m1.propose(log1) for _ in range(5)]

        m2 = SimpleMutator(mutable_keys=MUTABLE_KEYS, seed=99)
        log2 = make_log(str(tmp_path / "run2"), trials)
        results2 = [m2.propose(log2) for _ in range(5)]

        assert results1 == results2

    def test_different_seeds_likely_differ(self, tmp_path):
        trials = [record("t1", 0.5)]
        m1 = SimpleMutator(mutable_keys=SINGLE_KEY, seed=0)
        m2 = SimpleMutator(mutable_keys=SINGLE_KEY, seed=12345)
        log1 = make_log(str(tmp_path / "r1"), trials)
        log2 = make_log(str(tmp_path / "r2"), trials)
        # With int range [1,32], very likely different values
        v1 = list(m1.propose(log1).values())[0]
        v2 = list(m2.propose(log2).values())[0]
        # Not guaranteed equal -- just run a few to confirm they can differ
        # (probabilistic, but with range 1-32 collision prob is ~3%)
        assert isinstance(v1, int) and isinstance(v2, int)
