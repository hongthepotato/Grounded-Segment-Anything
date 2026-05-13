"""
Unit tests for ml_engine.training.checkpoint_manager.CheckpointManager.

All tests run on CPU with a tiny torch.nn.Linear — no GPU required.

Bugs caught and fixed in this file (tests verify the correct post-fix behaviour):

  Bug 1 — patience_counter off-by-one on improvement epochs (FIXED).
    _check_early_stopping now receives is_best directly from _is_best so it
    never double-counts improvement epochs. patience_counter stays 0 after
    an improving epoch.

  Bug 2 — early stopping ignored min_delta (FIXED).
    _check_early_stopping now delegates to _is_best, which applies min_delta
    consistently. Sub-threshold improvements correctly increment patience_counter.

  Bug 3 — patience_counter not persisted across checkpoint save/load (FIXED).
    patience_counter is now written into the checkpoint dict and restored by
    load_checkpoint, so early stopping state survives a training resume.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

from ml_engine.training.checkpoint_manager import CheckpointManager

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


_DEFAULT_MIN_DELTA = 0.001


def _make_model() -> nn.Linear:
    return nn.Linear(4, 2, bias=False)


def _make_optimizer(model: nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=0.01)


class _FakeScaler:
    """Minimal GradScaler stand-in for tests that don't need real AMP scaling."""

    def __init__(self, scale: float = 1024.0) -> None:
        self._scale = scale
        self._state: dict = {}

    def state_dict(self) -> dict:
        return {"scale": self._scale}

    def load_state_dict(self, d: dict) -> None:
        self._state = d


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "ckpt_cfg.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            checkpointing:
              save_interval: 5
              max_keep_checkpoints: 3
              save_trainable_only: false
              min_delta: 0.001
              early_stopping:
                enabled: true
                patience: 3
                min_delta: 0.001
        """)
    )
    return cfg


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def manager(output_dir: Path, config_file: Path) -> CheckpointManager:
    return CheckpointManager(
        output_dir=str(output_dir),
        config_path=str(config_file),
        monitor_metric="val_loss",
        mode="min",
    )


def _save(
    mgr: CheckpointManager,
    epoch: int,
    val_loss: float,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra_info: Optional[dict] = None,
) -> Optional[Path]:
    m = model or _make_model()
    opt = optimizer or _make_optimizer(m)
    return mgr.save_checkpoint(
        epoch=epoch,
        model=m,
        optimizer=opt,
        metrics={"val_loss": val_loss, "epoch": float(epoch)},
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# Bug 1 — patience_counter off-by-one on improvement epochs (FIXED)
# ---------------------------------------------------------------------------


class TestBug1PatienceCounterOnImprovement:
    def test_patience_counter_stays_zero_after_improvement(self, manager):
        """After one improving epoch the patience counter must remain at 0."""
        _save(manager, epoch=1, val_loss=0.5)
        assert manager.patience_counter == 0, (
            "patience_counter should be 0 after an improving epoch; "
            f"got {manager.patience_counter} (off-by-one bug)"
        )

    def test_should_stop_not_triggered_before_patience_exhausted(self, manager):
        """
        With patience=3, should_stop must stay False until there are 3
        consecutive non-improving saves AFTER the baseline is established.
        """
        _save(manager, epoch=1, val_loss=0.5)  # improvement — sets baseline
        _save(manager, epoch=2, val_loss=0.8)  # no improvement #1
        _save(manager, epoch=3, val_loss=0.8)  # no improvement #2
        assert manager.should_stop is False, (
            "should_stop should still be False after 2 non-improving epochs "
            f"when patience=3; current patience_counter={manager.patience_counter}"
        )

    def test_should_stop_triggers_only_after_full_patience(self, manager):
        """
        should_stop must flip to True only after exactly `patience` (3)
        non-improving epochs.
        """
        _save(manager, epoch=1, val_loss=0.5)  # improvement
        _save(manager, epoch=2, val_loss=0.8)  # no improvement #1
        _save(manager, epoch=3, val_loss=0.8)  # no improvement #2
        _save(manager, epoch=4, val_loss=0.8)  # no improvement #3 — triggers
        assert manager.should_stop is True

    def test_improvement_resets_patience_counter_to_zero(self, manager):
        """After a non-improving epoch, a subsequent improvement must reset counter to 0."""
        _save(manager, epoch=1, val_loss=0.5)  # improvement
        _save(manager, epoch=2, val_loss=0.8)  # no improvement → counter = 1
        _save(manager, epoch=3, val_loss=0.2)  # improvement → counter must reset to 0
        assert manager.patience_counter == 0


# ---------------------------------------------------------------------------
# Bug 2 — _check_early_stopping ignored min_delta (FIXED)
# ---------------------------------------------------------------------------


class TestBug2EarlyStoppingIgnoresMinDelta:
    def test_sub_threshold_improvement_still_increments_patience(self, manager):
        """
        A change smaller than min_delta (0.001) must not be treated as
        improvement.  Patience should increment exactly as if the metric
        had not changed at all.
        """
        _save(manager, epoch=1, val_loss=0.5)  # establishes best=0.5
        assert (manager.output_dir / "best.pth").exists()

        before_counter = manager.patience_counter
        _save(manager, epoch=2, val_loss=0.4995)  # delta=0.0005 < 0.001

        # best.pth should NOT be overwritten (improvement below min_delta)
        assert manager.best_metric == pytest.approx(0.5), (
            "best_metric must not update for sub-threshold improvements"
        )
        # patience counter must have incremented (no real improvement)
        assert manager.patience_counter == before_counter + 1, (
            "sub-threshold improvement must still increment patience_counter"
        )


# ---------------------------------------------------------------------------
# Bug 3 — patience_counter not persisted across checkpoint save/load (FIXED)
# ---------------------------------------------------------------------------


class TestBug3PatienceCounterNotPersisted:
    def test_patience_counter_survives_checkpoint_round_trip(self, manager, output_dir):
        """
        After advancing patience_counter (non-improving epochs), saving a
        checkpoint and loading it into a fresh manager must restore the
        patience_counter so early stopping continues from the correct position.
        """
        _save(manager, epoch=1, val_loss=0.5)  # improvement — sets baseline
        _save(manager, epoch=2, val_loss=0.8)  # counter → 1
        _save(manager, epoch=3, val_loss=0.8)  # counter → 2
        saved_counter = manager.patience_counter

        # Create a fresh manager and load the last checkpoint
        fresh = CheckpointManager.__new__(CheckpointManager)
        fresh.__dict__.update(manager.__dict__.copy())
        fresh.patience_counter = 0  # simulate a brand-new manager instance

        fresh.load_checkpoint("last", _make_model())
        assert fresh.patience_counter == saved_counter, (
            f"patience_counter should be {saved_counter} after load, got {fresh.patience_counter}"
        )


# ---------------------------------------------------------------------------
# Happy-path tests (these should pass before and after the bug fixes)
# ---------------------------------------------------------------------------


class TestSaveCheckpointFiles:
    def test_always_writes_last_pth(self, manager, output_dir):
        _save(manager, epoch=1, val_loss=0.5)
        assert (output_dir / "last.pth").exists()

    def test_periodic_checkpoint_at_interval(self, manager, output_dir):
        _save(manager, epoch=0, val_loss=0.5)
        assert (output_dir / "epoch_0000.pth").exists()

    def test_periodic_checkpoint_at_save_interval_multiple(self, manager, output_dir):
        _save(manager, epoch=5, val_loss=0.5)
        assert (output_dir / "epoch_0005.pth").exists()

    def test_no_periodic_checkpoint_between_intervals(self, manager, output_dir):
        _save(manager, epoch=3, val_loss=0.5)
        assert not (output_dir / "epoch_0003.pth").exists()

    def test_writes_best_pth_on_improvement(self, manager, output_dir):
        _save(manager, epoch=1, val_loss=0.5)
        assert (output_dir / "best.pth").exists()

    def test_overwrites_best_pth_on_improvement(self, manager, output_dir):
        _save(manager, epoch=1, val_loss=0.5)
        _save(manager, epoch=2, val_loss=0.3)
        ckpt = torch.load(output_dir / "best.pth", map_location="cpu", weights_only=True)
        assert ckpt["metrics"]["val_loss"] == pytest.approx(0.3)

    def test_does_not_overwrite_best_pth_when_worse(self, manager, output_dir):
        _save(manager, epoch=1, val_loss=0.3)
        _save(manager, epoch=2, val_loss=0.8)
        ckpt = torch.load(output_dir / "best.pth", map_location="cpu", weights_only=True)
        assert ckpt["metrics"]["val_loss"] == pytest.approx(0.3)


class TestIsBest:
    def test_min_mode_lower_value_is_best(self, manager):
        assert manager._is_best({"val_loss": 0.5}) is True

    def test_min_mode_higher_value_not_best(self, manager):
        manager.best_metric = 0.4
        assert manager._is_best({"val_loss": 0.5}) is False

    def test_improvement_below_min_delta_not_best(self, manager):
        manager.best_metric = 0.5
        assert manager._is_best({"val_loss": 0.4995}) is False  # delta=0.0005 < 0.001

    def test_improvement_above_min_delta_is_best(self, manager):
        manager.best_metric = 0.5
        assert manager._is_best({"val_loss": 0.498}) is True  # delta=0.002 > 0.001

    def test_max_mode_higher_value_is_best(self, output_dir, config_file):
        mgr = CheckpointManager(str(output_dir), str(config_file), "mAP", mode="max")
        assert mgr._is_best({"mAP": 0.8}) is True

    def test_max_mode_lower_value_not_best(self, output_dir, config_file):
        mgr = CheckpointManager(str(output_dir), str(config_file), "mAP", mode="max")
        mgr.best_metric = 0.9
        assert mgr._is_best({"mAP": 0.8}) is False

    def test_missing_metric_returns_false(self, manager):
        assert manager._is_best({"other": 0.1}) is False

    def test_updates_best_metric_on_improvement(self, manager):
        manager._is_best({"val_loss": 0.3})
        assert manager.best_metric == pytest.approx(0.3)

    def test_updates_best_epoch_on_improvement(self, manager):
        manager._is_best({"val_loss": 0.3, "epoch": 7.0})
        assert manager.best_epoch == 7


class TestCleanup:
    def test_keeps_at_most_max_keep_periodic_checkpoints(self, manager, output_dir):
        for epoch in [0, 5, 10, 15]:  # 4 saves, max_keep=3 → removes epoch_0000
            _save(manager, epoch=epoch, val_loss=0.5 - epoch * 0.01)
        remaining = list(output_dir.glob("epoch_*.pth"))
        assert len(remaining) == 3

    def test_oldest_checkpoint_deleted_first(self, manager, output_dir):
        for epoch in [0, 5, 10, 15]:
            _save(manager, epoch=epoch, val_loss=0.5 - epoch * 0.01)
        assert not (output_dir / "epoch_0000.pth").exists()
        assert (output_dir / "epoch_0015.pth").exists()

    def test_best_pth_not_deleted_by_cleanup(self, manager, output_dir):
        for epoch in [0, 5, 10, 15, 20]:
            _save(manager, epoch=epoch, val_loss=0.5 - epoch * 0.01)
        assert (output_dir / "best.pth").exists()

    def test_last_pth_always_present(self, manager, output_dir):
        for epoch in [0, 5, 10, 15]:
            _save(manager, epoch=epoch, val_loss=0.5)
        assert (output_dir / "last.pth").exists()


class TestEarlyStoppingDisabled:
    def test_disabled_early_stopping_never_sets_should_stop(self, tmp_path):
        cfg = tmp_path / "no_es.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                checkpointing:
                  save_interval: 1
                  max_keep_checkpoints: 100
                  save_trainable_only: false
                  early_stopping:
                    enabled: false
                    patience: 2
            """)
        )
        out = tmp_path / "ckpts"
        out.mkdir()
        mgr = CheckpointManager(str(out), str(cfg), "val_loss", mode="min")
        for i in range(10):
            _save(mgr, epoch=i, val_loss=0.9)
        assert mgr.should_stop is False


class TestLoadCheckpoint:
    def test_load_best_alias_resolves(self, manager, output_dir):
        _save(manager, epoch=2, val_loss=0.4)
        meta = manager.load_checkpoint("best", _make_model())
        assert meta["epoch"] == 2

    def test_load_last_alias_resolves(self, manager, output_dir):
        _save(manager, epoch=2, val_loss=0.4)
        meta = manager.load_checkpoint("last", _make_model())
        assert meta["epoch"] == 2

    def test_load_explicit_path(self, manager, output_dir):
        _save(manager, epoch=0, val_loss=0.5)
        meta = manager.load_checkpoint(str(output_dir / "epoch_0000.pth"), _make_model())
        assert meta["epoch"] == 0

    def test_raises_file_not_found(self, manager, output_dir):
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint(str(output_dir / "no_such.pth"), _make_model())

    @pytest.mark.parametrize(
        "bad_path",
        [
            "/nonexistent/path.pth",
            "../../etc/passwd",
        ],
    )
    def test_raises_on_path_traversal(self, manager, output_dir, bad_path):
        # Both absolute out-of-bounds paths and dotdot-relative traversals must be rejected.
        resolved = str(output_dir / bad_path) if not bad_path.startswith("/") else bad_path
        with pytest.raises(ValueError, match="outside output_dir"):
            manager.load_checkpoint(resolved, _make_model())

    def test_restores_model_weights(self, manager, output_dir):
        model = _make_model()
        original_weights = model.weight.data.clone()
        _save(manager, epoch=0, val_loss=0.5, model=model, optimizer=_make_optimizer(model))

        loaded = _make_model()
        loaded.weight.data.fill_(999.0)
        manager.load_checkpoint("last", loaded)
        assert torch.allclose(loaded.weight.data, original_weights)

    def test_restores_optimizer_state(self, manager, output_dir):
        model = _make_model()
        opt = _make_optimizer(model)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        _save(manager, epoch=0, val_loss=0.5, model=model, optimizer=opt)

        loaded_model = _make_model()
        loaded_opt = _make_optimizer(loaded_model)
        manager.load_checkpoint("last", loaded_model, optimizer=loaded_opt, load_optimizer=True)
        assert set(loaded_opt.state_dict()["state"].keys()) == set(opt.state_dict()["state"].keys())

    def test_skips_optimizer_when_load_optimizer_false(self, manager, output_dir):
        model = _make_model()
        opt = _make_optimizer(model)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        _save(manager, epoch=0, val_loss=0.5, model=model, optimizer=opt)

        loaded_model = _make_model()
        loaded_opt = _make_optimizer(loaded_model)
        manager.load_checkpoint("last", loaded_model, optimizer=loaded_opt, load_optimizer=False)
        assert loaded_opt.state_dict()["state"] == {}

    def test_restores_best_metric_from_checkpoint(self, manager, output_dir):
        _save(manager, epoch=0, val_loss=0.25)
        saved_best = manager.best_metric

        manager.best_metric = float("inf")  # simulate fresh manager state
        manager.load_checkpoint("best", _make_model())
        assert manager.best_metric == pytest.approx(saved_best)

    def test_returns_metadata_dict(self, manager, output_dir):
        _save(manager, epoch=3, val_loss=0.5)
        meta = manager.load_checkpoint("last", _make_model())
        assert isinstance(meta, dict)
        assert "epoch" in meta and "metrics" in meta


class TestSaveTrainableOnly:
    def test_saves_only_trainable_params(self, tmp_path):
        cfg = tmp_path / "to.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                checkpointing:
                  save_interval: 1
                  max_keep_checkpoints: 3
                  save_trainable_only: true
                  early_stopping:
                    enabled: false
                    patience: 10
            """)
        )
        out = tmp_path / "ckpts"
        out.mkdir()
        mgr = CheckpointManager(str(out), str(cfg), "val_loss", mode="min")

        trainable = nn.Linear(4, 2, bias=False)  # requires_grad=True
        frozen = nn.Linear(2, 1, bias=False)  # will be frozen
        for p in frozen.parameters():
            p.requires_grad = False
        combined = nn.Sequential(trainable, frozen)

        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, combined.parameters()), lr=0.01)
        mgr.save_checkpoint(epoch=1, model=combined, optimizer=opt, metrics={"val_loss": 0.5, "epoch": 1.0})

        ckpt = torch.load(out / "last.pth", map_location="cpu", weights_only=True)
        assert ckpt["trainable_only"] is True
        saved_keys = set(ckpt["model_state_dict"].keys())
        # frozen layer (index 1) param must not appear
        assert not any(k.startswith("1.") for k in saved_keys)

    def test_trainable_only_loads_without_error(self, tmp_path):
        cfg = tmp_path / "to2.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                checkpointing:
                  save_interval: 1
                  max_keep_checkpoints: 3
                  save_trainable_only: true
                  early_stopping:
                    enabled: false
                    patience: 10
            """)
        )
        out = tmp_path / "ckpts2"
        out.mkdir()
        mgr = CheckpointManager(str(out), str(cfg), "val_loss", mode="min")

        model = _make_model()
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        mgr.save_checkpoint(epoch=1, model=model, optimizer=opt, metrics={"val_loss": 0.5, "epoch": 1.0})
        mgr.load_checkpoint("last", _make_model())  # must not raise


class TestPathHelpers:
    def test_get_best_checkpoint_path(self, manager, output_dir):
        assert manager.get_best_checkpoint_path() == output_dir / "best.pth"

    def test_get_last_checkpoint_path(self, manager, output_dir):
        assert manager.get_last_checkpoint_path() == output_dir / "last.pth"


class TestCoverageGaps:
    """Tests for code paths not covered by the initial bug-fix test suite."""

    # ------------------------------------------------------------------ #
    # min_delta config fallbacks                                           #
    # ------------------------------------------------------------------ #

    def test_min_delta_from_top_level_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                checkpointing:
                  save_interval: 1
                  max_keep_checkpoints: 3
                  save_trainable_only: false
                  min_delta: 0.05
                  early_stopping:
                    enabled: true
                    patience: 5
            """)
        )
        out = tmp_path / "ckpts"
        out.mkdir()
        mgr = CheckpointManager(str(out), str(cfg), "val_loss", mode="min")
        assert mgr.min_delta == 0.05

    def test_min_delta_hardcoded_default(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            textwrap.dedent("""\
                checkpointing:
                  save_interval: 1
                  max_keep_checkpoints: 3
                  save_trainable_only: false
                  early_stopping:
                    enabled: true
                    patience: 5
            """)
        )
        out = tmp_path / "ckpts"
        out.mkdir()
        mgr = CheckpointManager(str(out), str(cfg), "val_loss", mode="min")
        assert mgr.min_delta == _DEFAULT_MIN_DELTA

    # ------------------------------------------------------------------ #
    # save_checkpoint: optional components                                 #
    # ------------------------------------------------------------------ #

    def test_save_includes_extra_info(self, manager: CheckpointManager, output_dir: Path) -> None:
        _save(manager, epoch=0, val_loss=0.5, extra_info={"run_id": "abc", "fold": 2})
        ckpt = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        assert ckpt["extra_info"] == {"run_id": "abc", "fold": 2}

    def test_save_includes_scheduler_state(self, manager: CheckpointManager, output_dir: Path) -> None:
        model = _make_model()
        opt = _make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=opt,
            metrics={"val_loss": 0.5, "epoch": 0.0},
            scheduler=scheduler,
        )
        ckpt = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" in ckpt

    def test_save_includes_scaler_state(self, manager: CheckpointManager, output_dir: Path) -> None:
        model = _make_model()
        opt = _make_optimizer(model)
        manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=opt,
            metrics={"val_loss": 0.5, "epoch": 0.0},
            scaler=_FakeScaler(scale=1024.0),
        )
        ckpt = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        assert "scaler_state_dict" in ckpt
        assert ckpt["scaler_state_dict"]["scale"] == 1024.0

    # ------------------------------------------------------------------ #
    # load_checkpoint: optional components                                 #
    # ------------------------------------------------------------------ #

    def test_load_restores_scheduler_state(self, manager: CheckpointManager, output_dir: Path) -> None:
        model = _make_model()
        opt = _make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        scheduler.step()
        expected_lr = scheduler.get_last_lr()
        manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=opt,
            metrics={"val_loss": 0.5, "epoch": 0.0},
            scheduler=scheduler,
        )
        model2 = _make_model()
        opt2 = _make_optimizer(model2)
        scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5, gamma=0.1)
        manager.load_checkpoint("last", model2, optimizer=opt2, scheduler=scheduler2)
        assert scheduler2.get_last_lr() == expected_lr

    def test_load_restores_scaler_state(self, manager: CheckpointManager, output_dir: Path) -> None:
        scaler = _FakeScaler(scale=2048.0)
        model = _make_model()
        opt = _make_optimizer(model)
        manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=opt,
            metrics={"val_loss": 0.5, "epoch": 0.0},
            scaler=scaler,
        )
        loaded_scaler = _FakeScaler()
        manager.load_checkpoint("last", _make_model(), scaler=loaded_scaler)
        assert loaded_scaler._state == {"scale": 2048.0}

    def test_load_skips_rng_when_flag_false(self, manager: CheckpointManager, output_dir: Path) -> None:
        _save(manager, epoch=0, val_loss=0.5)
        manager.load_checkpoint("last", _make_model(), load_rng_state=False)

    def test_load_checkpoint_without_rng_state_key(
        self, manager: CheckpointManager, output_dir: Path
    ) -> None:
        _save(manager, epoch=0, val_loss=0.5)
        ckpt = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        del ckpt["rng_state"]
        torch.save(ckpt, output_dir / "last.pth")
        manager.load_checkpoint("last", _make_model())

    def test_load_rng_exception_is_swallowed(self, manager: CheckpointManager, output_dir: Path) -> None:
        _save(manager, epoch=0, val_loss=0.5)
        ckpt = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        # Replace rng state with a zero-element tensor that set_rng_state will reject
        ckpt["rng_state"]["python"] = torch.zeros(1, dtype=torch.uint8)
        torch.save(ckpt, output_dir / "last.pth")
        manager.load_checkpoint("last", _make_model())  # must not propagate the RuntimeError

    # ------------------------------------------------------------------ #
    # _cleanup edge cases                                                  #
    # ------------------------------------------------------------------ #

    def test_cleanup_noop_when_below_max_keep(self, manager: CheckpointManager, output_dir: Path) -> None:
        _save(manager, epoch=0, val_loss=0.5)
        _save(manager, epoch=5, val_loss=0.4)
        # max_keep=3, history has 2 entries — nothing should be removed
        assert (output_dir / "epoch_0000.pth").exists()
        assert (output_dir / "epoch_0005.pth").exists()

    def test_cleanup_handles_already_deleted_file(self, manager: CheckpointManager, output_dir: Path) -> None:
        for epoch in [0, 5, 10, 15]:
            _save(manager, epoch=epoch, val_loss=1.0 - epoch * 0.01)
        oldest = output_dir / "epoch_0000.pth"
        if oldest.exists():
            oldest.unlink()
        _save(manager, epoch=20, val_loss=0.5)  # _cleanup must not raise FileNotFoundError
