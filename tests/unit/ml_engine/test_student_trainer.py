"""
Unit tests for ml_engine.distillation.student_trainer.StudentTrainer.

Covers the two pure-Python surfaces that require no GPU or real model weights:
  1. _build_train_args(): config dict → ultralytics train() keyword dict
  2. Metric extraction: results.results_dict → gate-compatible metric dict

The full train() path (actual YOLO training) requires ultralytics + GPU and is
not tested here. Heavy deps are stubbed in sys.modules:
  - ultralytics: not installed in CI/test env
  - ml_engine.distillation.pseudo_label/utils: import cv2 → numpy DLL issue

Patch targets for train() tests:
  core.constants.PRETRAINED_MODELS_DIR  <- pretrained weights directory
  ultralytics.YOLO                      <- stubbed via sys.modules
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Pre-stub heavy deps before importing StudentTrainer.
# ml_engine.distillation.__init__ imports pseudo_label → auto_labeler → cv2 (DLL).
# ultralytics is not installed in the test environment.
_HEAVY_STUBS = {
    "ultralytics": MagicMock(),
    "ml_engine.distillation.pseudo_label": MagicMock(),
    "ml_engine.distillation.utils": MagicMock(),
}
for _k, _v in _HEAVY_STUBS.items():
    sys.modules.setdefault(_k, _v)

from ml_engine.distillation.student_trainer import StudentTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(tmp_path: Path, config: dict = None) -> StudentTrainer:
    """Build a StudentTrainer with the given config and a tmp output directory."""
    return StudentTrainer(
        data_yaml=str(tmp_path / "data.yaml"),
        model_name="yolov8s",
        config=config or {},
        output_dir=str(tmp_path),
    )


def _run_train_mocked(
    tmp_path: Path,
    results_dict,
    model_name: str = "yolov8s",
    create_best_pt: bool = True,
) -> tuple[str, dict]:
    """
    Run StudentTrainer.train() with ultralytics fully mocked.

    Creates a fake pretrained weights file so the FileNotFoundError guard
    passes, and optionally creates a fake best.pt so the path check passes.
    Returns the (best_pt_path, metrics) tuple from trainer.train().
    """
    # Fake pretrained weights so the existence check passes.
    (tmp_path / f"{model_name}.pt").write_bytes(b"")

    mock_results = MagicMock()
    mock_results.results_dict = results_dict
    # Point ultralytics save_dir to our tmp dir so the fallback path resolves.
    mock_results.save_dir = str(tmp_path / "student")

    if create_best_pt:
        best_dir = tmp_path / "student" / "weights"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "best.pt").write_bytes(b"")

    mock_model = MagicMock()
    mock_model.train.return_value = mock_results
    _HEAVY_STUBS["ultralytics"].YOLO.return_value = mock_model

    with patch("core.constants.PRETRAINED_MODELS_DIR", tmp_path):
        trainer = StudentTrainer(
            data_yaml=str(tmp_path / "data.yaml"),
            model_name=model_name,
            config={},
            output_dir=str(tmp_path),
        )
        return trainer.train()


# ---------------------------------------------------------------------------
# _build_train_args
# ---------------------------------------------------------------------------


class TestBuildTrainArgs:
    def test_default_epochs(self, tmp_path):
        args = _make_trainer(tmp_path)._build_train_args()
        assert args["epochs"] == 300

    def test_epochs_from_config(self, tmp_path):
        args = _make_trainer(tmp_path, {"training": {"epochs": 50}})._build_train_args()
        assert args["epochs"] == 50

    def test_batch_size_mapped_to_batch(self, tmp_path):
        args = _make_trainer(tmp_path, {"training": {"batch_size": 16}})._build_train_args()
        assert args["batch"] == 16

    def test_output_dir_as_project(self, tmp_path):
        args = _make_trainer(tmp_path)._build_train_args()
        assert args["project"] == str(tmp_path)
        assert args["name"] == "student"

    def test_lrf_normal_values(self, tmp_path):
        """Standard config: lr=1e-3, min_lr=1e-5 → lrf=0.01."""
        args = _make_trainer(
            tmp_path,
            {
                "training": {"learning_rate": 1e-3},
                "scheduler": {"min_lr": 1e-5},
            },
        )._build_train_args()
        assert args["lrf"] == pytest.approx(0.01)
        assert 0.0 < args["lrf"] <= 1.0

    def test_lrf_clamped_when_lr_smaller_than_min_lr(self, tmp_path):
        """lr=1e-6 < min_lr=1e-5 would give lrf=10 without the clamp."""
        args = _make_trainer(
            tmp_path,
            {
                "training": {"learning_rate": 1e-6},
                "scheduler": {"min_lr": 1e-5},
            },
        )._build_train_args()
        assert args["lrf"] <= 1.0

    def test_lrf_clamped_with_zero_lr(self, tmp_path):
        """lr=0 hits the 1e-8 floor → lrf=1000 without clamp."""
        args = _make_trainer(
            tmp_path,
            {
                "training": {"learning_rate": 0.0},
                "scheduler": {"min_lr": 1e-5},
            },
        )._build_train_args()
        assert args["lrf"] <= 1.0

    def test_aug_keys_passed_through(self, tmp_path):
        args = _make_trainer(tmp_path, {"augmentation": {"mosaic": 0.8, "fliplr": 0.3}})._build_train_args()
        assert args["mosaic"] == pytest.approx(0.8)
        assert args["fliplr"] == pytest.approx(0.3)

    def test_aug_key_absent_not_included(self, tmp_path):
        """aug_keys missing from config must NOT be forwarded (would override ultralytics defaults)."""
        args = _make_trainer(tmp_path, {"augmentation": {}})._build_train_args()
        assert "mosaic" not in args

    def test_warmup_epochs_from_scheduler(self, tmp_path):
        args = _make_trainer(tmp_path, {"scheduler": {"warmup_epochs": 7}})._build_train_args()
        assert args["warmup_epochs"] == 7

    def test_save_period_from_evaluation(self, tmp_path):
        args = _make_trainer(tmp_path, {"evaluation": {"interval": 5}})._build_train_args()
        assert args["save_period"] == 5


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


class TestMetricExtraction:
    def test_seg_model_maps_mAP50M_to_mIoU(self, tmp_path):
        _, metrics = _run_train_mocked(tmp_path, {"metrics/mAP50(M)": 0.72})
        assert metrics["mIoU"] == pytest.approx(0.72)

    def test_det_model_maps_mAP50B_to_mAP50(self, tmp_path):
        _, metrics = _run_train_mocked(tmp_path, {"metrics/mAP50(B)": 0.65})
        assert metrics["mAP50"] == pytest.approx(0.65)

    def test_both_keys_present(self, tmp_path):
        _, metrics = _run_train_mocked(
            tmp_path,
            {"metrics/mAP50(M)": 0.71, "metrics/mAP50(B)": 0.68},
        )
        assert metrics["mIoU"] == pytest.approx(0.71)
        assert metrics["mAP50"] == pytest.approx(0.68)

    def test_non_numeric_values_filtered_out(self, tmp_path):
        _, metrics = _run_train_mocked(
            tmp_path,
            {"metrics/mAP50(B)": 0.6, "class_names": ["cat", "dog"]},
        )
        assert "class_names" not in metrics

    def test_raw_ultralytics_keys_preserved(self, tmp_path):
        """Raw ultralytics keys are kept alongside the gate-compatible aliases."""
        _, metrics = _run_train_mocked(
            tmp_path,
            {"metrics/mAP50(M)": 0.72, "metrics/mAP75(M)": 0.55},
        )
        assert "metrics/mAP75(M)" in metrics
        assert metrics["metrics/mAP75(M)"] == pytest.approx(0.55)

    def test_missing_results_dict_returns_empty_metrics(self, tmp_path):
        """If results has no results_dict, metrics is {} and gate will escalate."""
        (tmp_path / "yolov8s.pt").write_bytes(b"")
        best_dir = tmp_path / "student" / "weights"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "best.pt").write_bytes(b"")

        # MagicMock with spec=[] has no attributes at all → hasattr returns False.
        mock_results = MagicMock(spec=[])
        mock_results.save_dir = str(tmp_path / "student")

        mock_model = MagicMock()
        mock_model.train.return_value = mock_results
        _HEAVY_STUBS["ultralytics"].YOLO.return_value = mock_model

        with patch("core.constants.PRETRAINED_MODELS_DIR", tmp_path):
            trainer = StudentTrainer(
                data_yaml=str(tmp_path / "data.yaml"),
                model_name="yolov8s",
                config={},
                output_dir=str(tmp_path),
            )
            _, metrics = trainer.train()

        assert metrics == {}


# ---------------------------------------------------------------------------
# best.pt path resolution
# ---------------------------------------------------------------------------


class TestBestPtPath:
    def test_missing_best_pt_raises_file_not_found(self, tmp_path):
        """Training that produces no weights should raise a clear error."""
        with pytest.raises(FileNotFoundError, match="No best.pt produced"):
            _run_train_mocked(tmp_path, {}, create_best_pt=False)

    def test_returns_absolute_best_pt_path(self, tmp_path):
        best_pt_path, _ = _run_train_mocked(tmp_path, {"metrics/mAP50(B)": 0.6})
        assert Path(best_pt_path).is_absolute()
        assert best_pt_path.endswith("best.pt")


# ---------------------------------------------------------------------------
# scheduler.type -> ultralytics cos_lr
#
# configs/defaults/distillation.yaml ships `scheduler.type: "cosine"`, but
# _build_train_args never reads `type`. ultralytics defaults cos_lr=False
# (linear decay), so every student trains on a schedule the config says it is
# not using -- a silently ignored knob, not a crash.
# ---------------------------------------------------------------------------


class TestSchedulerType:
    def test_cosine_scheduler_type_enables_cos_lr(self, tmp_path):
        args = _make_trainer(tmp_path, {"scheduler": {"type": "cosine"}})._build_train_args()
        assert args.get("cos_lr") is True

    def test_linear_scheduler_type_leaves_cos_lr_off(self, tmp_path):
        args = _make_trainer(tmp_path, {"scheduler": {"type": "linear"}})._build_train_args()
        assert args.get("cos_lr") is False

    def test_absent_scheduler_type_defaults_to_linear(self, tmp_path):
        """No scheduler block -> ultralytics' own default (linear) must stand."""
        args = _make_trainer(tmp_path, {})._build_train_args()
        assert args.get("cos_lr") is False

    def test_cosine_scheduler_type_is_case_insensitive(self, tmp_path):
        args = _make_trainer(tmp_path, {"scheduler": {"type": "Cosine"}})._build_train_args()
        assert args.get("cos_lr") is True
