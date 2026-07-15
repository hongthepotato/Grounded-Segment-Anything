"""
Unit tests for ml_engine.training.training_manager.TrainingManager.

Focuses on the AMP dtype routing: bfloat16 path runs without GradScaler,
float16 path creates a GradScaler, invalid dtype strings raise cleanly.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.nn as nn
import yaml


def _write_config(tmpdir: str, mixed_precision: dict) -> str:
    """Write a minimal training_dynamics YAML and return its path."""
    config = {
        "training_dynamics": {
            "mixed_precision": mixed_precision,
            "gradient_clipping": {"enabled": False},
            "gradient_accumulation": {"steps": 1},
            "normalization": {"freeze_bn_teacher": False},
        }
    }
    path = os.path.join(tmpdir, "td.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


def _make_manager(mixed_precision: dict, tmpdir: str):
    """Build a minimal TrainingManager for dtype-routing tests."""
    from ml_engine.training.training_manager import TrainingManager

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    path = _write_config(tmpdir, mixed_precision)
    return TrainingManager(model=model, optimizer=optimizer, config_path=path)


class TestAmpDtypeRouting(unittest.TestCase):
    """Verify mixed_precision.dtype config drives GradScaler creation."""

    def test_bfloat16_does_not_create_scaler(self):
        """bf16 shares FP32's exponent range — loss scaling is unnecessary."""
        with tempfile.TemporaryDirectory() as td:
            mgr = _make_manager({"enabled": True, "dtype": "bfloat16"}, td)
        self.assertTrue(mgr.use_amp)
        self.assertEqual(mgr.amp_dtype, torch.bfloat16)
        self.assertIsNone(mgr.scaler)

    def test_float16_creates_scaler(self):
        """fp16 needs GradScaler to prevent gradient underflow."""
        with tempfile.TemporaryDirectory() as td:
            mgr = _make_manager({"enabled": True, "dtype": "float16"}, td)
        self.assertTrue(mgr.use_amp)
        self.assertEqual(mgr.amp_dtype, torch.float16)
        self.assertIsNotNone(mgr.scaler)

    def test_default_dtype_is_bfloat16(self):
        """Omitting dtype in config defaults to bfloat16."""
        with tempfile.TemporaryDirectory() as td:
            mgr = _make_manager({"enabled": True}, td)
        self.assertEqual(mgr.amp_dtype, torch.bfloat16)
        self.assertIsNone(mgr.scaler)

    def test_dtype_aliases_accepted(self):
        """bf16 and fp16 short aliases map to the expected torch dtypes."""
        with tempfile.TemporaryDirectory() as td:
            mgr_bf = _make_manager({"enabled": True, "dtype": "bf16"}, td)
            self.assertEqual(mgr_bf.amp_dtype, torch.bfloat16)
        with tempfile.TemporaryDirectory() as td:
            mgr_fp = _make_manager({"enabled": True, "dtype": "fp16"}, td)
            self.assertEqual(mgr_fp.amp_dtype, torch.float16)

    def test_invalid_dtype_raises(self):
        """Unknown dtype string fails loudly rather than silently falling back."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                _make_manager({"enabled": True, "dtype": "float8"}, td)
            self.assertIn("mixed_precision.dtype", str(ctx.exception))

    def test_amp_off_has_no_scaler(self):
        """mixed_precision.enabled=false disables AMP entirely."""
        with tempfile.TemporaryDirectory() as td:
            mgr = _make_manager({"enabled": False, "dtype": "bfloat16"}, td)
        self.assertFalse(mgr.use_amp)
        self.assertIsNone(mgr.scaler)


class TestTrainingStepWithBfloat16(unittest.TestCase):
    """
    End-to-end sanity: bf16 AMP training step completes without errors and
    without requiring a GradScaler. Runs on CPU to avoid GPU dependency.
    """

    def test_bf16_training_step_no_scaler_no_crash(self):
        if not torch.cpu.is_available():
            self.skipTest("CPU always available — skip-guard for completeness")
        with tempfile.TemporaryDirectory() as td:
            mgr = _make_manager({"enabled": True, "dtype": "bfloat16"}, td)

        def compute_loss(batch):
            out = mgr.model(batch["x"])
            return {"loss": ((out - batch["y"]) ** 2).mean()}

        batch = {
            "x": torch.randn(3, 4),
            "y": torch.randn(3, 2),
        }
        # Must complete without raising
        loss_dict = mgr.training_step(batch, compute_loss)
        self.assertIn("loss", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]).any())


if __name__ == "__main__":
    unittest.main()
