"""
Unit tests for ml_engine.evaluation.evaluator.

The full ``evaluate_detection`` / ``evaluate_segmentation`` loops drive a real
torch model through a dataloader and hit GroundingDINO + SAM specifics. Those
paths are integration territory and live in the nightly integration suite
(`tests/integration/test_teacher_training.py` as a reference, and follow-up
`tests/integration/test_inference_pipelines.py` per TODOS.md #9).

Unit scope here covers the parts that break silently without a real model:
- Device selection falls back to CPU when CUDA is unavailable.
- Box-format conversion (``_cxcywh_norm_to_xyxy_pixel``) produces correct pixel
  coordinates, including the edge case of empty input.
- Construction wires the SimpleMetricsConverter correctly.
"""

from __future__ import annotations

import pytest
import torch

from ml_engine.evaluation.evaluator import (
    ModelEvaluator,
    _cxcywh_norm_to_xyxy_pixel,
)
from ml_engine.evaluation.metrics import SimpleMetricsConverter


class TestModelEvaluatorConstruction:
    """Smoke tests for constructor wiring — no real model required."""

    def test_default_device_falls_back_to_cpu_when_no_cuda(self) -> None:
        # On CI runners without CUDA, even device='cuda' must resolve to CPU.
        evaluator = ModelEvaluator(device="cuda")

        if not torch.cuda.is_available():
            assert evaluator.device.type == "cpu"
        else:
            assert evaluator.device.type == "cuda"

    def test_explicit_cpu_device(self) -> None:
        evaluator = ModelEvaluator(device="cpu")
        assert evaluator.device.type == "cpu"

    def test_confidence_threshold_honored(self) -> None:
        evaluator = ModelEvaluator(device="cpu", confidence_threshold=0.7)
        assert evaluator.confidence_threshold == 0.7

    def test_max_samples_for_viz_honored(self) -> None:
        evaluator = ModelEvaluator(device="cpu", max_samples_for_viz=5)
        assert evaluator.max_samples_for_viz == 5

    def test_simple_converter_wired(self) -> None:
        evaluator = ModelEvaluator(device="cpu")
        assert isinstance(evaluator.simple_converter, SimpleMetricsConverter)


class TestCxcywhNormToXyxyPixel:
    """Box-format converter — used inside detection eval loop."""

    def test_normalized_center_box_to_pixel(self) -> None:
        # A box at center (0.5, 0.5) with w=h=0.5 on a 100x100 image.
        # Pixel space: cx=50, cy=50, w=50, h=50 → xyxy = (25, 25, 75, 75).
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
        result = _cxcywh_norm_to_xyxy_pixel(boxes, img_h=100, img_w=100)

        assert result.shape == (1, 4)
        assert result[0].tolist() == pytest.approx([25.0, 25.0, 75.0, 75.0])

    def test_empty_input_returns_empty_tensor(self) -> None:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        result = _cxcywh_norm_to_xyxy_pixel(boxes, img_h=100, img_w=100)

        assert result.shape == (0, 4)

    def test_rectangular_image_scales_x_and_y_independently(self) -> None:
        # Image is 200 wide, 100 tall. Box centered at (0.5, 0.5) size (1.0, 1.0)
        # → full-image box in pixel space.
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
        result = _cxcywh_norm_to_xyxy_pixel(boxes, img_h=100, img_w=200)

        assert result[0].tolist() == pytest.approx([0.0, 0.0, 200.0, 100.0])

    def test_multiple_boxes(self) -> None:
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.5, 0.5],  # top-left quadrant
                [0.75, 0.75, 0.5, 0.5],  # bottom-right quadrant
            ],
            dtype=torch.float32,
        )
        result = _cxcywh_norm_to_xyxy_pixel(boxes, img_h=100, img_w=100)

        assert result.shape == (2, 4)
        assert result[0].tolist() == pytest.approx([0.0, 0.0, 50.0, 50.0])
        assert result[1].tolist() == pytest.approx([50.0, 50.0, 100.0, 100.0])
