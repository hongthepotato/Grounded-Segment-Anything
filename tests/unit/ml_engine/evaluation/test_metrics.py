"""
Unit tests for ml_engine.evaluation.metrics.

Highest silent-failure risk in the codebase. A wrong mIoU aggregation is
invisible until you compare to another implementation; the model trains,
eval runs, numbers come out, and nobody questions them. These tests use
hand-computable inputs to pin down exact expected values.

Structure:
- TestSegmentationMetrics: exact IoU / Dice / precision / recall against
  constructed masks with known geometric overlap.
- TestDetectionMetrics: shape of compute() output and handling of empty
  prediction sets.
- TestSimpleMetricsConverter: grade thresholds + per-class summary output.
"""

from __future__ import annotations

import pytest
import torch

from ml_engine.evaluation.metrics import (
    DetectionMetrics,
    SegmentationMetrics,
    SimpleMetricsConverter,
)

# ============================================================================
# SegmentationMetrics — hand-computable IoU / Dice on constructed masks.
# ============================================================================


def _rect_mask(h: int, w: int, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Build a binary mask with a solid rectangle at (y0, x0, y1, x1)."""
    m = torch.zeros((h, w), dtype=torch.float32)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = 1.0
    return m


class TestSegmentationMetricsPerfectMatch:
    """Identical pred and gt masks → IoU=1.0, Dice=1.0, recall=1.0, precision=1.0."""

    def test_perfect_overlap_one_class(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["foreground"])

        mask = _rect_mask(h=10, w=10, box=(2, 2, 6, 6)).unsqueeze(0)  # [1, 10, 10]
        pred = {"masks": mask, "labels": torch.tensor([0]), "scores": torch.tensor([1.0])}
        tgt = {"masks": mask.clone(), "labels": torch.tensor([0])}

        metrics.update([pred], [tgt])
        result = metrics.compute()

        assert result["mIoU"] == pytest.approx(1.0, abs=1e-6)
        assert result["mean_dice"] == pytest.approx(1.0, abs=1e-6)
        assert result["precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)


class TestSegmentationMetricsPartialOverlap:
    """Pred box overlaps half of gt box → IoU = 0.5 / 1.5 = 0.333..."""

    def test_half_overlap(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["fg"], iou_threshold=0.3)

        gt = _rect_mask(h=10, w=10, box=(0, 0, 4, 4)).unsqueeze(0)  # 16 px
        pred = _rect_mask(h=10, w=10, box=(2, 0, 6, 4)).unsqueeze(0)  # 16 px
        # intersection = rows 2..4, cols 0..4 = 2 * 4 = 8 px
        # union = 16 + 16 - 8 = 24 px
        # IoU = 8/24 = 1/3

        metrics.update(
            [{"masks": pred, "labels": torch.tensor([0]), "scores": torch.tensor([1.0])}],
            [{"masks": gt, "labels": torch.tensor([0])}],
        )
        result = metrics.compute()

        assert result["per_class_iou"]["fg"] == pytest.approx(1 / 3, abs=1e-3)
        # Dice = 2*8 / (16+16) = 0.5
        assert result["per_class_dice"]["fg"] == pytest.approx(0.5, abs=1e-3)


class TestSegmentationMetricsMissedDetections:
    """Unmatched ground-truth boxes must count as IoU=0 — penalizes missed detections."""

    def test_unmatched_gt_drops_miou(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["fg"], iou_threshold=0.5)

        # Two GTs, but only one prediction that matches the first.
        gt_a = _rect_mask(10, 10, (0, 0, 4, 4)).unsqueeze(0)
        gt_b = _rect_mask(10, 10, (5, 5, 9, 9)).unsqueeze(0)
        gts = torch.cat([gt_a, gt_b], dim=0)

        pred = gt_a.clone()  # perfect match for gt_a only

        metrics.update(
            [{"masks": pred, "labels": torch.tensor([0]), "scores": torch.tensor([1.0])}],
            [{"masks": gts, "labels": torch.tensor([0, 0])}],
        )
        result = metrics.compute()

        # Per-class IoU is mean of [1.0, 0.0] = 0.5 — unmatched GT pulls it down.
        assert result["per_class_iou"]["fg"] == pytest.approx(0.5, abs=1e-3)
        # recall = tp / (tp + fn) = 1 / (1 + 1) = 0.5
        assert result["recall"] == pytest.approx(0.5, abs=1e-3)


class TestSegmentationMetricsEmptyInputs:
    """Empty predictions + non-empty GT → all false negatives. Empty GT + preds → all FP."""

    def test_empty_preds_all_fn(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["fg"])

        gt = _rect_mask(10, 10, (0, 0, 4, 4)).unsqueeze(0)
        empty_mask = torch.zeros((0, 10, 10), dtype=torch.float32)

        metrics.update(
            [{"masks": empty_mask, "labels": torch.tensor([], dtype=torch.long), "scores": torch.tensor([])}],
            [{"masks": gt, "labels": torch.tensor([0])}],
        )
        result = metrics.compute()

        assert result["precision"] == pytest.approx(0.0, abs=1e-6)  # 0 TP / (0 TP + 0 FP)
        assert result["recall"] == pytest.approx(0.0, abs=1e-6)  # 0 / 1

    def test_empty_gt_all_fp(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["fg"])

        pred = _rect_mask(10, 10, (0, 0, 4, 4)).unsqueeze(0)
        empty_gt = torch.zeros((0, 10, 10), dtype=torch.float32)

        metrics.update(
            [{"masks": pred, "labels": torch.tensor([0]), "scores": torch.tensor([1.0])}],
            [{"masks": empty_gt, "labels": torch.tensor([], dtype=torch.long)}],
        )
        result = metrics.compute()

        # precision = 0 TP / (0 + 1 FP) = 0
        assert result["precision"] == pytest.approx(0.0, abs=1e-6)


class TestSegmentationMetricsReset:
    def test_reset_clears_state(self) -> None:
        metrics = SegmentationMetrics(num_classes=1, class_names=["fg"])

        mask = _rect_mask(10, 10, (0, 0, 4, 4)).unsqueeze(0)
        metrics.update(
            [{"masks": mask, "labels": torch.tensor([0]), "scores": torch.tensor([1.0])}],
            [{"masks": mask, "labels": torch.tensor([0])}],
        )
        metrics.reset()

        result = metrics.compute()
        assert result["mIoU"] == 0.0
        assert result["precision"] == pytest.approx(0.0, abs=1e-6)
        assert result["recall"] == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# DetectionMetrics — shape of output + empty-input handling.
# ============================================================================


class TestDetectionMetricsShape:
    """Output dict must carry all keys downstream code expects."""

    def test_compute_returns_expected_keys(self) -> None:
        metrics = DetectionMetrics(num_classes=2, class_names=["a", "b"])

        preds = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        tgts = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([0]),
            }
        ]
        metrics.update(preds, tgts)
        result = metrics.compute()

        assert "mAP50" in result
        assert "mAP50_95" in result
        assert "per_class_ap" in result
        assert "per_class_counts" in result

    def test_per_class_counts_tracked(self) -> None:
        metrics = DetectionMetrics(num_classes=2, class_names=["a", "b"])

        preds = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        tgts = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32),
                "labels": torch.tensor([0, 1]),
            }
        ]
        metrics.update(preds, tgts)
        result = metrics.compute()

        assert result["per_class_counts"]["a"] == 1
        assert result["per_class_counts"]["b"] == 1


class TestDetectionMetricsReset:
    def test_reset_clears_counts(self) -> None:
        metrics = DetectionMetrics(num_classes=1, class_names=["a"])

        preds = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        tgts = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "labels": torch.tensor([0]),
            }
        ]
        metrics.update(preds, tgts)
        assert metrics.class_counts[0] == 1
        metrics.reset()

        assert metrics.class_counts[0] == 0


# ============================================================================
# SimpleMetricsConverter — grade thresholds + summary generation.
# ============================================================================


class TestSimpleMetricsConverterGrades:
    @pytest.mark.parametrize(
        "score,expected_grade",
        [
            (95.0, "Excellent"),
            (85.0, "Very Good"),
            (75.0, "Good"),
            (65.0, "Average"),
            (55.0, "Needs Improvement"),
            (10.0, "Poor"),
        ],
    )
    def test_grade_thresholds(self, score: float, expected_grade: str) -> None:
        converter = SimpleMetricsConverter()
        grade = converter._get_grade(score)
        assert grade == expected_grade


class TestSimpleMetricsConverterDetection:
    def test_converts_high_mAP_to_excellent(self) -> None:
        converter = SimpleMetricsConverter()
        technical = {
            "mAP50": 0.95,
            "mAP50_95": 0.80,
            "per_class_ap": {"dog": 0.95},
            "per_class_counts": {"dog": 100},
        }
        simple = converter.convert_detection(technical)

        assert simple["overall_score"] == 95.0
        assert simple["grade"] == "Excellent"
        assert "95" in simple["summary"]
        assert simple["per_class"][0]["class"] == "dog"

    def test_low_sample_count_warning(self) -> None:
        converter = SimpleMetricsConverter()
        technical = {
            "mAP50": 0.9,
            "per_class_ap": {"rare_class": 0.9},
            "per_class_counts": {"rare_class": 10},
        }
        simple = converter.convert_detection(technical)

        assert "Low sample count" in simple["per_class"][0]["warning"]


class TestSimpleMetricsConverterSegmentation:
    def test_segmentation_summary_contains_rates(self) -> None:
        converter = SimpleMetricsConverter()
        technical = {
            "mIoU": 0.75,
            "mean_dice": 0.82,
            "recall": 0.9,
            "per_class_iou": {"tumor": 0.75},
            "per_class_counts": {"tumor": 100},
        }
        simple = converter.convert_segmentation(technical)

        assert simple["overall_score"] == 75.0
        assert simple["coverage_rate"] == 90.0
        assert simple["quality_rate"] == 82.0
        assert simple["grade"] == "Good"
