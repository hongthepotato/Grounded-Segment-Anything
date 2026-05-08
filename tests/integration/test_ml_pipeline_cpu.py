"""Integration tests for ML pipeline loss functions and training config.

CPU-only — no GPU, no real model weights.  Three test groups:

  1. SegmentationLoss — exact analytical values, invariants, edge cases
  2. build_criterion (GroundingDINOCriterion) — callable, bounded, near-zero on perfect match
  3. build_teacher_training_config — YAML + DataManager contract + override merging

Analytical derivations (B=1, N=1, H=1, W=1, logit=0, target=1):
  sigmoid(0) = 0.5
  BCE(0,1) = log(2)
  focal: p_t=0.5, (1-p_t)²=0.25, alpha_t=0.25 → log(2)*0.25*0.25 = log(2)/16 ≈ 0.04332
  dice: intersection=0.5, (2*0.5+1)/(0.5+1+1) = 2/2.5 = 0.8 → loss = 0.2
  iou:  intersection=0.5, union=1.0, (0.5+1)/(1+1) = 0.75 → loss = 0.25
  total: 20*(log2/16) + 0.2 + 0.25 = 5*log2/4 + 0.45 ≈ 1.3164
"""

from __future__ import annotations

import math
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import torch

from ml_engine.training.losses import SegmentationLoss, build_criterion

pytestmark = pytest.mark.integration

# ── analytical constant ────────────────────────────────────────────────────
_SINGLE_PIXEL_TOTAL = 5 * math.log(2) / 4 + 0.45  # ≈ 1.3164


# ── helpers ───────────────────────────────────────────────────────────────


def _seg(
    logit: float = 0.0,
    target: float = 1.0,
    b: int = 1,
    n: int = 1,
    h: int = 1,
    w: int = 1,
    requires_grad: bool = False,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (predictions, targets) for SegmentationLoss with uniform values."""
    pred = torch.full((b, n, h, w), logit, requires_grad=requires_grad)
    preds = {"pred_masks": pred}
    tgts = {
        "masks": torch.full((b, n, h, w), target),
        "valid_mask": torch.ones(b, n, dtype=torch.bool),
    }
    return preds, tgts


def _gdino_outputs(batch: int = 1, num_queries: int = 5, num_tokens: int = 256) -> Dict[str, torch.Tensor]:
    return {
        "pred_logits": torch.rand(batch, num_queries, num_tokens),
        "pred_boxes": torch.rand(batch, num_queries, 4),
    }


def _gdino_targets(batch: int = 1, num_gt: int = 2, num_tokens: int = 256) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.zeros(num_gt, dtype=torch.long),
            "boxes": torch.rand(num_gt, 4),
            "token_labels": torch.zeros(num_gt, num_tokens),
        }
        for _ in range(batch)
    ]


@pytest.fixture(scope="module")
def mock_dm() -> MagicMock:
    dm = MagicMock()
    dm.get_dataset_info.return_value = {
        "num_classes": 2,
        "class_mapping": {0: "ok", 1: "defect"},
        "split": {"train": 80, "val": 20},
    }
    dm.get_required_models.return_value = {"grounding_dino": "gdino.pt"}
    return dm


# ══════════════════════════════════════════════════════════════════════════════
# 1. SegmentationLoss — basics
# ══════════════════════════════════════════════════════════════════════════════


class TestSegmentationLossBasics:
    def test_returns_scalar_non_negative(self) -> None:
        criterion = SegmentationLoss()
        preds, tgts = _seg()
        loss_dict = criterion(preds, tgts)
        total = loss_dict["loss"]
        assert total.shape == torch.Size([])
        assert total.dtype == torch.float32
        assert total.item() >= 0.0

    def test_all_five_keys_present(self) -> None:
        criterion = SegmentationLoss()
        preds, tgts = _seg()
        loss_dict = criterion(preds, tgts)
        for key in ("loss", "loss_focal", "loss_dice", "loss_iou", "loss_iou_quality"):
            assert key in loss_dict, f"missing key: {key}"

    def test_component_keys_are_detached(self) -> None:
        """Components should not carry gradients (they are for logging only)."""
        criterion = SegmentationLoss()
        preds, tgts = _seg(requires_grad=True)
        loss_dict = criterion(preds, tgts)
        for key in ("loss_focal", "loss_dice", "loss_iou", "loss_iou_quality"):
            assert not loss_dict[key].requires_grad, f"{key} should be detached"

    def test_main_loss_has_gradient(self) -> None:
        criterion = SegmentationLoss()
        preds, tgts = _seg(requires_grad=True)
        loss_dict = criterion(preds, tgts)
        assert loss_dict["loss"].requires_grad

    def test_gradient_flows_to_pred_masks(self) -> None:
        criterion = SegmentationLoss()
        preds, tgts = _seg(requires_grad=True)
        loss_dict = criterion(preds, tgts)
        loss_dict["loss"].backward()
        grad = preds["pred_masks"].grad
        assert grad is not None
        assert math.isfinite(grad.sum().item())

    def test_3d_input_raises_value_error(self) -> None:
        criterion = SegmentationLoss()
        with pytest.raises(ValueError, match="4D"):
            criterion({"pred_masks": torch.rand(2, 64, 64)}, {"masks": torch.rand(2, 64, 64)})


# ══════════════════════════════════════════════════════════════════════════════
# 2. SegmentationLoss — exact analytical values
# ══════════════════════════════════════════════════════════════════════════════


class TestSegmentationLossExactValues:
    """All expected values derived analytically from first principles (see module docstring)."""

    def test_total_single_pixel_logit0_target1(self) -> None:
        """5*log(2)/4 + 0.45 ≈ 1.3164 — catches wrong alpha, gamma, smooth, or formula."""
        criterion = SegmentationLoss()
        preds, tgts = _seg(logit=0.0, target=1.0, h=1, w=1)
        total = criterion(preds, tgts)["loss"].item()
        assert total == pytest.approx(_SINGLE_PIXEL_TOTAL, abs=5e-4)

    def test_focal_component_single_pixel(self) -> None:
        """Focal = log(2)/16 ≈ 0.04332 — isolate focal by zeroing other weights."""
        criterion = SegmentationLoss(loss_weights={"focal": 1.0, "dice": 0.0, "iou": 0.0, "iou_quality": 0.0})
        preds, tgts = _seg(logit=0.0, target=1.0, h=1, w=1)
        total = criterion(preds, tgts)["loss"].item()
        assert total == pytest.approx(math.log(2) / 16, abs=1e-5)

    def test_dice_component_single_pixel(self) -> None:
        """Dice = 1 - 2/2.5 = 0.2 — catches missing sigmoid or wrong smooth."""
        criterion = SegmentationLoss(loss_weights={"focal": 0.0, "dice": 1.0, "iou": 0.0, "iou_quality": 0.0})
        preds, tgts = _seg(logit=0.0, target=1.0, h=1, w=1)
        total = criterion(preds, tgts)["loss"].item()
        assert total == pytest.approx(0.2, abs=1e-5)

    def test_iou_component_single_pixel(self) -> None:
        """IoU = 1 - 1.5/2 = 0.25 — catches wrong union formula or missing sigmoid."""
        criterion = SegmentationLoss(loss_weights={"focal": 0.0, "dice": 0.0, "iou": 1.0, "iou_quality": 0.0})
        preds, tgts = _seg(logit=0.0, target=1.0, h=1, w=1)
        total = criterion(preds, tgts)["loss"].item()
        assert total == pytest.approx(0.25, abs=1e-5)

    def test_focal_reduction_is_mean_not_sum(self) -> None:
        """Focal loss over 64-pixel image must equal focal over 1-pixel image.

        If the implementation uses sum instead of mean, the 64-pixel version
        returns 64× the single-pixel value (≈ 2.77 vs 0.043).
        """
        criterion = SegmentationLoss(loss_weights={"focal": 1.0, "dice": 0.0, "iou": 0.0, "iou_quality": 0.0})
        preds_1x1, tgts_1x1 = _seg(logit=0.0, target=1.0, h=1, w=1)
        preds_8x8, tgts_8x8 = _seg(logit=0.0, target=1.0, h=8, w=8)
        loss_1x1 = criterion(preds_1x1, tgts_1x1)["loss"].item()
        loss_8x8 = criterion(preds_8x8, tgts_8x8)["loss"].item()
        # Mean: both should be log(2)/16.  Sum: 8x8 would be 64 × 1x1 value.
        assert loss_8x8 == pytest.approx(loss_1x1, abs=1e-5)

    def test_components_sum_to_total(self) -> None:
        """20*focal + dice + iou + iou_quality must equal the reported total."""
        criterion = SegmentationLoss()
        preds, tgts = _seg(logit=0.0, target=1.0, h=4, w=4)
        loss_dict = criterion(preds, tgts)
        recomputed = (
            20.0 * loss_dict["loss_focal"].item()
            + 1.0 * loss_dict["loss_dice"].item()
            + 1.0 * loss_dict["loss_iou"].item()
            + 1.0 * loss_dict["loss_iou_quality"].item()
        )
        assert recomputed == pytest.approx(loss_dict["loss"].item(), abs=1e-5)

    def test_target0_focal_is_3x_target1_focal(self) -> None:
        """For logit=0, target=0: focal = 3*log(2)/16 (alpha_t=0.75 vs 0.25 for target=1)."""
        criterion = SegmentationLoss(loss_weights={"focal": 1.0, "dice": 0.0, "iou": 0.0, "iou_quality": 0.0})
        preds_pos, tgts_pos = _seg(logit=0.0, target=1.0, h=1, w=1)
        preds_neg, tgts_neg = _seg(logit=0.0, target=0.0, h=1, w=1)
        loss_pos = criterion(preds_pos, tgts_pos)["loss"].item()
        loss_neg = criterion(preds_neg, tgts_neg)["loss"].item()
        # target=0: alpha_t=0.75; target=1: alpha_t=0.25 → ratio = 3
        assert loss_neg == pytest.approx(3.0 * loss_pos, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SegmentationLoss — invariants (monotonicity, bounds, behavior)
# ══════════════════════════════════════════════════════════════════════════════


class TestSegmentationLossInvariants:
    def test_perfect_prediction_near_zero(self) -> None:
        """logit=+100, target=1: sigmoid≈1, focal≈0, dice≈0, iou≈0 → total < 0.01."""
        criterion = SegmentationLoss()
        preds, tgts = _seg(logit=100.0, target=1.0, h=8, w=8)
        total = criterion(preds, tgts)["loss"].item()
        assert total < 0.01, f"perfect prediction should give near-zero loss, got {total:.6f}"

    def test_complete_failure_large_loss(self) -> None:
        """logit=-100, target=1: BCE≈100, (1-p_t)²≈1 → focal≈25, total > 100.

        Catches inverted focal weight (using p_t^gamma instead of (1-p_t)^gamma):
        that bug gives focal≈0 instead of ≈25, making total ≈ 2 instead of ≈ 500.
        """
        criterion = SegmentationLoss()
        preds, tgts = _seg(logit=-100.0, target=1.0, h=8, w=8)
        total = criterion(preds, tgts)["loss"].item()
        assert total > 100, f"complete failure should give large loss, got {total:.4f}"

    def test_loss_monotone_with_prediction_quality(self) -> None:
        """Loss should decrease as predictions improve: logit=-100 > logit=0 > logit=+100."""
        criterion = SegmentationLoss()
        loss_worst = criterion(*_seg(logit=-100.0, target=1.0, h=4, w=4))["loss"].item()
        loss_mid = criterion(*_seg(logit=0.0, target=1.0, h=4, w=4))["loss"].item()
        loss_best = criterion(*_seg(logit=100.0, target=1.0, h=4, w=4))["loss"].item()
        assert loss_worst > loss_mid > loss_best

    def test_all_false_valid_mask_returns_exactly_zero(self) -> None:
        criterion = SegmentationLoss()
        preds = {"pred_masks": torch.rand(2, 3, 8, 8)}
        tgts = {
            "masks": torch.rand(2, 3, 8, 8),
            "valid_mask": torch.zeros(2, 3, dtype=torch.bool),
        }
        loss_dict = criterion(preds, tgts)
        assert loss_dict["loss"].item() == 0.0

    def test_partial_valid_mask_ignores_invalid_masks(self) -> None:
        """Loss with valid_mask=[[True,False]] must equal loss on the single-valid-mask case."""
        criterion = SegmentationLoss()
        # Single valid mask — reference
        preds_single, tgts_single = _seg(logit=0.0, target=1.0, b=1, n=1, h=4, w=4)
        loss_single = criterion(preds_single, tgts_single)["loss"].item()

        # Same mask in slot 0, random junk in slot 1 (invalid)
        pred_two = torch.cat([preds_single["pred_masks"], torch.full((1, 1, 4, 4), -999.0)], dim=1)
        tgt_two_masks = torch.cat([tgts_single["masks"], torch.full((1, 1, 4, 4), 999.0)], dim=1)
        preds_partial = {"pred_masks": pred_two}
        tgts_partial = {
            "masks": tgt_two_masks,
            "valid_mask": torch.tensor([[True, False]]),
        }
        loss_partial = criterion(preds_partial, tgts_partial)["loss"].item()
        assert loss_partial == pytest.approx(loss_single, abs=1e-5)

    def test_valid_mask_defaults_to_all_true_when_absent(self) -> None:
        """Omitting valid_mask key should behave identically to all-True mask."""
        criterion = SegmentationLoss()
        preds, tgts_explicit = _seg(logit=0.0, target=1.0, b=2, n=3, h=4, w=4)
        tgts_implicit = {"masks": tgts_explicit["masks"]}  # no valid_mask key
        loss_explicit = criterion(preds, tgts_explicit)["loss"].item()
        loss_implicit = criterion(preds, tgts_implicit)["loss"].item()
        assert loss_explicit == pytest.approx(loss_implicit, abs=1e-6)

    def test_custom_loss_weights_scale_components(self) -> None:
        """SegmentationLoss(loss_weights={"focal":0, "dice":5, "iou":0}) → total = 5 * dice."""
        criterion = SegmentationLoss(loss_weights={"focal": 0.0, "dice": 5.0, "iou": 0.0, "iou_quality": 0.0})
        preds, tgts = _seg(logit=0.0, target=1.0, h=4, w=4)
        loss_dict = criterion(preds, tgts)
        expected = 5.0 * loss_dict["loss_dice"].item()
        assert loss_dict["loss"].item() == pytest.approx(expected, abs=1e-6)

    def test_iou_quality_branch_fires_when_iou_predictions_provided(self) -> None:
        """Providing iou_predictions should produce nonzero loss_iou_quality when miscalibrated."""
        criterion = SegmentationLoss()
        # logit=+100 → binary prediction = all 1s, target = all 1s → actual_iou ≈ 1.0
        preds, tgts = _seg(logit=100.0, target=1.0, b=1, n=1, h=8, w=8)
        preds["iou_predictions"] = torch.zeros(1, 1)  # predicts iou=0, actual≈1 → large MSE
        loss_dict = criterion(preds, tgts)
        assert loss_dict["loss_iou_quality"].item() > 0.5, (
            "miscalibrated iou_predictions should produce significant quality loss"
        )

    def test_iou_quality_zero_when_predictions_absent(self) -> None:
        criterion = SegmentationLoss()
        preds, tgts = _seg(logit=0.0, target=1.0)
        assert criterion(preds, tgts)["loss_iou_quality"].item() == 0.0

    def test_nan_predictions_propagate_to_loss(self) -> None:
        """NaN in pred_masks propagates to loss (not silently masked, not raised).

        Pinned contract: AMP `GradScaler.step()` and similar trainer-side
        guards rely on NaN propagating through the loss so they can detect
        the bad step and skip the optimizer update. If a future change
        starts raising or sanitizing on NaN inputs, those AMP guards break:
          - Masking to 0 → scaler sees finite loss, applies garbage gradients
          - Raising → training crashes instead of gracefully skipping
        Flip this test only after auditing trainer-side NaN handling in
        ml_engine/training/training_manager.py.
        """
        criterion = SegmentationLoss()
        preds = {"pred_masks": torch.full((1, 1, 4, 4), float("nan"))}
        tgts = {"masks": torch.ones(1, 1, 4, 4), "valid_mask": torch.ones(1, 1, dtype=torch.bool)}
        loss = criterion(preds, tgts)["loss"].item()
        assert math.isnan(loss), f"expected NaN propagation, got {loss}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. SegmentationLoss — error handling
# ══════════════════════════════════════════════════════════════════════════════


class TestSegmentationLossErrors:
    def test_mismatched_batch_size_raises(self) -> None:
        criterion = SegmentationLoss()
        preds = {"pred_masks": torch.rand(2, 1, 8, 8)}
        tgts = {"masks": torch.rand(3, 1, 8, 8), "valid_mask": torch.ones(3, 1, dtype=torch.bool)}
        with pytest.raises((RuntimeError, ValueError)):
            criterion(preds, tgts)

    def test_wrong_iou_predictions_shape_raises(self) -> None:
        """iou_predictions must be [B, N]; [B, N, K] multimask is a common mistake."""
        criterion = SegmentationLoss()
        preds, tgts = _seg(b=1, n=2, h=4, w=4)
        preds["iou_predictions"] = torch.rand(1, 2, 3)  # [B, N, K] — wrong shape
        with pytest.raises(ValueError, match=r"iou_predictions must be shape \[B, N\]"):
            criterion(preds, tgts)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GroundingDINOCriterion — basic correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCriterionBasics:
    def test_returns_non_empty_dict(self) -> None:
        criterion = build_criterion(num_classes=1)
        loss_dict = criterion(_gdino_outputs(), _gdino_targets())
        assert len(loss_dict) > 0

    def test_main_loss_keys_present(self) -> None:
        criterion = build_criterion(num_classes=1)
        loss_dict = criterion(_gdino_outputs(), _gdino_targets())
        for key in ("loss_ce", "loss_bbox", "loss_giou"):
            assert key in loss_dict

    def test_gradient_flows_through_losses(self) -> None:
        criterion = build_criterion(num_classes=1)
        outputs = {
            "pred_logits": torch.rand(1, 5, 256, requires_grad=True),
            "pred_boxes": torch.rand(1, 5, 4, requires_grad=True),
        }
        targets = _gdino_targets()
        loss_dict = criterion(outputs, targets)
        total = loss_dict["loss_ce"] + loss_dict["loss_bbox"] + loss_dict["loss_giou"]
        total.backward()
        assert outputs["pred_logits"].grad is not None
        assert outputs["pred_boxes"].grad is not None

    def test_missing_token_labels_raises_assertion(self) -> None:
        criterion = build_criterion(num_classes=1)
        bad_targets = [{"labels": torch.zeros(2, dtype=torch.long), "boxes": torch.rand(2, 4)}]
        with pytest.raises(AssertionError, match="token_labels required"):
            criterion(_gdino_outputs(), bad_targets)

    def test_boxes_wrong_shape_raises(self) -> None:
        """(M, 3) boxes instead of (M, 4) — catches shape contract violations."""
        criterion = build_criterion(num_classes=1)
        bad_targets = [
            {
                "labels": torch.zeros(2, dtype=torch.long),
                "boxes": torch.rand(2, 3),
                "token_labels": torch.zeros(2, 256),
            }
        ]
        with pytest.raises((RuntimeError, AssertionError)):
            criterion(_gdino_outputs(), bad_targets)

    def test_multi_class_produces_finite_losses(self) -> None:
        criterion = build_criterion(num_classes=10)
        loss_dict = criterion(_gdino_outputs(), _gdino_targets())
        assert all(math.isfinite(v.item()) for v in loss_dict.values())


# ══════════════════════════════════════════════════════════════════════════════
# 6. GroundingDINOCriterion — strict bounds
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCriterionBounds:
    def test_bbox_loss_non_negative(self) -> None:
        criterion = build_criterion(num_classes=1)
        loss_dict = criterion(_gdino_outputs(), _gdino_targets())
        assert loss_dict["loss_bbox"].item() >= 0.0

    def test_giou_loss_in_0_to_2(self) -> None:
        """GIoU(a,b) ∈ [-1,1] → loss_giou = 1-GIoU ∈ [0,2] per match."""
        criterion = build_criterion(num_classes=1)
        for _ in range(10):
            loss_dict = criterion(_gdino_outputs(), _gdino_targets())
            val = loss_dict["loss_giou"].item()
            assert 0.0 <= val <= 2.0, f"loss_giou={val:.4f} out of [0, 2]"

    def test_ce_loss_non_negative(self) -> None:
        criterion = build_criterion(num_classes=1)
        for _ in range(5):
            loss_dict = criterion(_gdino_outputs(), _gdino_targets())
            assert loss_dict["loss_ce"].item() >= 0.0

    def test_class_error_in_0_to_100(self) -> None:
        """class_error is a recall-based percentage; must stay in [0, 100]."""
        criterion = build_criterion(num_classes=1)
        for _ in range(5):
            loss_dict = criterion(_gdino_outputs(), _gdino_targets())
            err = loss_dict["class_error"].item()
            assert 0.0 <= err <= 100.0, f"class_error={err:.2f} out of [0, 100]"


# ══════════════════════════════════════════════════════════════════════════════
# 7. GroundingDINOCriterion — perfect detection near-zero
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCriterionPerfectDetection:
    def test_zero_bbox_loss_on_exact_box_match(self) -> None:
        """When predicted box equals GT box, L1=0 and GIoU=1 → bbox + giou losses ≈ 0."""
        criterion = build_criterion(num_classes=1)
        gt_box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])  # cxcywh, area=0.04 (non-degenerate)
        outputs = {
            "pred_logits": torch.full((1, 1, 256), -100.0),  # low confidence
            "pred_boxes": gt_box.unsqueeze(0),  # [1, 1, 4]
        }
        targets = [
            {
                "labels": torch.zeros(1, dtype=torch.long),
                "boxes": gt_box,
                "token_labels": torch.zeros(1, 256),
            }
        ]
        loss_dict = criterion(outputs, targets)
        assert loss_dict["loss_bbox"].item() == pytest.approx(0.0, abs=1e-5)
        assert loss_dict["loss_giou"].item() == pytest.approx(0.0, abs=1e-5)

    def test_near_zero_ce_on_low_confidence_zero_token_labels(self) -> None:
        """Predicting near-zero probability everywhere with zero token labels → loss_ce ≈ 0."""
        criterion = build_criterion(num_classes=1)
        outputs = {
            "pred_logits": torch.full((1, 1, 256), -100.0),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        }
        targets = [
            {
                "labels": torch.zeros(1, dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "token_labels": torch.zeros(1, 256),
            }
        ]
        loss_dict = criterion(outputs, targets)
        assert loss_dict["loss_ce"].item() < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 8. build_teacher_training_config
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildTeacherTrainingConfig:
    def test_required_keys_present_with_correct_values(self, mock_dm: MagicMock) -> None:
        from ml_engine.training.config import build_teacher_training_config

        config = build_teacher_training_config(mock_dm)
        assert "batch_size" in config
        assert "optimizer" in config
        assert "num_classes" in config
        assert config["num_classes"] == 2
        assert config["batch_size"] >= 1

    def test_lr_not_a_top_level_key(self, mock_dm: MagicMock) -> None:
        """teacher_training.yaml has no 'lr' key; this test catches future drift."""
        from ml_engine.training.config import build_teacher_training_config

        config = build_teacher_training_config(mock_dm)
        assert "lr" not in config

    def test_data_manager_methods_both_called(self) -> None:
        from ml_engine.training.config import build_teacher_training_config

        dm = MagicMock()
        dm.get_dataset_info.return_value = {
            "num_classes": 1,
            "class_mapping": {0: "defect"},
            "split": {"train": 90, "val": 10},
        }
        dm.get_required_models.return_value = {"grounding_dino": "x.pt"}
        build_teacher_training_config(dm)
        assert dm.get_dataset_info.called
        assert dm.get_required_models.called

    def test_no_required_models_raises_value_error(self) -> None:
        from ml_engine.training.config import build_teacher_training_config

        dm = MagicMock()
        dm.get_dataset_info.return_value = {
            "num_classes": 1,
            "class_mapping": {0: "defect"},
            "split": {"train": 90, "val": 10},
        }
        dm.get_required_models.return_value = {}
        with pytest.raises(ValueError, match="No models to train"):
            build_teacher_training_config(dm)

    def test_batch_size_override(self, mock_dm: MagicMock) -> None:
        from ml_engine.training.config import build_teacher_training_config

        config = build_teacher_training_config(mock_dm, overrides={"batch_size": 999})
        assert config["batch_size"] == 999

    def test_dotted_key_override_expands_to_nested(self, mock_dm: MagicMock) -> None:
        """'lora.r' → {'lora': {'r': 32}} via _expand_dotted_keys + merge_configs."""
        from ml_engine.training.config import build_teacher_training_config

        config = build_teacher_training_config(mock_dm, overrides={"lora.r": 32})
        assert config["lora"]["r"] == 32

    def test_missing_class_mapping_raises(self) -> None:
        from ml_engine.training.config import build_teacher_training_config

        dm = MagicMock()
        dm.get_dataset_info.return_value = {"num_classes": 1}  # no class_mapping
        dm.get_required_models.return_value = {"grounding_dino": "x.pt"}
        with pytest.raises((KeyError, ValueError)):
            build_teacher_training_config(dm)


# ══════════════════════════════════════════════════════════════════════════════
# 9. GroundingDINOCriterion — GIoU mathematical invariants
#
# Direct property tests on loss_giou via the build_criterion pipeline. These
# would have caught the +1e-6 epsilon bias in the original
# groundingdino.util.box_ops.box_iou denominator: that epsilon biases self-IoU
# as 1 - 1e-6/(area+1e-6), which scales as 1/area — invisible for normal-sized
# boxes (~2.5e-5 at side=0.2) but catastrophic for small ones (~1% loss at
# side=0.01, ~50% loss at side=0.001). loss_giou is multiplied by 2.0 in
# training, so this directly bleeds gradient signal away from small-object
# localization.
#
# Tests go through build_criterion() rather than calling generalized_box_iou
# directly so they don't couple to any specific GIoU import path.
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCriterionGIoUInvariants:
    @pytest.fixture(scope="class")
    def criterion(self):
        # build_criterion constructs GroundingDINOCriterion + HungarianMatcher
        # with the full aux/encoder weight_dict. Building once per class (instead
        # of ~33 times across these tests) shaves ~3s off the suite. Safe to
        # share: the criterion is a stateless nn.Module — forward() only reads
        # from self.
        return build_criterion(num_classes=1)

    @staticmethod
    def _giou_loss(criterion, pred_box: torch.Tensor, gt_box: torch.Tensor) -> float:
        """loss_giou for a 1-pred / 1-target setup. Uses very-low logits so
        classification cost is constant and matching is forced (1x1)."""
        outputs = {
            "pred_logits": torch.full((1, 1, 256), -100.0),
            "pred_boxes": pred_box.unsqueeze(0),
        }
        targets = [
            {
                "labels": torch.zeros(1, dtype=torch.long),
                "boxes": gt_box,
                "token_labels": torch.zeros(1, 256),
            }
        ]
        return criterion(outputs, targets)["loss_giou"].item()

    @pytest.mark.parametrize("side", [0.2, 0.05, 0.01])
    def test_self_match_loss_is_zero_across_box_sizes(self, criterion, side: float) -> None:
        """Perfect prediction must give loss_giou == 0 for ANY non-degenerate box.

        With the original +1e-6 in the IoU denominator the bias scales as
        1e-6/(side**2 + 1e-6): side=0.20 → 2.5e-5, side=0.05 → 4.0e-4,
        side=0.01 → 9.9e-3 (~1% loss on perfect prediction).
        """
        box = torch.tensor([[0.5, 0.5, side, side]])
        loss = self._giou_loss(criterion, box, box)
        predicted_eps_bias = 1e-6 / (side * side + 1e-6)
        assert loss == pytest.approx(0.0, abs=1e-6), (
            f"side={side}: loss_giou={loss:.3e}, expected ~0. "
            f"Predicted +1e-6 epsilon bias = {predicted_eps_bias:.3e}"
        )

    def test_self_match_loss_invariant_to_box_size(self, criterion) -> None:
        """Perfect-match loss_giou must not depend on box size.

        Catches the epsilon bug as a *ratio* property: even if absolute values
        are tolerated as float noise, a 400x spread across realistic box sizes
        is unambiguous evidence of a denominator bias.
        """
        sides = [0.2, 0.05, 0.01]
        losses = {
            s: self._giou_loss(criterion, torch.tensor([[0.5, 0.5, s, s]]), torch.tensor([[0.5, 0.5, s, s]]))
            for s in sides
        }
        max_loss = max(losses.values())
        assert max_loss < 1e-5, f"loss_giou varies with box size (denominator bias likely): {losses}"

    def test_giou_loss_symmetric(self, criterion) -> None:
        """GIoU is symmetric: loss(A as pred, B as gt) == loss(B as pred, A as gt)."""
        a = torch.tensor([[0.3, 0.4, 0.2, 0.15]])
        b = torch.tensor([[0.6, 0.5, 0.25, 0.3]])
        ab = self._giou_loss(criterion, a, b)
        ba = self._giou_loss(criterion, b, a)
        assert ab == pytest.approx(ba, abs=1e-6), f"asymmetric: {ab} vs {ba}"

    def test_giou_loss_translation_invariant(self, criterion) -> None:
        """Translating both boxes by the same offset preserves loss_giou."""
        a = torch.tensor([[0.3, 0.3, 0.2, 0.2]])
        b = torch.tensor([[0.4, 0.4, 0.2, 0.2]])
        offset = torch.tensor([[0.1, 0.1, 0.0, 0.0]])
        baseline = self._giou_loss(criterion, a, b)
        shifted = self._giou_loss(criterion, a + offset, b + offset)
        assert baseline == pytest.approx(shifted, abs=1e-6), f"translation-variant: {baseline} vs {shifted}"

    def test_giou_loss_monotone_under_separation(self, criterion) -> None:
        """As prediction translates away from target, loss_giou must monotonically
        increase. Catches sign / direction bugs in the GIoU enclosing-box term."""
        target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        prev = -float("inf")
        for shift in [0.0, 0.05, 0.1, 0.2, 0.4, 0.7]:
            pred = target + torch.tensor([[shift, 0.0, 0.0, 0.0]])
            loss = self._giou_loss(criterion, pred, target)
            assert loss >= prev - 1e-6, f"non-monotone at shift={shift}: loss={loss:.4f} < prev={prev:.4f}"
            prev = loss

    def test_giou_loss_upper_bound_for_disjoint_boxes(self, criterion) -> None:
        """Far-apart small boxes → loss_giou approaches 2.0 (giou → -1).

        pred at (0,0)-(0.05,0.05), gt at (0.95,0.95)-(1.0,1.0):
          union ≈ 0.005, enclosing area ≈ 1.0
          giou = 0 - (1.0 - 0.005)/1.0 ≈ -0.995
          loss = 1 - giou ≈ 1.995
        """
        pred = torch.tensor([[0.025, 0.025, 0.05, 0.05]])
        gt = torch.tensor([[0.975, 0.975, 0.05, 0.05]])
        loss = self._giou_loss(criterion, pred, gt)
        assert 1.9 < loss < 2.0, f"expected ~1.995, got {loss}"

    def test_giou_loss_in_valid_range(self, criterion) -> None:
        """For any pair of non-degenerate boxes: loss_giou ∈ [0, 2]."""
        torch.manual_seed(0)
        for _ in range(20):
            # cxcywh in [0.1, 0.9] with width/height in [0.05, 0.4]
            cxcy_a = torch.rand(2) * 0.8 + 0.1
            wh_a = torch.rand(2) * 0.35 + 0.05
            cxcy_b = torch.rand(2) * 0.8 + 0.1
            wh_b = torch.rand(2) * 0.35 + 0.05
            a = torch.cat([cxcy_a, wh_a]).unsqueeze(0)
            b = torch.cat([cxcy_b, wh_b]).unsqueeze(0)
            loss = self._giou_loss(criterion, a, b)
            assert 0.0 - 1e-6 <= loss <= 2.0 + 1e-6, f"out of range: {loss}"
