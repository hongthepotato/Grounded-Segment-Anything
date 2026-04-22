"""
Unit tests for loss functions.

Tests:
- HungarianMatcher: bipartite matching with token-level classification cost
- GroundingDINOCriterion: full DETR-style loss (labels + boxes + aux)
- build_criterion: sanity check on weight dict construction
- SegmentationLoss: forward pass including all-invalid-masks edge case
"""

import unittest
import torch
import torch.nn as nn


def _make_matcher_outputs(B: int, N: int, num_tokens: int, device="cpu"):
    """Synthetic HungarianMatcher inputs."""
    outputs = {
        "pred_logits": torch.randn(B, N, num_tokens, device=device),
        "pred_boxes": torch.rand(B, N, 4, device=device),
    }
    return outputs


def _make_matcher_targets(B: int, M: int, num_tokens: int, device="cpu"):
    """Synthetic targets with M objects per image."""
    targets = []
    for _ in range(B):
        targets.append({
            "labels": torch.zeros(M, dtype=torch.long, device=device),
            "boxes": torch.rand(M, 4, device=device),
            "token_labels": torch.randint(0, 2, (M, num_tokens), dtype=torch.float, device=device),
        })
    return targets


class TestHungarianMatcher(unittest.TestCase):
    """Tests for HungarianMatcher bipartite matching."""

    def setUp(self):
        from ml_engine.training.losses import HungarianMatcher
        self.matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

    def test_output_shape(self):
        """Each batch element gets one (pred_idx, tgt_idx) tuple."""
        B, N, M, T = 2, 10, 3, 20
        outputs = _make_matcher_outputs(B, N, T)
        targets = _make_matcher_targets(B, M, T)

        indices = self.matcher(outputs, targets)

        self.assertEqual(len(indices), B)
        for pred_idx, tgt_idx in indices:
            # Each matched pair has the same length
            self.assertEqual(len(pred_idx), len(tgt_idx))
            # Indices within valid range
            self.assertTrue((pred_idx < N).all())
            self.assertTrue((tgt_idx < M).all())

    def test_single_target(self):
        """With 1 target per image, exactly 1 match per image."""
        B, N, M, T = 2, 10, 1, 20
        outputs = _make_matcher_outputs(B, N, T)
        targets = _make_matcher_targets(B, M, T)

        indices = self.matcher(outputs, targets)

        for pred_idx, tgt_idx in indices:
            self.assertEqual(len(pred_idx), 1)

    def test_with_text_token_mask(self):
        """Matcher handles text_token_mask correctly (no crash)."""
        B, N, M, T = 2, 10, 3, 20
        outputs = _make_matcher_outputs(B, N, T)
        outputs["text_token_mask"] = torch.ones(B, T, dtype=torch.bool)
        targets = _make_matcher_targets(B, M, T)

        indices = self.matcher(outputs, targets)
        self.assertEqual(len(indices), B)


class TestGroundingDINOCriterion(unittest.TestCase):
    """Tests for GroundingDINOCriterion full forward pass."""

    def setUp(self):
        from ml_engine.training.losses import build_criterion
        self.criterion = build_criterion(num_classes=80, num_decoder_layers=2)

    def test_forward_returns_expected_keys(self):
        """Forward pass produces loss_ce, loss_bbox, loss_giou keys."""
        B, N, T = 2, 10, 20
        outputs = {
            "pred_logits": torch.randn(B, N, T),
            "pred_boxes": torch.rand(B, N, 4),
        }
        targets = _make_matcher_targets(B, M=3, num_tokens=T)

        losses = self.criterion(outputs, targets)

        self.assertIn("loss_ce", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss_giou", losses)

    def test_auxiliary_losses(self):
        """Auxiliary decoder losses are keyed with _0, _1, ... suffixes."""
        B, N, T = 2, 10, 20
        aux_outputs = [
            {"pred_logits": torch.randn(B, N, T), "pred_boxes": torch.rand(B, N, 4)}
        ]
        outputs = {
            "pred_logits": torch.randn(B, N, T),
            "pred_boxes": torch.rand(B, N, 4),
            "aux_outputs": aux_outputs,
        }
        targets = _make_matcher_targets(B, M=2, num_tokens=T)

        losses = self.criterion(outputs, targets)

        self.assertIn("loss_ce_0", losses)
        self.assertIn("loss_bbox_0", losses)

    def test_losses_are_finite(self):
        """No NaN or Inf in any loss component."""
        B, N, T = 2, 10, 20
        outputs = {
            "pred_logits": torch.randn(B, N, T),
            "pred_boxes": torch.rand(B, N, 4),
        }
        targets = _make_matcher_targets(B, M=3, num_tokens=T)

        losses = self.criterion(outputs, targets)

        for name, val in losses.items():
            self.assertFalse(torch.isnan(val).any(), f"NaN in {name}")
            self.assertFalse(torch.isinf(val).any(), f"Inf in {name}")

    def test_enc_outputs_produces_enc_keys(self):
        """
        enc_outputs path produces loss_*_enc keys.

        The encoder uses binary objectness targets (token_labels=all-ones, labels=all-zeros)
        so every valid token is equally activated — different signal from the decoder path.
        This test verifies the path runs and keys are present.
        """
        B, N, T = 2, 10, 20
        outputs = {
            "pred_logits": torch.randn(B, N, T),
            "pred_boxes": torch.rand(B, N, 4),
            "enc_outputs": {
                "pred_logits": torch.randn(B, N, T),
                "pred_boxes": torch.rand(B, N, 4),
            },
        }
        targets = _make_matcher_targets(B, M=2, num_tokens=T)

        losses = self.criterion(outputs, targets)

        self.assertIn("loss_ce_enc", losses)
        self.assertIn("loss_bbox_enc", losses)
        self.assertIn("loss_giou_enc", losses)
        for name in ("loss_ce_enc", "loss_bbox_enc", "loss_giou_enc"):
            self.assertFalse(torch.isnan(losses[name]).any(), f"NaN in {name}")

    def test_backward_gradient_flow(self):
        """
        Regression: weighted total loss must support .backward() with no NaN gradients.

        A NaN in any auxiliary loss component would silently zero out or corrupt gradients
        if not caught here. This test mirrors what GroundingDINOTrainer.compute_loss() does.
        """
        B, N, T = 2, 10, 20
        pred_logits = torch.randn(B, N, T, requires_grad=True)
        pred_boxes = torch.rand(B, N, 4, requires_grad=True)
        aux_pred_logits = torch.randn(B, N, T, requires_grad=True)
        aux_pred_boxes = torch.rand(B, N, 4, requires_grad=True)

        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "aux_outputs": [
                {"pred_logits": aux_pred_logits, "pred_boxes": aux_pred_boxes}
            ],
        }
        targets = _make_matcher_targets(B, M=3, num_tokens=T)

        loss_dict = self.criterion(outputs, targets)

        total_loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k]
            for k in loss_dict
            if k in self.criterion.weight_dict
        )

        # Must not raise
        total_loss.backward()

        # Gradients must exist and be finite
        for name, param in [("pred_logits", pred_logits), ("pred_boxes", pred_boxes)]:
            self.assertIsNotNone(param.grad, f"No gradient on {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient on {name}")


class TestBuildCriterion(unittest.TestCase):
    """Tests for build_criterion weight dict construction."""

    def test_weight_dict_covers_all_decoder_layers(self):
        """weight_dict has entries for all aux layers and encoder."""
        from ml_engine.training.losses import build_criterion
        num_layers = 4
        criterion = build_criterion(num_classes=10, num_decoder_layers=num_layers)

        # Aux losses: 0 .. num_layers-2
        for i in range(num_layers - 1):
            self.assertIn(f"loss_ce_{i}", criterion.weight_dict)
            self.assertIn(f"loss_bbox_{i}", criterion.weight_dict)
            self.assertIn(f"loss_giou_{i}", criterion.weight_dict)

        # Encoder
        self.assertIn("loss_ce_enc", criterion.weight_dict)
        self.assertIn("loss_bbox_enc", criterion.weight_dict)
        self.assertIn("loss_giou_enc", criterion.weight_dict)

    def test_weight_values(self):
        """Paper weights: loss_ce=2.0, loss_bbox=5.0, loss_giou=2.0."""
        from ml_engine.training.losses import build_criterion
        criterion = build_criterion(num_classes=10)

        self.assertAlmostEqual(criterion.weight_dict["loss_ce"], 2.0)
        self.assertAlmostEqual(criterion.weight_dict["loss_bbox"], 5.0)
        self.assertAlmostEqual(criterion.weight_dict["loss_giou"], 2.0)


class TestSegmentationLoss(unittest.TestCase):
    """Tests for SegmentationLoss, focusing on edge cases."""

    def setUp(self):
        from ml_engine.training.losses import SegmentationLoss
        self.loss_fn = SegmentationLoss()

    def test_all_invalid_masks_backward(self):
        """
        Regression: all-invalid valid_mask must not crash .backward().

        Zero tensors without requires_grad=True cause:
            RuntimeError: element 0 of tensors does not require grad
        """
        B, N, H, W = 2, 3, 16, 16
        pred_masks = torch.randn(B, N, H, W, requires_grad=True)
        predictions = {"pred_masks": pred_masks}
        targets = {
            "masks": torch.zeros(B, N, H, W),
            "valid_mask": torch.zeros(B, N, dtype=torch.bool),  # all invalid
        }

        loss_dict = self.loss_fn(predictions, targets)

        # Must not raise — this was the bug
        loss_dict["loss"].backward()

    def test_all_invalid_masks_returns_zero(self):
        """All-invalid path returns zero loss."""
        B, N, H, W = 2, 3, 16, 16
        predictions = {"pred_masks": torch.randn(B, N, H, W)}
        targets = {
            "masks": torch.zeros(B, N, H, W),
            "valid_mask": torch.zeros(B, N, dtype=torch.bool),
        }

        loss_dict = self.loss_fn(predictions, targets)

        self.assertEqual(loss_dict["loss"].item(), 0.0)

    def test_happy_path_produces_valid_loss(self):
        """Standard forward pass with all-valid masks."""
        B, N, H, W = 2, 3, 16, 16
        predictions = {"pred_masks": torch.randn(B, N, H, W)}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": torch.ones(B, N, dtype=torch.bool),
        }

        loss_dict = self.loss_fn(predictions, targets)

        self.assertFalse(torch.isnan(loss_dict["loss"]))
        self.assertGreater(loss_dict["loss"].item(), 0.0)

    def test_partial_valid_mask(self):
        """Partial valid_mask uses only valid entries."""
        B, N, H, W = 2, 4, 16, 16
        valid_mask = torch.zeros(B, N, dtype=torch.bool)
        valid_mask[:, :2] = True  # first 2 objects valid

        predictions = {"pred_masks": torch.randn(B, N, H, W)}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": valid_mask,
        }

        loss_dict = self.loss_fn(predictions, targets)

        self.assertFalse(torch.isnan(loss_dict["loss"]))


if __name__ == "__main__":
    unittest.main()
