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

from ml_engine.utils.box_ops import box_iou, generalized_box_iou


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
        targets.append(
            {
                "labels": torch.zeros(M, dtype=torch.long, device=device),
                "boxes": torch.rand(M, 4, device=device),
                "token_labels": torch.randint(0, 2, (M, num_tokens), dtype=torch.float, device=device),
            }
        )
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
        aux_outputs = [{"pred_logits": torch.randn(B, N, T), "pred_boxes": torch.rand(B, N, 4)}]
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
            "aux_outputs": [{"pred_logits": aux_pred_logits, "pred_boxes": aux_pred_boxes}],
        }
        targets = _make_matcher_targets(B, M=3, num_tokens=T)

        loss_dict = self.criterion(outputs, targets)

        total_loss = sum(
            loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict if k in self.criterion.weight_dict
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


class TestMatcherRobustToInfLogits(unittest.TestCase):
    """
    Regression tests for the FP16/autocast NaN bug in HungarianMatcher.

    Root cause: GroundingDINO's ContrastiveEmbed fills padded text positions in
    pred_logits with -inf. The focal-cost chain (sigmoid + log(p+eps) + post-hoc
    mask multiply) produces `inf * 0 = NaN` at those positions under reduced
    precision. Fix: torch.where(text_mask, cost, 0.0) instead of cost * mask.float().

    The matcher is designed to run INSIDE an autocast context (see
    training_manager.training_step) — autocast handles op-level dtype routing
    (cdist/matmul auto-promote). These tests mirror that invocation pattern.
    """

    def _build_inputs_with_inf(self, B, N, num_tokens, num_valid, M, device="cpu"):
        """
        Simulate the bug shape: pred_logits with -inf at padding positions,
        finite values elsewhere; few valid tokens relative to num_tokens.
        Build in FP32 — the autocast context in each test downcasts as needed.
        """
        pred_logits = torch.full((B, N, num_tokens), float("-inf"), device=device)
        pred_logits[:, :, :num_valid] = torch.randn(B, N, num_valid, device=device)

        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": torch.rand(B, N, 4, device=device),
        }
        text_token_mask = torch.zeros(B, num_tokens, dtype=torch.bool, device=device)
        text_token_mask[:, :num_valid] = True
        outputs["text_token_mask"] = text_token_mask

        targets = []
        for _ in range(B):
            tl = torch.zeros(M, num_tokens, device=device)
            tl[:, :num_valid] = 1.0 / num_valid
            targets.append(
                {
                    "labels": torch.zeros(M, dtype=torch.long, device=device),
                    "boxes": torch.rand(M, 4, device=device),
                    "token_labels": tl,
                }
            )
        return outputs, targets

    def test_no_nan_with_inf_padding_fp32(self):
        """Baseline: matcher produces finite cost matrix in pure FP32."""
        from ml_engine.training.losses import HungarianMatcher

        matcher = HungarianMatcher()

        # Heavy padding shape mirroring the real bug: 4 valid tokens out of 256.
        outputs, targets = self._build_inputs_with_inf(B=1, N=900, num_tokens=256, num_valid=4, M=6)
        indices = matcher(outputs, targets)  # must not raise
        self.assertEqual(len(indices), 1)
        self.assertEqual(len(indices[0][0]), 6)  # one match per target

    def test_no_nan_with_inf_padding_under_bf16_autocast(self):
        """Inside autocast(bfloat16) — what production does on Ampere+/Ada."""
        if not torch.cuda.is_available():
            self.skipTest("autocast(cuda, bfloat16) requires CUDA")
        from ml_engine.training.losses import HungarianMatcher

        matcher = HungarianMatcher()
        outputs, targets = self._build_inputs_with_inf(
            B=1,
            N=900,
            num_tokens=256,
            num_valid=4,
            M=6,
            device="cuda",
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            indices = matcher(outputs, targets)
        self.assertEqual(len(indices), 1)

    def test_no_nan_with_inf_padding_under_fp16_autocast(self):
        """
        Inside autocast(float16) — the original bug reproduction path. With the
        torch.where fix, this passes despite 1e-8 underflowing in FP16.
        """
        if not torch.cuda.is_available():
            self.skipTest("autocast(cuda, float16) requires CUDA")
        from ml_engine.training.losses import HungarianMatcher

        matcher = HungarianMatcher()
        outputs, targets = self._build_inputs_with_inf(
            B=1,
            N=900,
            num_tokens=256,
            num_valid=4,
            M=6,
            device="cuda",
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            indices = matcher(outputs, targets)
        self.assertEqual(len(indices), 1)


class TestLossLabelsDtypeAgnostic(unittest.TestCase):
    """
    Regression: loss_labels must handle the case where pred_logits comes in as
    fp16/bf16 (autocast output) while token_labels is fp32 (explicit in
    dino_utils.py). index_put_ is strict about dtype match.

    These tests invoke the criterion inside an autocast context — the same way
    training_manager does in production.
    """

    def _build_inputs(self, B, N, T, M, device="cpu"):
        """FP32 inputs; autocast inside the test will downcast pred_logits."""
        outputs = {
            "pred_logits": torch.randn(B, N, T, device=device),
            "pred_boxes": torch.rand(B, N, 4, device=device),
        }
        targets = []
        for _ in range(B):
            targets.append(
                {
                    "labels": torch.zeros(M, dtype=torch.long, device=device),
                    "boxes": torch.rand(M, 4, device=device),
                    # token_labels stays fp32 by design (dino_utils.py:120)
                    "token_labels": torch.randint(0, 2, (M, T), dtype=torch.float32, device=device),
                }
            )
        return outputs, targets

    def test_loss_labels_under_fp16_autocast(self):
        """
        Mimics the exact shape that crashed production: pred_logits downcast
        to fp16 by autocast, token_labels still fp32 by design.
        """
        if not torch.cuda.is_available():
            self.skipTest("autocast(cuda, float16) requires CUDA")
        from ml_engine.training.losses import build_criterion

        criterion = build_criterion(num_classes=10, num_decoder_layers=2).cuda()

        outputs, targets = self._build_inputs(B=2, N=10, T=20, M=3, device="cuda")

        # Must not raise "Index put requires source and destination dtypes match"
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            losses = criterion(outputs, targets)
        self.assertIn("loss_ce", losses)
        for name, val in losses.items():
            self.assertFalse(torch.isnan(val).any(), f"NaN in {name}")

    def test_loss_labels_under_bf16_autocast(self):
        """Same regression in the bfloat16 path."""
        if not torch.cuda.is_available():
            self.skipTest("autocast(cuda, bfloat16) requires CUDA")
        from ml_engine.training.losses import build_criterion

        criterion = build_criterion(num_classes=10, num_decoder_layers=2).cuda()

        outputs, targets = self._build_inputs(B=2, N=10, T=20, M=3, device="cuda")

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = criterion(outputs, targets)
        self.assertIn("loss_ce", losses)


class TestLocalBoxIoU(unittest.TestCase):
    """Direct math tests on box_iou — bias-free IoU with clamp(min=1e-12)
    in the union denominator. The vendored groundingdino box_iou used `+ 1e-6`
    instead, which biases self-IoU below 1.0 with error 1e-6/(area+1e-6) —
    invisible at large boxes, ~1% at 1%×1%, ~50% at 0.1%×0.1%.
    """

    @staticmethod
    def _xyxy(cx: float, cy: float, w: float, h: float) -> torch.Tensor:
        return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])

    def test_self_match_iou_exactly_one_normal_box(self) -> None:

        box = self._xyxy(0.5, 0.5, 0.2, 0.2)
        iou, union = box_iou(box, box)
        self.assertAlmostEqual(iou.item(), 1.0, places=6)
        self.assertAlmostEqual(union.item(), 0.04, places=6)

    def test_self_match_iou_exactly_one_for_tiny_box(self) -> None:
        """0.1% × 0.1% box — old groundingdino code gave 0.5 here; should be 1.0."""

        box = self._xyxy(0.5, 0.5, 0.001, 0.001)
        iou, _ = box_iou(box, box)
        self.assertAlmostEqual(iou.item(), 1.0, places=6)

    def test_self_match_iou_exactly_one_for_microscopic_box(self) -> None:
        """1e-4 × 1e-4 box — old code gave ~0.01 (read perfect overlap as no match)."""

        box = self._xyxy(0.5, 0.5, 1e-4, 1e-4)
        iou, _ = box_iou(box, box)
        self.assertAlmostEqual(iou.item(), 1.0, places=6)

    def test_disjoint_boxes_iou_zero(self) -> None:

        a = self._xyxy(0.1, 0.1, 0.05, 0.05)
        b = self._xyxy(0.9, 0.9, 0.05, 0.05)
        iou, union = box_iou(a, b)
        self.assertAlmostEqual(iou.item(), 0.0, places=7)
        self.assertAlmostEqual(union.item(), 0.005, places=6)  # 2 × 0.05²

    def test_half_overlap_iou_one_third(self) -> None:
        """Two unit-side squares offset by 0.5 → inter=0.5, union=1.5, iou=1/3."""

        a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        b = torch.tensor([[0.5, 0.0, 1.5, 1.0]])
        iou, _ = box_iou(a, b)
        self.assertAlmostEqual(iou.item(), 1.0 / 3.0, places=6)

    def test_pairwise_shape_NxM(self) -> None:

        a = torch.rand(3, 4)
        a[:, 2:] = a[:, :2] + 0.1  # ensure x2>x1, y2>y1
        b = torch.rand(5, 4)
        b[:, 2:] = b[:, :2] + 0.1
        iou, union = box_iou(a, b)
        self.assertEqual(iou.shape, (3, 5))
        self.assertEqual(union.shape, (3, 5))

    def test_zero_area_point_box_no_nan(self) -> None:
        """Degenerate point box (area=0): clamp(min=1e-12) prevents 0/0 → NaN."""

        point = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # area=0, passes >= assertion
        iou, _ = box_iou(point, point)
        self.assertFalse(torch.isnan(iou).any().item())
        self.assertFalse(torch.isinf(iou).any().item())


class TestLocalGeneralizedBoxIoU(unittest.TestCase):
    """Direct math tests on generalized_box_iou."""

    @staticmethod
    def _xyxy(cx: float, cy: float, w: float, h: float) -> torch.Tensor:
        return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])

    def test_self_match_giou_exactly_one_across_box_sizes(self) -> None:

        for side in [0.5, 0.2, 0.05, 0.01, 0.001, 1e-4]:
            box = self._xyxy(0.5, 0.5, side, side)
            giou = generalized_box_iou(box, box).item()
            self.assertAlmostEqual(
                giou,
                1.0,
                places=6,
                msg=f"side={side}: giou={giou:.10f}, expected 1.0",
            )

    def test_disjoint_far_boxes_giou_near_minus_one(self) -> None:
        """Two tiny boxes at opposite corners: enclosing≈1.0, union≈0.005,
        giou ≈ 0 - (1.0 - 0.005)/1.0 ≈ -0.995."""

        a = self._xyxy(0.025, 0.025, 0.05, 0.05)
        b = self._xyxy(0.975, 0.975, 0.05, 0.05)
        giou = generalized_box_iou(a, b).item()
        self.assertLess(giou, -0.99)
        self.assertGreater(giou, -1.0)

    def test_giou_in_valid_range_random_pairs(self) -> None:

        torch.manual_seed(0)
        for _ in range(50):
            a = torch.rand(1, 4)
            a[:, 2:] = a[:, :2] + torch.rand(1, 2) * 0.4 + 0.05
            b = torch.rand(1, 4)
            b[:, 2:] = b[:, :2] + torch.rand(1, 2) * 0.4 + 0.05
            giou = generalized_box_iou(a, b).item()
            self.assertGreaterEqual(giou, -1.0 - 1e-6)
            self.assertLessEqual(giou, 1.0 + 1e-6)

    def test_giou_symmetric(self) -> None:

        a = self._xyxy(0.3, 0.4, 0.2, 0.15)
        b = self._xyxy(0.6, 0.5, 0.25, 0.3)
        ab = generalized_box_iou(a, b).item()
        ba = generalized_box_iou(b, a).item()
        self.assertAlmostEqual(ab, ba, places=7)

    def test_giou_translation_invariant(self) -> None:

        a = self._xyxy(0.3, 0.3, 0.2, 0.2)
        b = self._xyxy(0.4, 0.4, 0.2, 0.2)
        offset = torch.tensor([0.1, 0.1, 0.1, 0.1])
        baseline = generalized_box_iou(a, b).item()
        shifted = generalized_box_iou(a + offset, b + offset).item()
        self.assertAlmostEqual(baseline, shifted, places=6)

    def test_degenerate_xyxy_raises(self) -> None:
        """Strictly degenerate boxes (x2 < x1 or y2 < y1) must trigger assertion."""

        bad = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # x2 < x1, y2 < y1
        ok = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        with self.assertRaises(AssertionError):
            generalized_box_iou(bad, ok)
        with self.assertRaises(AssertionError):
            generalized_box_iou(ok, bad)

    def test_zero_area_point_box_no_nan(self) -> None:
        """Point box passes >= assertion; clamp prevents NaN in GIoU enclosing term."""

        point = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        giou = generalized_box_iou(point, point)
        self.assertFalse(torch.isnan(giou).any().item())
        self.assertFalse(torch.isinf(giou).any().item())

    def test_pairwise_shape_NxM(self) -> None:

        torch.manual_seed(1)
        a = torch.rand(3, 4)
        a[:, 2:] = a[:, :2] + 0.1
        b = torch.rand(5, 4)
        b[:, 2:] = b[:, :2] + 0.1
        giou = generalized_box_iou(a, b)
        self.assertEqual(giou.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
