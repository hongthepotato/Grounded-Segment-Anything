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


class TestBoxOpsDtypeSafety(unittest.TestCase):
    """Dtype-safety regression coverage for ``box_iou`` / ``generalized_box_iou``.

    Issue #91 (closes TODO #42) promotes fp16/bf16 inputs to fp32 at function
    entry, then runs all IoU/GIoU math in fp32 (or fp64 for fp64 inputs). The
    ``clamp(min=torch.finfo(t.dtype).tiny)`` final guard now fires only on
    genuinely zero-area boxes, where its floor (1.18e-38 fp32 / 2.22e-308 fp64)
    turns ``0/0`` into ``0/tiny == 0`` so callers see IoU = 0 instead of NaN.

    Output dtype contract:

    * fp16 / bf16 input → fp32 output (matches the pre-#91 de-facto behavior
      via torchvision's incidental upcast — see canary below).
    * fp32 input → fp32 output.
    * fp64 input → fp64 output (preserved, never downcast).

    Pre-#91 history (kept for context): #86 introduced the dtype-aware clamp
    (``finfo(t.dtype).tiny`` instead of literal ``1e-12``) to prevent fp16 NaN
    on degenerate boxes. The adversarial review during the #86 ship caught that
    this distorted small *non-degenerate* fp16 boxes by ~40% on the
    ``enclosing`` denominator (which the union-side torchvision upcast didn't
    protect). #91's fp32 promotion eliminates that distortion — verified in
    ``test_distinct_pair_giou_small_boxes_match_fp64_on_same_coords``.
    """

    @staticmethod
    def _xyxy(cx: float, cy: float, w: float, h: float, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]], dtype=dtype)

    # Mantissa precision: fp16 ~3 decimals, bf16 ~2 (worse than fp16!), fp32 ~7,
    # fp64 ~15. bf16 is the production AMP default in training_manager.
    _DTYPES_PLACES = [
        (torch.float16, 3),
        (torch.bfloat16, 2),
        (torch.float32, 6),
        (torch.float64, 12),
    ]

    def test_self_iou_one_across_dtypes(self) -> None:
        """Self-IoU == 1.0 for a normal box in fp16, bf16, fp32, fp64."""
        for dtype, places in self._DTYPES_PLACES:
            with self.subTest(dtype=str(dtype)):
                box = self._xyxy(0.5, 0.5, 0.1, 0.1, dtype=dtype)
                iou, _ = box_iou(box, box)
                self.assertAlmostEqual(iou.item(), 1.0, places=places)

    def test_self_giou_one_across_dtypes(self) -> None:
        """Self-GIoU == 1.0 for a normal box in fp16, bf16, fp32, fp64."""
        for dtype, places in self._DTYPES_PLACES:
            with self.subTest(dtype=str(dtype)):
                box = self._xyxy(0.5, 0.5, 0.1, 0.1, dtype=dtype)
                giou = generalized_box_iou(box, box)
                self.assertAlmostEqual(giou.item(), 1.0, places=places)

    def test_distinct_pair_giou_normal_boxes_close_to_truth(self) -> None:
        """Non-self distinct pair, normal-sized boxes, GIoU close to fp64 truth.

        This is what self-pair tests can't catch: with self-pairs,
        ``(enclosing - union) == 0`` so the GIoU enclosing term drops out.
        With a distinct pair we exercise the enclosing path directly. At
        normal scale (~10% of frame) the math is no-op-clamp territory for
        all dtypes, so they should match fp64 truth within their respective
        precision budgets.
        """
        truth = generalized_box_iou(
            self._xyxy(0.4, 0.4, 0.1, 0.1, torch.float64),
            self._xyxy(0.6, 0.6, 0.1, 0.1, torch.float64),
        ).item()
        for dtype, places in self._DTYPES_PLACES:
            with self.subTest(dtype=str(dtype)):
                a = self._xyxy(0.4, 0.4, 0.1, 0.1, dtype)
                b = self._xyxy(0.6, 0.6, 0.1, 0.1, dtype)
                g = generalized_box_iou(a, b).item()
                self.assertAlmostEqual(g, truth, places=places)

    def test_distinct_pair_giou_small_boxes_match_fp64_on_same_coords(self) -> None:
        """Small-box (side=1e-3) distinct-pair: low-precision input through
        the fp32 promotion path should match fp64 result on the same
        already-quantized coords to within fp32 arithmetic noise.

        This is the test that was impossible before #91. Under the old
        ``finfo(fp16).tiny`` clamp, the ``enclosing`` denominator in
        ``generalized_box_iou`` ran in fp16, where the 6.10e-5 floor
        clobbered the true ~3.7e-5 enclosing area and distorted GIoU 40%
        off truth (-0.570 vs -0.944). After fp32 promotion the math runs
        in fp32 where the floor (1.18e-38) is a true no-op; the only
        remaining error is the caller's choice of fp16/bf16 input
        quantization, which we factor out by comparing against fp64
        computed FROM THE SAME quantized coords.

        Note on chosen coords: centres 0.495/0.500 with side 1e-3 stay
        non-degenerate in both fp16 (~5e-4 resolution at 0.5) and bf16
        (~4e-3 resolution at 0.5) — verified empirically. The assertion
        in generalized_box_iou never fires here.
        """
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                a = self._xyxy(0.495, 0.495, 1e-3, 1e-3, dtype)
                b = self._xyxy(0.500, 0.500, 1e-3, 1e-3, dtype)
                # Same already-quantized coords, cast up to fp64 — gives the
                # representational ceiling for those input values.
                a64, b64 = a.double(), b.double()
                g_truth = generalized_box_iou(a64, b64).item()
                g_low = generalized_box_iou(a, b).item()
                # fp32 arithmetic noise floor for IoU/GIoU is ~1e-6. The
                # important assertion is "match the same-coords fp64 result,"
                # not "match the original-truth fp64 result" — the latter
                # would force us to debug fp16 input quantization, which is
                # the caller's problem.
                self.assertAlmostEqual(
                    g_low,
                    g_truth,
                    places=5,
                    msg=(
                        f"{dtype}: GIoU = {g_low:+.6f}, fp64-from-same-coords = "
                        f"{g_truth:+.6f}. Pre-#91 this gap was ~40% on fp16 because the "
                        f"enclosing clamp clobbered valid sub-tiny areas."
                    ),
                )

    def test_output_dtype_contract(self) -> None:
        """fp16/bf16/fp32 → fp32 output, fp64 → fp64 output (preserved).

        This locks in the #91 dtype contract: low-precision inputs are
        promoted to at least fp32 internally; fp64 in either argument is
        preserved end-to-end. Mixed-dtype calls promote to the wider of
        ``{fp32-lifted-low-precision, other_input_dtype}`` — so
        ``box_iou(fp16, fp64) → fp64`` rather than silently downcasting
        the fp64 box. Anyone changing ``_LOW_PRECISION`` or the promotion
        logic in box_ops.py should think twice.
        """
        # All 16 combinations of {fp16, bf16, fp32, fp64} × {fp16, bf16, fp32, fp64}.
        # The matrix locks in the full contract: anyone refactoring _promote_target
        # (or replacing it with `.float()`) will fail loudly on at least one row.
        cases = [
            # Homogeneous-dtype callers.
            (torch.float16, torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16, torch.float32),
            (torch.float32, torch.float32, torch.float32),
            (torch.float64, torch.float64, torch.float64),
            # Two low-precision dtypes mixed — both lift to fp32 via _promote_target.
            (torch.float16, torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.float16, torch.float32),
            # Low-precision + fp32 — our promotion picks fp32 as the common target.
            (torch.float16, torch.float32, torch.float32),
            (torch.float32, torch.float16, torch.float32),
            (torch.bfloat16, torch.float32, torch.float32),
            (torch.float32, torch.bfloat16, torch.float32),
            # Low-precision + fp64 — our promotion picks fp64 (preserves fp64).
            (torch.float16, torch.float64, torch.float64),
            (torch.float64, torch.float16, torch.float64),
            (torch.bfloat16, torch.float64, torch.float64),
            (torch.float64, torch.bfloat16, torch.float64),
            # fp32 + fp64 — our promotion check is False (neither is low-precision),
            # so PyTorch native promotion handles it. Locks in that we don't
            # accidentally downcast fp64 here.
            (torch.float32, torch.float64, torch.float64),
            (torch.float64, torch.float32, torch.float64),
        ]
        for d1, d2, expected in cases:
            with self.subTest(boxes1=str(d1), boxes2=str(d2)):
                a = self._xyxy(0.4, 0.4, 0.1, 0.1, d1)
                b = self._xyxy(0.6, 0.6, 0.1, 0.1, d2)
                iou, _ = box_iou(a, b)
                giou = generalized_box_iou(a, b)
                self.assertEqual(
                    iou.dtype, expected, msg=f"({d1}, {d2}): IoU {iou.dtype}, expected {expected}"
                )
                self.assertEqual(
                    giou.dtype, expected, msg=f"({d1}, {d2}): GIoU {giou.dtype}, expected {expected}"
                )

    def test_zero_area_point_box_no_nan_inf_across_dtypes(self) -> None:
        """Degenerate point box (area=0) → no NaN/Inf in IoU or GIoU in any
        input precision. This is the primary win of the dtype-aware clamp:
        ``1e-12`` underflows to 0 in fp16/bf16 and the resulting ``0/0``
        produces NaN on the all-low-precision path; the new clamp prevents it."""
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                point = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=dtype)
                iou, _ = box_iou(point, point)
                self.assertFalse(torch.isnan(iou).any().item(), msg=f"{dtype}: IoU NaN")
                self.assertFalse(torch.isinf(iou).any().item(), msg=f"{dtype}: IoU Inf")
                giou = generalized_box_iou(point, point)
                self.assertFalse(torch.isnan(giou).any().item(), msg=f"{dtype}: GIoU NaN")
                self.assertFalse(torch.isinf(giou).any().item(), msg=f"{dtype}: GIoU Inf")

    def test_torchvision_box_area_upcasts_low_precision(self) -> None:
        """Canary: ``torchvision.ops.box_area`` upcasts fp16 AND bf16 → fp32.

        Post-#91 this is mostly defense in depth — our own ``_LOW_PRECISION``
        promotion at the top of box_iou/generalized_box_iou handles the
        common cases. But it's still load-bearing for one path: when only ONE
        side of the call is low-precision and the OTHER side is fp32
        (e.g., the caller hand-mixes ``fp16_pred`` with ``fp32_target``).
        Our promotion picks fp32 as the common target, and ``box_area`` of
        the fp16 side relies on torchvision's upcast to land in fp32. If
        torchvision ever drops this upcast, that case regresses to fp16
        area, re-exposing the small-box GIoU distortion #91 fixed.
        """
        from torchvision.ops.boxes import box_area

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                b = torch.tensor([[0.45, 0.45, 0.55, 0.55]], dtype=dtype)
                self.assertEqual(
                    box_area(b).dtype,
                    torch.float32,
                    msg=(
                        f"torchvision.ops.box_area no longer upcasts {dtype} → fp32. "
                        f"Audit ml_engine/utils/box_ops.py and confirm the dtype-aware "
                        f"clamp still floors the denominator correctly under {dtype}."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
