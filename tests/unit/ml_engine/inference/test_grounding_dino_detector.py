"""
Adversarial unit tests for GroundingDINODetector and logits_to_class_scores.

Probes:
- text_threshold is actually applied (not silently dropped — the original bug)
- strict > boundary semantics (score == threshold gets zeroed)
- class-winner flip: changing threshold changes which class is assigned
- confidence magnitudes are correct post-mask (not further scaled)
- input tensor is not mutated by logits_to_class_scores
- multi-token dilution: partial token masking reduces mean predictably
- NMS still fires correctly after text_threshold pre-filtering
- zero-threshold is a true no-op (bit-identical output vs no-arg call)
- device of mask matches logits device (no cross-device op)
- detect() wiring: text_threshold reaches logits_to_class_scores, not discarded
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ml_engine.inference.detectors.grounding_dino import (
    GroundingDINODetector,
    logits_to_class_scores,
)

_MODULE = "ml_engine.inference.detectors.grounding_dino"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detector(
    raw_logits: torch.Tensor,
    boxes: torch.Tensor,
) -> GroundingDINODetector:
    """Detector with real __init__ but model replaced by a mock."""
    d = GroundingDINODetector("dummy_cfg", "dummy_ckpt", device="cpu")
    mock_model = MagicMock()
    mock_model.return_value = {
        "pred_logits": raw_logits.unsqueeze(0),
        "pred_boxes": boxes.unsqueeze(0),
    }
    mock_model.tokenizer = MagicMock()
    d._model = mock_model
    return d


def _unit_boxes(nq: int) -> torch.Tensor:
    """nq non-overlapping boxes so NMS never reduces the count."""
    boxes = torch.zeros(nq, 4)
    for i in range(nq):
        boxes[i] = torch.tensor([0.1 * i, 0.5, 0.05, 0.05])
    return boxes


def _overlapping_boxes(nq: int) -> torch.Tensor:
    """nq identical boxes — NMS will reduce to 1."""
    boxes = torch.zeros(nq, 4)
    boxes[:] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    return boxes


def _run_detect(
    detector: GroundingDINODetector,
    prompts: List[str],
    positive_map: Dict[int, List[int]],
    *,
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float = 0.7,
):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with (
        patch(f"{_MODULE}.preprocess_caption", return_value="mocked."),
        patch(f"{_MODULE}.preprocess_image", return_value=torch.zeros(3, 32, 32)),
        patch(f"{_MODULE}.build_positive_map", return_value=positive_map),
    ):
        return detector.detect(
            image,
            prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
        )


def _raw(sigmoid_score: float) -> float:
    """Return the raw logit whose sigmoid is exactly sigmoid_score."""
    return torch.logit(torch.tensor(sigmoid_score)).item()


# ---------------------------------------------------------------------------
# logits_to_class_scores — adversarial pure-tensor tests
# ---------------------------------------------------------------------------


class TestLogitsToClassScoresAdversarial:
    # --- the original bug: was text_threshold silently ignored? ---

    def test_param_is_not_silently_ignored(self):
        """Changing text_threshold must change the output — proves the param is wired."""
        positive_map = {0: [0]}
        logits = torch.tensor([[0.4]])
        s_default = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.0)
        s_filtered = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert s_default[0, 0].item() == pytest.approx(0.4)
        assert s_filtered[0, 0].item() == pytest.approx(0.0), (
            "text_threshold 0.5 must zero a token score of 0.4 — "
            "if both are 0.4 the parameter is still silently dropped"
        )

    # --- strict > boundary ---

    def test_score_exactly_at_threshold_is_zeroed(self):
        # 0.5 > 0.5 is False → must be zeroed (strict greater-than)
        positive_map = {0: [0]}
        logits = torch.tensor([[0.5]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.0), (
            "score == threshold should be zeroed (strict >), not kept"
        )

    def test_score_just_above_threshold_is_kept(self):
        positive_map = {0: [0]}
        logits = torch.tensor([[0.5001]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.5001, abs=1e-4)

    def test_score_just_below_threshold_is_zeroed(self):
        positive_map = {0: [0]}
        logits = torch.tensor([[0.4999]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.0)

    # --- input tensor must not be mutated ---

    def test_input_logits_not_mutated(self):
        positive_map = {0: [0, 1]}
        original = torch.tensor([[0.8, 0.3]])
        clone = original.clone()
        logits_to_class_scores(original, positive_map, 1, text_threshold=0.5)
        assert torch.allclose(original, clone), "logits_to_class_scores must not modify its input"

    # --- class-winner flip ---

    def test_masking_changes_winning_class(self):
        # Class 0 token at position 0: score 0.7 (passes 0.5)
        # Class 1 token at position 1: score 0.4 (fails 0.5)
        # Without threshold: class 0 wins (0.7 vs 0.4) — same either way
        # Class 0 token at position 0: score 0.4 (fails 0.5)
        # Class 1 token at position 1: score 0.7 (passes 0.5)
        # → winner should be class 1 at threshold 0.5
        positive_map = {0: [0], 1: [1]}
        # query where class 0 = 0.4 (weak), class 1 = 0.7 (strong)
        logits = torch.tensor([[0.4, 0.7]])

        scores_no_thresh = logits_to_class_scores(logits, positive_map, 2, text_threshold=0.0)
        scores_thresh = logits_to_class_scores(logits, positive_map, 2, text_threshold=0.5)

        # Both have class 1 winning already; but now class 0 must be zero
        assert scores_thresh[0, 0].item() == pytest.approx(0.0)
        assert scores_thresh[0, 1].item() == pytest.approx(0.7)
        # Without threshold, class 0 still contributes
        assert scores_no_thresh[0, 0].item() == pytest.approx(0.4)

    def test_masking_zeroes_losing_class_token(self):
        # Class 0=0.8 (strong), class 1=0.6 (weak); threshold=0.7 zeroes class 1's token
        # (0.6 ≤ 0.7). Class 0 still wins, but class 1 drops from 0.6 to 0.0.
        positive_map = {0: [0], 1: [1]}
        logits = torch.tensor([[0.8, 0.6]])
        scores_no = logits_to_class_scores(logits, positive_map, 2, text_threshold=0.0)
        scores_th = logits_to_class_scores(logits, positive_map, 2, text_threshold=0.7)

        assert scores_no[0].argmax().item() == 0
        assert scores_th[0, 0].item() == pytest.approx(0.8)
        assert scores_th[0, 1].item() == pytest.approx(0.0)

    # --- confidence magnitude preservation ---

    def test_passing_token_score_is_not_rescaled(self):
        # A token that passes threshold must come through unchanged, not scaled
        positive_map = {0: [0]}
        for score in [0.51, 0.75, 0.99]:
            logits = torch.tensor([[score]])
            s = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
            assert s[0, 0].item() == pytest.approx(score, abs=1e-5), (
                f"passing token score {score} must not be rescaled"
            )

    # --- multi-token partial masking (dilution) ---

    def test_multi_token_dilution_is_exact(self):
        # Two tokens for class 0: scores [0.8, 0.3]; threshold=0.5
        # token 0 (0.8) passes → contributes 0.8
        # token 1 (0.3) fails → contributes 0.0
        # mean([0.8, 0.0]) = 0.4  (divided by 2, not just by passing count)
        positive_map = {0: [0, 1]}
        logits = torch.tensor([[0.8, 0.3]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.4, abs=1e-5)

    def test_three_token_two_passing_dilution(self):
        # Tokens [0.9, 0.7, 0.2]; threshold=0.5 → [0.9, 0.7, 0.0]; mean = 0.5333...
        positive_map = {0: [0, 1, 2]}
        logits = torch.tensor([[0.9, 0.7, 0.2]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        expected = (0.9 + 0.7 + 0.0) / 3
        assert scores[0, 0].item() == pytest.approx(expected, abs=1e-5)

    def test_all_tokens_failing_gives_zero_class_score(self):
        positive_map = {0: [0, 1, 2]}
        logits = torch.tensor([[0.3, 0.4, 0.2]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.0)

    # --- zero threshold is a true no-op ---

    def test_zero_threshold_bit_identical_to_no_arg(self):
        positive_map = {0: [0, 1], 1: [2, 3]}
        logits = torch.rand(6, 8, generator=torch.Generator().manual_seed(42))
        s_none = logits_to_class_scores(logits, positive_map, 2)
        s_zero = logits_to_class_scores(logits, positive_map, 2, text_threshold=0.0)
        assert torch.equal(s_none, s_zero), "text_threshold=0.0 must be bit-identical to default"

    # --- threshold=1.0 drops everything (sigmoid < 1.0 always) ---

    def test_threshold_one_always_zeros_all(self):
        positive_map = {0: [0], 1: [1]}
        # Use values very close to 1.0 — they still can't reach 1.0 via sigmoid
        logits = torch.tensor([[0.9999, 0.9998], [0.5, 0.9]])
        scores = logits_to_class_scores(logits, positive_map, 2, text_threshold=1.0)
        assert scores.abs().max().item() == pytest.approx(0.0)

    # --- device consistency ---

    def test_mask_stays_on_cpu(self):
        positive_map = {0: [0]}
        logits = torch.tensor([[0.8]])  # CPU
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores.device.type == "cpu"

    # --- output shape is always correct ---

    def test_output_shape_various_sizes(self):
        for nq, max_len, nc in [(1, 5, 1), (10, 20, 3), (100, 256, 80)]:
            pm = {c: [c] for c in range(nc)}
            logits = torch.rand(nq, max_len)
            scores = logits_to_class_scores(logits, pm, nc, text_threshold=0.5)
            assert scores.shape == (nq, nc)

    # --- unordered token indices ---

    def test_unsorted_tok_indices_work_correctly(self):
        # Class 0's tokens at positions [4, 1, 7] (deliberately out of order)
        positive_map = {0: [4, 1, 7]}
        logits = torch.zeros(2, 10)
        logits[0, 4] = 0.9
        logits[0, 1] = 0.8
        logits[0, 7] = 0.3  # below threshold
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        expected = (0.9 + 0.8 + 0.0) / 3
        assert scores[0, 0].item() == pytest.approx(expected, abs=1e-5)

    # --- multiple queries, class scores independent per query ---

    def test_each_query_masked_independently(self):
        # 4 queries, 1 class, token at position 0
        # Scores: 0.9, 0.6, 0.5, 0.3 with threshold=0.5
        # Expected after mask: 0.9, 0.6, 0.0 (exactly 0.5), 0.0
        positive_map = {0: [0]}
        logits = torch.tensor([[0.9], [0.6], [0.5], [0.3]])
        scores = logits_to_class_scores(logits, positive_map, 1, text_threshold=0.5)
        assert scores[0, 0].item() == pytest.approx(0.9)
        assert scores[1, 0].item() == pytest.approx(0.6)
        assert scores[2, 0].item() == pytest.approx(0.0)  # exactly at boundary
        assert scores[3, 0].item() == pytest.approx(0.0)

    # --- classes not in positive_map stay zero ---

    def test_missing_class_in_map_stays_zero(self):
        # 3 classes but only class 1 is in positive_map
        positive_map = {1: [0]}
        logits = torch.tensor([[0.9]])
        scores = logits_to_class_scores(logits, positive_map, 3, text_threshold=0.0)
        assert scores[0, 0].item() == pytest.approx(0.0)
        assert scores[0, 1].item() == pytest.approx(0.9)
        assert scores[0, 2].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GroundingDINODetector.detect — adversarial wiring tests
# ---------------------------------------------------------------------------


class TestDetectTextThresholdAdversarial:
    """Verify text_threshold is correctly wired into detect() and not discarded."""

    def _build(self, sigmoid_scores_per_query: List[float], tok_pos: int = 2):
        """
        One class, one token at tok_pos. Each entry in sigmoid_scores_per_query
        is the intended sigmoided value for that query.
        """
        nq = len(sigmoid_scores_per_query)
        raw = torch.zeros(nq, 8)
        for q, s in enumerate(sigmoid_scores_per_query):
            raw[q, tok_pos] = _raw(s)
        boxes = _unit_boxes(nq)
        return _detector(raw, boxes), {0: [tok_pos]}

    # --- old bug reproduced: param silently dropped → both calls identical ---

    def test_text_threshold_changes_detection_count(self):
        """If the param is silently dropped both calls return same count — catch that."""
        detector, pm = self._build([0.9, 0.3])
        r_no = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.0)
        r_th = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(r_no) == 2, "without threshold both queries should pass box_threshold=0.1"
        assert len(r_th) == 1, "with threshold=0.5, query with score=0.3 must be dropped"

    # --- boundary: score exactly at threshold ---

    def test_score_exactly_at_threshold_dropped(self):
        detector, pm = self._build([0.5])
        # With threshold=0.5: token score 0.5 is NOT > 0.5 → zeroed → below box_threshold
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(result) == 0, "score == threshold must be zeroed (strict >)"

    def test_score_epsilon_above_threshold_kept(self):
        detector, pm = self._build([0.501])
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(result) == 1

    # --- threshold=1.0 always drops everything ---

    def test_threshold_one_drops_all_detections(self):
        detector, pm = self._build([0.99, 0.9, 0.7])
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.0, text_threshold=1.0)
        assert len(result) == 0, "threshold=1.0 must zero all tokens (sigmoid < 1.0 always)"

    # --- threshold=0.0 is a true no-op relative to box_threshold alone ---

    def test_threshold_zero_is_no_op(self):
        # text_threshold=0.0 preserves all tokens — scores [0.9, 0.6, 0.1] all pass
        # box_threshold=0.05. text_threshold=0.5 zeroes the 0.1 query → only 2 survive.
        detector, pm = self._build([0.9, 0.6, 0.1])
        r_zero = _run_detect(detector, ["cat"], pm, box_threshold=0.05, text_threshold=0.0)
        r_active = _run_detect(detector, ["cat"], pm, box_threshold=0.05, text_threshold=0.5)
        assert len(r_zero) == 3, "text_threshold=0.0 must not filter any detections"
        assert len(r_active) == 2, "text_threshold=0.5 must zero the 0.1-score query"

    # --- confidence values after masking are not rescaled ---

    def test_confidence_magnitude_is_post_mask_score(self):
        # Query with sigmoid score 0.8; threshold=0.5 → token kept at 0.8
        # Expected confidence ≈ 0.8 (the actual sigmoid of the stored raw logit)
        detector, pm = self._build([0.8])
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(result) == 1
        assert result.confidences[0] == pytest.approx(0.8, abs=1e-4)

    def test_zeroed_query_confidence_not_in_result(self):
        # Two queries: scores 0.9 and 0.3; threshold=0.5 drops 0.3
        # Result should contain only confidence ≈ 0.9
        detector, pm = self._build([0.9, 0.3])
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(result) == 1
        assert result.confidences[0] == pytest.approx(0.9, abs=1e-4)

    # --- class ids are correct after masking ---

    def test_class_ids_correct_when_one_class_zeroed(self):
        # Two classes; class 0 token weak (0.3), class 1 token strong (0.8)
        # threshold=0.5 zeroes class 0 → detection is class 1
        nq = 1
        raw = torch.zeros(nq, 8)
        raw[0, 1] = _raw(0.3)  # class 0 token
        raw[0, 3] = _raw(0.8)  # class 1 token
        boxes = _unit_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [1], 1: [3]}

        result = _run_detect(detector, ["cat", "dog"], pm, box_threshold=0.1, text_threshold=0.5)
        assert len(result) == 1
        assert result.class_ids[0] == 1, "class 0's token was zeroed; class 1 should win"

    def test_class_assignment_without_threshold_is_original_winner(self):
        # Same setup: class 0=0.3, class 1=0.8 → class 1 wins even without threshold
        nq = 1
        raw = torch.zeros(nq, 8)
        raw[0, 1] = _raw(0.3)
        raw[0, 3] = _raw(0.8)
        boxes = _unit_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [1], 1: [3]}
        result = _run_detect(detector, ["cat", "dog"], pm, box_threshold=0.1, text_threshold=0.0)
        assert len(result) == 1
        assert result.class_ids[0] == 1  # class 1 still wins by score

    # --- NMS still fires after text_threshold ---

    def test_nms_still_removes_duplicate_overlapping_boxes(self):
        # 3 queries all at the same box location; all pass text_threshold
        # NMS with iou=0.5 should keep only 1
        nq = 3
        raw = torch.zeros(nq, 8)
        for q in range(nq):
            raw[q, 2] = _raw(0.9 - q * 0.05)  # 0.9, 0.85, 0.80 — all pass 0.5
        boxes = _overlapping_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [2]}
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5, nms_threshold=0.5)
        assert len(result) == 1, "NMS must collapse overlapping detections"

    def test_nms_sees_post_threshold_scores(self):
        # 2 overlapping boxes; box 0 has score 0.3 (zeroed by text_threshold),
        # box 1 has score 0.9 (passes). NMS keeps 1 box with confidence ≈ 0.9.
        nq = 2
        raw = torch.zeros(nq, 8)
        raw[0, 2] = _raw(0.3)
        raw[1, 2] = _raw(0.9)
        boxes = _overlapping_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [2]}
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.1, text_threshold=0.5, nms_threshold=0.5)
        # Box 0 was zeroed; only box 1 survives
        assert len(result) == 1
        assert result.confidences[0] == pytest.approx(0.9, abs=1e-4)

    # --- multi-token dilution reaches detect output ---

    def test_multi_token_dilution_can_push_below_box_threshold(self):
        # Class 0 has 2 tokens: scores 0.9 and 0.3; threshold=0.5
        # After mask: [0.9, 0.0]; mean = 0.45
        # box_threshold=0.46 → 0.45 < 0.46 → detection dropped
        nq = 1
        raw = torch.zeros(nq, 8)
        raw[0, 1] = _raw(0.9)
        raw[0, 2] = _raw(0.3)
        boxes = _unit_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [1, 2]}
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.46, text_threshold=0.5)
        assert len(result) == 0, "multi-token dilution must reduce class score below box_threshold"

    def test_multi_token_dilution_kept_when_box_threshold_lower(self):
        # Same as above but box_threshold=0.44 → diluted score 0.45 > 0.44 → kept
        nq = 1
        raw = torch.zeros(nq, 8)
        raw[0, 1] = _raw(0.9)
        raw[0, 2] = _raw(0.3)
        boxes = _unit_boxes(nq)
        detector = _detector(raw, boxes)
        pm = {0: [1, 2]}
        result = _run_detect(detector, ["cat"], pm, box_threshold=0.44, text_threshold=0.5)
        assert len(result) == 1

    # --- sweeping the threshold reveals monotone behaviour ---

    def test_detection_count_monotone_decreasing_with_threshold(self):
        # 5 queries with scores 0.2, 0.4, 0.6, 0.8, 0.95
        # As threshold increases, count can only stay the same or decrease
        scores = [0.2, 0.4, 0.6, 0.8, 0.95]
        detector, pm = self._build(scores)
        thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        counts = []
        for th in thresholds:
            r = _run_detect(detector, ["cat"], pm, box_threshold=0.01, text_threshold=th)
            counts.append(len(r))
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"detection count must not increase as threshold rises: "
                f"threshold {thresholds[i]}→{thresholds[i + 1]} gave {counts[i]}→{counts[i + 1]}"
            )

    def test_empty_positive_map_returns_empty_result(self):
        # When build_positive_map returns {} (tokenizer can't map any class),
        # detect() must return an empty DetectionResult without raising.
        detector, _ = self._build([0.9])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch(f"{_MODULE}.preprocess_caption", return_value="mocked."),
            patch(f"{_MODULE}.preprocess_image", return_value=torch.zeros(3, 32, 32)),
            patch(f"{_MODULE}.build_positive_map", return_value={}),
        ):
            result = detector.detect(
                image,
                ["cat"],
                box_threshold=0.5,
                text_threshold=0.5,
                nms_threshold=0.5,
            )
        assert len(result) == 0
        assert result.boxes_xyxy.shape == (0, 4)
        assert result.confidences.shape == (0,)
        assert result.class_ids.shape == (0,)
