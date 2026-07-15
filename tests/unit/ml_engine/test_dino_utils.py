"""Unit tests for ml_engine.training.dino_utils.

Two functions:
- build_detr_targets(...): PURE tensor logic (no model/tokenizer). Builds the
  DETR-format targets fed into the GroundingDINO Hungarian-matching loss. A bug
  here silently corrupts training targets, so every branch is exercised.
- build_positive_map(...): orchestrates the BERT tokenizer + GroundingDINO span
  utilities. Tested with those two utilities and the tokenizer mocked, so no real
  BERT weights are needed — the logic under test is OUR class->token-span mapping
  and its error path, not the third-party utilities.

All tests run on CPU with tiny synthetic tensors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from ml_engine.training import dino_utils
from ml_engine.training.dino_utils import build_detr_targets, build_positive_map

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# build_detr_targets
# ---------------------------------------------------------------------------


def _positive_map(num_classes: int = 3, max_text_len: int = 8) -> torch.Tensor:
    """A positive map whose row i is all-(i+1), so token_labels[obj] == row picked
    makes it trivial to assert which class index was selected.

    Synthetic non-normalized rows on purpose: the real create_positive_map_from_span
    returns rows normalized to sum ~1, but these tests only verify row ASSIGNMENT
    (which row went where), not the numerical normalization."""
    pm = torch.zeros(num_classes, max_text_len, dtype=torch.float32)
    for i in range(num_classes):
        pm[i] = float(i + 1)  # row 0 -> 1.0, row 1 -> 2.0, ... (distinct, non-zero)
    return pm


class TestBuildDetrTargets:
    def test_returns_one_target_dict_per_batch_element(self):
        boxes = torch.rand(2, 3, 4)
        labels = torch.tensor([[10, 20, -1], [30, -1, -1]])
        cat2idx = {10: 0, 20: 1, 30: 2}
        targets = build_detr_targets(boxes, labels, _positive_map(), cat2idx, CPU)
        assert isinstance(targets, list)
        assert len(targets) == 2
        for t in targets:
            assert set(t.keys()) == {"labels", "boxes", "token_labels"}

    def test_padding_labels_are_filtered(self):
        boxes = torch.rand(1, 4, 4)
        labels = torch.tensor([[10, -1, 20, -1]])  # 2 valid, 2 padding
        cat2idx = {10: 0, 20: 1}
        targets = build_detr_targets(boxes, labels, _positive_map(), cat2idx, CPU)
        # NOTE: "labels" holds the original category IDs (10, 20), NOT 0-based class
        # indices. The loss only consumes len(labels) for object counting; class
        # info flows through token_labels. Asserting IDs documents this on purpose.
        assert targets[0]["labels"].tolist() == [10, 20]
        assert targets[0]["boxes"].shape == (2, 4)

    def test_valid_boxes_come_from_the_unpadded_rows(self):
        boxes = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4) / 100.0
        labels = torch.tensor([[10, -1, 20]])  # keep rows 0 and 2
        cat2idx = {10: 0, 20: 1}
        targets = build_detr_targets(boxes, labels, _positive_map(), cat2idx, CPU)
        assert torch.equal(targets[0]["boxes"], boxes[0][[0, 2]])

    def test_token_labels_select_the_mapped_positive_map_row(self):
        # category_id != index on purpose: cat 10->idx0, cat 20->idx1, cat 30->idx2
        pm = _positive_map(num_classes=3, max_text_len=5)
        boxes = torch.rand(1, 3, 4)
        labels = torch.tensor([[30, 10, 20]])
        cat2idx = {10: 0, 20: 1, 30: 2}
        targets = build_detr_targets(boxes, labels, pm, cat2idx, CPU)
        tl = targets[0]["token_labels"]
        assert torch.equal(tl[0], pm[2])  # cat 30 -> idx 2
        assert torch.equal(tl[1], pm[0])  # cat 10 -> idx 0
        assert torch.equal(tl[2], pm[1])  # cat 20 -> idx 1

    def test_token_labels_shape_and_dtype(self):
        pm = _positive_map(num_classes=2, max_text_len=7)
        boxes = torch.rand(1, 2, 4)
        labels = torch.tensor([[10, 20]])
        targets = build_detr_targets(boxes, labels, pm, {10: 0, 20: 1}, CPU)
        tl = targets[0]["token_labels"]
        assert tl.shape == (2, 7)
        assert tl.dtype == torch.float32

    def test_all_padding_batch_element_yields_empty_targets(self):
        pm = _positive_map(num_classes=2, max_text_len=6)
        boxes = torch.rand(1, 3, 4)
        labels = torch.tensor([[-1, -1, -1]])
        targets = build_detr_targets(boxes, labels, pm, {10: 0}, CPU)
        assert targets[0]["labels"].numel() == 0
        assert targets[0]["boxes"].shape == (0, 4)
        assert targets[0]["token_labels"].shape == (0, 6)

    def test_mixed_padding_across_batch_elements(self):
        # One call where b=0 is all-padding and b=1 has valid objects — exercises
        # both the empty and non-empty branches of the per-element loop together.
        pm = _positive_map(num_classes=2, max_text_len=5)
        boxes = torch.rand(2, 3, 4)
        labels = torch.tensor([[-1, -1, -1], [10, 20, -1]])
        targets = build_detr_targets(boxes, labels, pm, {10: 0, 20: 1}, CPU)
        assert targets[0]["labels"].numel() == 0
        assert targets[0]["token_labels"].shape == (0, 5)
        assert targets[1]["labels"].tolist() == [10, 20]
        assert torch.equal(targets[1]["token_labels"][0], pm[0])
        assert torch.equal(targets[1]["token_labels"][1], pm[1])

    def test_empty_batch_returns_empty_list(self):
        boxes = torch.empty(0, 3, 4)
        labels = torch.empty(0, 3, dtype=torch.long)
        assert build_detr_targets(boxes, labels, _positive_map(), {10: 0}, CPU) == []

    def test_unknown_category_id_raises_value_error(self):
        boxes = torch.rand(1, 1, 4)
        labels = torch.tensor([[99]])  # 99 not in the mapping
        with pytest.raises(ValueError, match="Unknown category_id 99"):
            build_detr_targets(boxes, labels, _positive_map(), {10: 0}, CPU)

    def test_unnormalized_boxes_log_a_warning_but_still_build(self, caplog):
        pm = _positive_map(num_classes=1, max_text_len=4)
        boxes = torch.tensor([[[0.1, 0.1, 0.2, 5.0]]])  # 5.0 > 1 -> not normalized
        labels = torch.tensor([[10]])
        with caplog.at_level("WARNING"):
            targets = build_detr_targets(boxes, labels, pm, {10: 0}, CPU)
        assert any("not normalized" in r.message for r in caplog.records)
        # still produces a valid target despite the warning
        assert targets[0]["labels"].tolist() == [10]

    def test_normalized_boxes_do_not_warn(self, caplog):
        pm = _positive_map(num_classes=1, max_text_len=4)
        boxes = torch.tensor([[[0.1, 0.1, 0.2, 0.3]]])  # all in [0, 1]
        labels = torch.tensor([[10]])
        with caplog.at_level("WARNING"):
            build_detr_targets(boxes, labels, pm, {10: 0}, CPU)
        assert not any("not normalized" in r.message for r in caplog.records)

    def test_token_labels_placed_on_requested_device(self):
        pm = _positive_map(num_classes=1, max_text_len=4)
        boxes = torch.rand(1, 1, 4)
        labels = torch.tensor([[10]])
        targets = build_detr_targets(boxes, labels, pm, {10: 0}, CPU)
        assert targets[0]["token_labels"].device == CPU


# ---------------------------------------------------------------------------
# build_positive_map (BERT tokenizer + GroundingDINO span utils mocked)
# ---------------------------------------------------------------------------


def _fake_tokenizer():
    """A tokenizer whose call returns an object exposing .to(device) -> itself."""
    encoding = MagicMock(name="encoding")
    encoding.to.return_value = encoding
    tok = MagicMock(name="tokenizer", return_value=encoding)
    return tok, encoding


class TestBuildPositiveMap:
    def test_passes_encoding_ordered_spans_and_returns_result(self):
        tok, encoding = _fake_tokenizer()
        class_names = ["cat", "dog"]
        expected = torch.ones(2, 8)
        with (
            patch.object(
                dino_utils,
                "build_captions_and_token_span",
                return_value=("cat . dog", {"cat": [(0, 3)], "dog": [(6, 9)]}),
            ),
            patch.object(dino_utils, "create_positive_map_from_span", return_value=expected) as mock_span,
        ):
            out = build_positive_map(tok, class_names, max_text_len=8, device=CPU)
        # NOT a tautology: assert the exact downstream call, not just that the
        # mocked return flows back. Catches arg-drift (wrong encoding / spans / len).
        mock_span.assert_called_once_with(encoding, [[(0, 3)], [(6, 9)]], max_text_len=8)
        assert out is expected  # result returned (via .to(device), a no-op on the CPU tensor)

    def test_token_spans_passed_in_class_order(self):
        """token_span_per_class must be built in class-index order (0..N-1),
        regardless of dict ordering in cat2tokenspan."""
        tok, encoding = _fake_tokenizer()
        class_names = ["cat", "dog", "bird"]
        cat2span = {"bird": [(2,)], "cat": [(0,)], "dog": [(1,)]}  # scrambled dict order
        with (
            patch.object(dino_utils, "build_captions_and_token_span", return_value=("cap", cat2span)),
            patch.object(
                dino_utils, "create_positive_map_from_span", return_value=torch.zeros(3, 4)
            ) as mock_span,
        ):
            build_positive_map(tok, class_names, max_text_len=4, device=CPU)
        call = mock_span.call_args
        assert call.args[0] is encoding  # the tokenized encoding is forwarded
        # spans built in CLASS-INDEX order (0..N-1), not dict order
        assert call.args[1] == [cat2span["cat"], cat2span["dog"], cat2span["bird"]]
        assert call.kwargs["max_text_len"] == 4

    def test_missing_class_in_span_raises_value_error(self):
        tok, _ = _fake_tokenizer()
        class_names = ["cat", "ghost"]
        with (
            patch.object(
                dino_utils,
                "build_captions_and_token_span",
                return_value=("cat . ghost", {"cat": [(0, 3)]}),  # "ghost" missing
            ),
            patch.object(dino_utils, "create_positive_map_from_span", return_value=torch.zeros(2, 4)),
        ):
            with pytest.raises(ValueError, match="Class 'ghost' not found in cat2tokenspan"):
                build_positive_map(tok, class_names, max_text_len=4, device=CPU)

    def test_tokenizer_called_with_padding_and_pt(self):
        tok, encoding = _fake_tokenizer()
        with (
            patch.object(
                dino_utils, "build_captions_and_token_span", return_value=("cat", {"cat": [(0, 3)]})
            ),
            patch.object(dino_utils, "create_positive_map_from_span", return_value=torch.zeros(1, 4)),
        ):
            build_positive_map(tok, ["cat"], max_text_len=4, device=CPU)
        tok.assert_called_once_with("cat", padding="longest", return_tensors="pt")
        encoding.to.assert_called_once_with(CPU)
