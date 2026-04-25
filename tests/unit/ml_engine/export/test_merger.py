"""
Unit tests for ml_engine.export.merger — written adversarially.

Covers the three branches of `merge_lora_weights`, `save_merged_model`'s
metadata + class-name handling, and `load_merged_model`'s error paths.
Specifically tests the edges where the current implementation is sloppy
or surprising — those tests are marked `xfail` with a clear reason so
they act as tracked-but-pending design issues, not silent green stamps.

Stubs duck-type the PEFT shape (hasattr-based check) using lightweight
real classes — no PEFT, no GroundingDINO, no actual LoRA. Keeps the
suite fast (~1s) and isolated from external state.
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from ml_engine.export.merger import (
    load_merged_model,
    merge_lora_weights,
    save_merged_model,
)

# ---------------------------------------------------------------------------
# Helpers — tiny model stubs
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal nn.Module with a state_dict for save/load tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


class _TinyModelDifferentShape(nn.Module):
    """Same param names as _TinyModel but different shapes — for shape-mismatch tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 2)  # in_features=8 vs _TinyModel's 4


class _TinyModelExtraParam(nn.Module):
    """Has all of _TinyModel's params plus an extra one — for strict=False tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.extra = nn.Linear(2, 1)  # not in _TinyModel


class _PeftWrapperStub:
    """
    Mimics a PEFT model: exposes `merge_and_unload()` returning the merged module.
    Tracks whether merge was actually invoked so tests can assert on that.
    """

    def __init__(self, merged: nn.Module) -> None:
        self._merged = merged
        self.merge_called = False

    def merge_and_unload(self) -> nn.Module:
        self.merge_called = True
        return self._merged


class _GroundingDINOLoRAStub:
    """Mimics GroundingDINOLoRA's nested-PEFT shape (model.model.merge_and_unload)."""

    def __init__(self, merged: nn.Module) -> None:
        self.model = _PeftWrapperStub(merged)


class _BothLevelsHaveMergeStub:
    """
    Pathological shape: BOTH self.merge_and_unload AND self.model.merge_and_unload.
    The merger picks the nested one silently — undocumented precedence.
    """

    def __init__(self, nested_target: nn.Module, direct_target: nn.Module) -> None:
        self.model = _PeftWrapperStub(nested_target)
        self._direct_target = direct_target
        self.direct_merge_called = False

    def merge_and_unload(self) -> nn.Module:
        self.direct_merge_called = True
        return self._direct_target


# ===========================================================================
# merge_lora_weights — three branches + the precedence ambiguity
# ===========================================================================


class TestMergeLoraWeights:
    def test_nested_peft_wrapper_returns_merged(self):
        """GroundingDINOLoRA pattern: model.model.merge_and_unload() is called."""
        target = _TinyModel()
        wrapper = _GroundingDINOLoRAStub(target)

        result = merge_lora_weights(wrapper)

        assert result is target
        assert wrapper.model.merge_called is True

    def test_direct_peft_returns_merged(self):
        """When the model itself has merge_and_unload, call it directly."""
        target = _TinyModel()
        direct = _PeftWrapperStub(target)

        result = merge_lora_weights(direct)

        assert result is target
        assert direct.merge_called is True

    def test_plain_model_returned_unchanged_with_warning(self, caplog):
        """No PEFT shape → return as-is, log a warning, do NOT raise."""
        plain = _TinyModel()

        with caplog.at_level("WARNING"):
            result = merge_lora_weights(plain)

        assert result is plain
        assert any("does not have LoRA adapters" in rec.message for rec in caplog.records)

    def test_nested_takes_precedence_over_direct_silently(self):
        """
        When BOTH `self.model.merge_and_unload` AND `self.merge_and_unload` exist,
        the nested branch wins and the direct one is silently ignored. This
        precedence is undocumented and would be surprising if a real model class
        ever ended up with both shapes (e.g. via mixin composition).
        """
        nested_target = _TinyModel()
        direct_target = _TinyModel()
        ambiguous = _BothLevelsHaveMergeStub(nested_target, direct_target)

        result = merge_lora_weights(ambiguous)

        # Nested branch wins
        assert result is nested_target
        assert ambiguous.model.merge_called is True
        # Direct branch is bypassed — caller's expectation may or may not match
        assert ambiguous.direct_merge_called is False

    def test_nested_attribute_without_merge_falls_through_to_direct(self):
        """
        If `self.model` exists but `self.model.merge_and_unload` does NOT exist
        (e.g. self.model is None, or some wrapped torch module), and `self.merge_and_unload`
        does exist, the merger should fall through to the direct branch. This documents
        that the two hasattr-checks are independent.
        """

        class _NestedAttributeButNoMerge:
            """`model` attr exists but doesn't expose merge_and_unload."""

            def __init__(self, merged: nn.Module) -> None:
                self.model = nn.Linear(2, 2)  # plain layer, no merge_and_unload
                self._merged = merged
                self.merge_called = False

            def merge_and_unload(self) -> nn.Module:
                self.merge_called = True
                return self._merged

        target = _TinyModel()
        weird = _NestedAttributeButNoMerge(target)

        result = merge_lora_weights(weird)

        assert result is target
        assert weird.merge_called is True


# ===========================================================================
# save_merged_model — checkpoint structure + metadata edges
# ===========================================================================


class TestSaveMergedModel:
    def test_saves_state_dict_and_default_metadata(self, tmp_path: Path):
        model = _TinyModel()
        out = tmp_path / "model.pth"

        returned = save_merged_model(model, out)

        assert returned == out
        assert out.exists()

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert ckpt["class_names"] == []
        assert ckpt["metadata"]["format"] == "merged_grounding_dino"
        assert ckpt["metadata"]["peft_merged"] is True
        assert ckpt["metadata"]["requires_peft"] is False

    def test_class_names_round_trip(self, tmp_path: Path):
        model = _TinyModel()
        out = tmp_path / "model.pth"
        names = ["dog", "cat", "wombat"]

        save_merged_model(model, out, class_names=names)

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt["class_names"] == names

    def test_class_names_none_and_empty_list_both_become_empty(self, tmp_path: Path):
        """
        Surface a usability concern: `class_names=None` and `class_names=[]` produce
        the same on-disk representation (empty list). Caller cannot distinguish
        "no classes provided" from "explicitly empty class set" after a save/load.
        """
        out_none = tmp_path / "none.pth"
        out_empty = tmp_path / "empty.pth"

        save_merged_model(_TinyModel(), out_none, class_names=None)
        save_merged_model(_TinyModel(), out_empty, class_names=[])

        ckpt_none = torch.load(out_none, map_location="cpu", weights_only=False)
        ckpt_empty = torch.load(out_empty, map_location="cpu", weights_only=False)

        assert ckpt_none["class_names"] == ckpt_empty["class_names"] == []

    def test_extra_metadata_merged_into_metadata_dict(self, tmp_path: Path):
        model = _TinyModel()
        out = tmp_path / "model.pth"

        save_merged_model(
            model,
            out,
            extra_metadata={"epochs": 12, "mAP50": 0.83, "git_sha": "abc1234"},
        )

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        # Defaults still present
        assert ckpt["metadata"]["format"] == "merged_grounding_dino"
        # Extras layered in
        assert ckpt["metadata"]["epochs"] == 12
        assert ckpt["metadata"]["mAP50"] == 0.83
        assert ckpt["metadata"]["git_sha"] == "abc1234"

    @pytest.mark.xfail(
        reason=(
            "BUG: extra_metadata is .update()-merged after the framework defaults, "
            "so a caller can silently overwrite peft_merged=True or the format string. "
            "load_merged_model's format-prefix check then misfires. Fix: validate "
            "extra_metadata keys don't collide with reserved metadata keys, or merge "
            "extras BEFORE defaults so framework values win."
        ),
        strict=True,
    )
    def test_extra_metadata_cannot_clobber_framework_defaults(self, tmp_path: Path):
        """
        A caller passing extra_metadata={'peft_merged': False, 'format': 'CUSTOM'}
        should NOT be able to silently invert the framework's own metadata. Currently
        it can. This test asserts the desired (safer) behavior; xfail tracks the bug.
        """
        out = tmp_path / "clobbered.pth"
        save_merged_model(
            _TinyModel(),
            out,
            extra_metadata={"peft_merged": False, "format": "tampered"},
        )

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt["metadata"]["peft_merged"] is True  # FAILS today
        assert ckpt["metadata"]["format"] == "merged_grounding_dino"  # FAILS today

    def test_creates_parent_directories(self, tmp_path: Path):
        model = _TinyModel()
        out = tmp_path / "exports" / "v2" / "deep" / "model.pth"

        save_merged_model(model, out)

        assert out.exists()

    def test_model_name_propagates_into_format_string(self, tmp_path: Path):
        model = _TinyModel()
        out = tmp_path / "sam.pth"

        save_merged_model(model, out, model_name="sam")

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt["metadata"]["format"] == "merged_sam"

    def test_overwrites_existing_file(self, tmp_path: Path):
        """torch.save overwrites without warning. Documenting this behavior."""
        out = tmp_path / "model.pth"
        save_merged_model(_TinyModel(), out, class_names=["v1"])

        # Second call to the same path with different class_names
        save_merged_model(_TinyModel(), out, class_names=["v2"])

        ckpt = torch.load(out, map_location="cpu", weights_only=False)
        assert ckpt["class_names"] == ["v2"]


# ===========================================================================
# load_merged_model — error paths + strict-mode behavior
# ===========================================================================


class TestLoadMergedModel:
    def test_loads_state_dict_into_target_model(self, tmp_path: Path):
        """Round-trip: save then load produces matching parameters."""
        source = _TinyModel()
        with torch.no_grad():
            source.linear.weight.fill_(0.42)

        out = tmp_path / "model.pth"
        save_merged_model(source, out, class_names=["a"])

        target = _TinyModel()
        assert not torch.equal(source.linear.weight, target.linear.weight)

        loaded = load_merged_model(out, target)

        assert loaded is target
        assert torch.equal(loaded.linear.weight, source.linear.weight)

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path: Path):
        target = _TinyModel()
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_merged_model(tmp_path / "missing.pth", target)

    def test_non_merged_format_logs_warning_but_loads(self, tmp_path: Path, caplog):
        """A checkpoint without 'merged_*' format prefix loads but warns."""
        source = _TinyModel()
        out = tmp_path / "raw.pth"
        torch.save(
            {
                "model_state_dict": source.state_dict(),
                "class_names": [],
                "metadata": {"format": "raw_state_dict"},
            },
            out,
        )

        target = _TinyModel()
        with caplog.at_level("WARNING"):
            load_merged_model(out, target)

        assert any("may not be a merged model format" in rec.message for rec in caplog.records)

    def test_strict_true_rejects_missing_keys(self, tmp_path: Path):
        """Default strict=True: target with extra params doesn't match → RuntimeError."""
        source = _TinyModel()
        out = tmp_path / "model.pth"
        save_merged_model(source, out)

        target = _TinyModelExtraParam()  # has 'extra.weight', 'extra.bias' that source lacks
        with pytest.raises(RuntimeError, match="(?i)missing|unexpected"):
            load_merged_model(out, target, strict=True)

    def test_strict_false_accepts_missing_keys(self, tmp_path: Path):
        """strict=False: extra params in target stay at their init values, no error."""
        source = _TinyModel()
        with torch.no_grad():
            source.linear.weight.fill_(0.7)

        out = tmp_path / "model.pth"
        save_merged_model(source, out)

        target = _TinyModelExtraParam()
        loaded = load_merged_model(out, target, strict=False)

        # The shared param was loaded
        assert torch.equal(loaded.linear.weight, source.linear.weight)
        # The extra param wasn't in the checkpoint — keeps its random init (just verify it exists)
        assert hasattr(loaded, "extra")

    def test_strict_true_rejects_shape_mismatch(self, tmp_path: Path):
        """Mismatched tensor shapes always raise, regardless of strict."""
        source = _TinyModel()
        out = tmp_path / "model.pth"
        save_merged_model(source, out)

        target = _TinyModelDifferentShape()  # linear.weight is (2,8) vs source's (2,4)
        with pytest.raises(RuntimeError, match=r"(?i)size mismatch|shape"):
            load_merged_model(out, target, strict=True)

    @pytest.mark.xfail(
        reason=(
            "BUG: load_merged_model assumes checkpoint['metadata'] is a dict and "
            "checkpoint['model_state_dict'] exists. Saving a tampered/legacy checkpoint "
            "where these assumptions don't hold raises raw KeyError / AttributeError "
            "instead of a clean error like 'malformed merged-model checkpoint'. Fix: "
            "wrap the metadata-and-state-dict access in explicit checks."
        ),
        strict=True,
    )
    def test_malformed_checkpoint_raises_clean_error(self, tmp_path: Path):
        """A checkpoint missing the model_state_dict key should fail loudly, not via KeyError."""
        out = tmp_path / "bad.pth"
        torch.save({"metadata": {"format": "merged_grounding_dino"}}, out)  # missing model_state_dict

        target = _TinyModel()
        # Currently raises KeyError, not a domain-meaningful error
        with pytest.raises(RuntimeError, match=r"(?i)malformed|missing"):
            load_merged_model(out, target)

    @pytest.mark.xfail(
        reason=(
            "BUG: when checkpoint['metadata'] is a non-dict (e.g. a string from a "
            "legacy save), `metadata.get('format', '')` raises AttributeError. The "
            "code should treat unexpected metadata types as 'unknown format' and "
            "warn rather than crash."
        ),
        strict=True,
    )
    def test_non_dict_metadata_raises_clean_error(self, tmp_path: Path):
        """metadata: not_a_dict should not crash with AttributeError."""
        out = tmp_path / "weird.pth"
        torch.save(
            {
                "model_state_dict": _TinyModel().state_dict(),
                "metadata": "this used to be a string",
            },
            out,
        )

        target = _TinyModel()
        with pytest.raises(RuntimeError, match=r"(?i)malformed|unknown"):
            load_merged_model(out, target)
