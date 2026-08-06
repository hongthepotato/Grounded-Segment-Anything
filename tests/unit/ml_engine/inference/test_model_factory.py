"""
Bug-hunting tests for ml_engine.inference.model_factory.

Module context (traced before writing these):

- UPSTREAM: `_resolve_dino_checkpoint` is reached from
  AutoLabeler._factory.create_detector (auto_labeler.py:90) with specs built by
  distillation/pseudo_label.py from resolved teacher artifacts. In the
  orchestrated pipeline, worker.py embeds the job id in every output dir
  ({type}_{id[:8]}), so adapters land in a FRESH directory per training run.
- DOWNSTREAM: the resolved path is handed to GroundingDINODetector, which loads
  it via the vendored load_model -> checkpoint["model"] (strict=False). Whatever
  bytes sit at the cache path ARE the model that labels images.

The bug under test: the merged base+LoRA cache is trusted on existence alone
(`if merged_ckpt.exists(): return`). If the adapter at lora_adapter_path is ever
rewritten in place (a fixed "production adapter" path, scp-ing new weights over
old — impossible in the orchestrated path today, so LATENT), inference silently
serves the merge of the OLD adapter. No error, no log; every downstream
annotation comes from an outdated fine-tune. On a platform whose core value is
the fine-tuning loop, silently-wrong-model is the worst failure class.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_engine.inference.config import (
    DETECTOR_SOURCE_BASE_LORA,
    GroundingDINOModelSpec,
    SegmenterModelSpec,
)
from ml_engine.inference.model_factory import InferenceModelFactory

MERGE_FN = "ml_engine.models.teacher.grounding_dino_lora.load_grounding_dino_with_lora"


@pytest.fixture
def lora_layout(tmp_path):
    """A base checkpoint + adapter dir + pre-existing merged cache on disk."""
    base_ckpt = tmp_path / "base" / "groundingdino_swint_ogc.pth"
    base_ckpt.parent.mkdir()
    base_ckpt.write_bytes(b"base-weights")

    adapter_dir = tmp_path / "teacher_run" / "lora_adapters"
    adapter_dir.mkdir(parents=True)
    adapter_file = adapter_dir / "adapter_model.bin"
    adapter_file.write_bytes(b"adapter-v1")

    cache = adapter_dir.parent / "_merged_for_inference.pth"
    cache.write_bytes(b"merged-from-adapter-v1")

    return {
        "base_ckpt": base_ckpt,
        "adapter_dir": adapter_dir,
        "adapter_file": adapter_file,
        "cache": cache,
    }


def _spec(layout, merged_cache_path=None) -> GroundingDINOModelSpec:
    return GroundingDINOModelSpec(
        source=DETECTOR_SOURCE_BASE_LORA,
        base_checkpoint=str(layout["base_ckpt"]),
        lora_adapter_path=str(layout["adapter_dir"]),
        merged_cache_path=merged_cache_path,
    )


def _set_mtime(path: Path, epoch: float) -> None:
    os.utime(path, (epoch, epoch))


class TestMergedCacheStaleness:
    """The cache must be an optimization, never an alternative source of truth."""

    def test_retrained_adapter_invalidates_the_merged_cache(self, lora_layout):
        """
        REVEALING TEST (step 3): written against the CURRENT code, expected to
        FAIL on it.

        Scenario: the adapter file is REWRITTEN (retrained in place) after the
        merged cache was produced — adapter mtime is now newer than the cache.
        Serving the old cache means serving the old fine-tune. The factory must
        re-merge.
        """
        # Cache produced at t=1000; adapter retrained at t=2000.
        _set_mtime(lora_layout["cache"], 1_000)
        _set_mtime(lora_layout["adapter_file"], 2_000)

        factory = InferenceModelFactory(device="cpu")
        with (
            patch(MERGE_FN, MagicMock(return_value=MagicMock())) as merge,
            patch("ml_engine.inference.model_factory.torch.save"),
        ):
            resolved = factory._resolve_dino_checkpoint(_spec(lora_layout))

        assert resolved == str(lora_layout["cache"])
        merge.assert_called_once()  # stale inputs -> MUST re-merge, not serve old bytes

    def test_updated_base_checkpoint_invalidates_the_merged_cache(self, lora_layout):
        """Same trap for the other merge input: a newer BASE checkpoint."""
        _set_mtime(lora_layout["cache"], 1_000)
        _set_mtime(lora_layout["adapter_file"], 500)
        _set_mtime(lora_layout["base_ckpt"], 2_000)

        factory = InferenceModelFactory(device="cpu")
        with (
            patch(MERGE_FN, MagicMock(return_value=MagicMock())) as merge,
            patch("ml_engine.inference.model_factory.torch.save"),
        ):
            factory._resolve_dino_checkpoint(_spec(lora_layout))

        merge.assert_called_once()

    def test_fresh_cache_is_reused_without_remerging(self, lora_layout):
        """
        The other half of the contract: when the cache is NEWER than both
        inputs, it must be served as-is — the whole point of the cache is to
        skip the expensive merge. Guards the fix from over-invalidating.
        """
        _set_mtime(lora_layout["adapter_file"], 1_000)
        _set_mtime(lora_layout["base_ckpt"], 1_000)
        _set_mtime(lora_layout["cache"], 2_000)

        factory = InferenceModelFactory(device="cpu")
        with patch(MERGE_FN, MagicMock()) as merge:
            resolved = factory._resolve_dino_checkpoint(_spec(lora_layout))

        assert resolved == str(lora_layout["cache"])
        merge.assert_not_called()

    def test_newer_dotfile_in_adapter_dir_does_not_trigger_remerge(self, lora_layout):
        """
        Foreign dotfiles (NFS .nfsXXXX lockfiles, editor droppings) are not
        adapter inputs and must not count as staleness evidence. Without the
        skip, a lingering .nfs file newer than the cache would force a
        re-merge on EVERY resolve until it disappears.
        """
        _set_mtime(lora_layout["adapter_file"], 1_000)
        _set_mtime(lora_layout["base_ckpt"], 1_000)
        _set_mtime(lora_layout["cache"], 2_000)
        nfs_stray = lora_layout["adapter_dir"] / ".nfs000000000001"
        nfs_stray.write_bytes(b"lock")
        _set_mtime(nfs_stray, 3_000)  # newer than the cache

        factory = InferenceModelFactory(device="cpu")
        with patch(MERGE_FN, MagicMock()) as merge:
            resolved = factory._resolve_dino_checkpoint(_spec(lora_layout))

        assert resolved == str(lora_layout["cache"])
        merge.assert_not_called()

    def test_equal_mtimes_count_as_fresh(self, lora_layout):
        """
        Boundary pin: staleness is STRICTLY newer (`>`), not `>=`. On coarse-
        mtime filesystems a cache written in the same clock tick as its inputs
        is legitimate; re-merging on equality would re-merge on every resolve,
        silently destroying the cache's purpose (wrong cost, right result --
        the kind of regression nothing else fails on).
        """
        for p in (lora_layout["adapter_file"], lora_layout["base_ckpt"], lora_layout["cache"]):
            _set_mtime(p, 1_000)

        factory = InferenceModelFactory(device="cpu")
        with patch(MERGE_FN, MagicMock()) as merge:
            resolved = factory._resolve_dino_checkpoint(_spec(lora_layout))

        assert resolved == str(lora_layout["cache"])
        merge.assert_not_called()

    def test_missing_cache_triggers_merge_and_save(self, lora_layout):
        """Characterizes the cold path: no cache -> merge once, save at the cache path."""
        lora_layout["cache"].unlink()

        factory = InferenceModelFactory(device="cpu")
        merged_model = MagicMock()
        with (
            patch(MERGE_FN, MagicMock(return_value=merged_model)) as merge,
            patch("ml_engine.inference.model_factory.torch.save") as save,
        ):
            resolved = factory._resolve_dino_checkpoint(_spec(lora_layout))

        assert resolved == str(lora_layout["cache"])
        merge.assert_called_once_with(
            base_checkpoint=str(lora_layout["base_ckpt"]),
            lora_adapter_path=str(lora_layout["adapter_dir"]),
            merge=True,
        )
        save.assert_called_once()
        # Saved under the envelope the vendored loader expects: {"model": state_dict}
        payload = save.call_args[0][0]
        assert set(payload.keys()) == {"model"}


class TestSpecValidation:
    """Error contracts: a misconfigured spec must fail loudly, not half-build."""

    def test_checkpoint_source_requires_checkpoint_path(self):
        factory = InferenceModelFactory(device="cpu")
        spec = GroundingDINOModelSpec(checkpoint_path=None)
        with pytest.raises(ValueError, match="checkpoint_path required"):
            factory._resolve_dino_checkpoint(spec)

    def test_unknown_detector_source_rejected(self):
        factory = InferenceModelFactory(device="cpu")
        spec = GroundingDINOModelSpec(source="frankenmodel")
        with pytest.raises(ValueError, match="Unknown detector source"):
            factory._resolve_dino_checkpoint(spec)

    def test_base_lora_source_requires_both_inputs(self):
        factory = InferenceModelFactory(device="cpu")
        spec = GroundingDINOModelSpec(source=DETECTOR_SOURCE_BASE_LORA, base_checkpoint="x")
        with pytest.raises(ValueError, match="base_checkpoint and .*lora_adapter_path"):
            factory._resolve_dino_checkpoint(spec)

    def test_sam_hq_backend_requires_base_checkpoint(self):
        factory = InferenceModelFactory(device="cpu")
        spec = SegmenterModelSpec(backend="sam_hq", base_checkpoint=None)
        with pytest.raises(ValueError, match="base_checkpoint is required"):
            factory.create_segmenter(spec)

    def test_unknown_segmenter_backend_rejected(self):
        factory = InferenceModelFactory(device="cpu")
        spec = SegmenterModelSpec(backend="sam9000")
        with pytest.raises(ValueError, match="Unknown segmenter backend"):
            factory.create_segmenter(spec)
