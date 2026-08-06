"""
Unified model factory for inference backends.

Centralizes model artifact resolution and backend construction so orchestration
code does not need backend-specific loading logic.
"""

import logging
from pathlib import Path

import torch

from ml_engine.inference.config import (
    DETECTOR_SOURCE_BASE_LORA,
    DETECTOR_SOURCE_CHECKPOINT,
    SEGMENTER_MOBILE_SAM,
    SEGMENTER_SAM_HQ,
    GroundingDINOModelSpec,
    SegmenterModelSpec,
)
from ml_engine.inference.detectors.base import DetectorProtocol
from ml_engine.inference.detectors.grounding_dino import GroundingDINODetector
from ml_engine.inference.segmenters.base import SegmenterProtocol

logger = logging.getLogger(__name__)


class InferenceModelFactory:
    """Build detector/segmenter instances from typed model specs."""

    def __init__(self, device: str):
        self.device = device

    def create_detector(self, spec: GroundingDINOModelSpec) -> DetectorProtocol:
        """Construct detector from model spec."""
        checkpoint_path = self._resolve_dino_checkpoint(spec)
        return GroundingDINODetector(
            config_path=spec.config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

    def create_segmenter(self, spec: SegmenterModelSpec) -> SegmenterProtocol:
        """Construct segmenter from model spec."""
        if spec.backend == SEGMENTER_SAM_HQ:
            from ml_engine.inference.segmenters.sam_hq import SAMHQSegmenter

            if not spec.base_checkpoint:
                raise ValueError("segmenter.base_checkpoint is required when backend='sam_hq'")
            return SAMHQSegmenter(
                base_checkpoint=spec.base_checkpoint,
                lora_adapter_path=spec.lora_adapter_path,
                model_type=spec.model_type,
                device=self.device,
            )

        if spec.backend == SEGMENTER_MOBILE_SAM:
            from ml_engine.inference.segmenters.mobile_sam import MobileSAMSegmenter

            if not spec.checkpoint_path:
                raise ValueError("segmenter.checkpoint_path is required when backend='mobile_sam'")
            return MobileSAMSegmenter(
                checkpoint_path=spec.checkpoint_path,
                device=self.device,
            )

        raise ValueError(f"Unknown segmenter backend: {spec.backend}")

    @staticmethod
    def _merge_inputs_newer_than(merged_ckpt: Path, spec: GroundingDINOModelSpec) -> bool:
        """
        True if the base checkpoint or any adapter file is newer than the cache.

        Standard derived-artifact staleness test (same rule as make/ninja):
        re-derive when any input outdates the output. mtime is a proxy, not
        proof — it misses backdated swaps — but the exact alternative (hashing
        gigabytes of weights per resolve) would defeat the cache's purpose.
        """
        try:
            cache_mtime = merged_ckpt.stat().st_mtime
        except OSError as e:
            logger.info("Merged cache %s unreadable (%s); re-merging.", merged_ckpt, e)
            return True

        newest = 0.0
        try:
            base = Path(spec.base_checkpoint) if spec.base_checkpoint else None
            if base is not None and base.is_file():
                newest = base.stat().st_mtime
        except OSError as e:
            logger.info(
                "Cannot stat base checkpoint %s (%s); it contributes no staleness evidence.",
                spec.base_checkpoint,
                e,
            )
        try:
            adapter_dir = Path(spec.lora_adapter_path) if spec.lora_adapter_path else None
            if adapter_dir is not None and adapter_dir.is_dir():
                for f in adapter_dir.rglob("*"):
                    # Just ignore the hidden files
                    if f.name.startswith("."):
                        continue
                    try:
                        if f.is_file():
                            newest = max(newest, f.stat().st_mtime)
                    except OSError:
                        # Per-file: one unreadable stray must not blind the
                        # check to every OTHER file's mtime.
                        continue
        except OSError:
            pass
        stale = newest > cache_mtime
        if stale:
            logger.info("Merged cache %s is older than its inputs; re-merging.", merged_ckpt)
        return stale

    def _resolve_dino_checkpoint(self, spec: GroundingDINOModelSpec) -> str:
        """Resolve detector checkpoint path from source policy."""
        if spec.source == DETECTOR_SOURCE_CHECKPOINT:
            if not spec.checkpoint_path:
                raise ValueError("detector.checkpoint_path required for checkpoint source")
            return spec.checkpoint_path

        if spec.source != DETECTOR_SOURCE_BASE_LORA:
            raise ValueError(f"Unknown detector source: {spec.source}")

        if not spec.base_checkpoint or not spec.lora_adapter_path:
            raise ValueError(
                "detector.base_checkpoint and detector.lora_adapter_path are required for "
                "base_plus_lora source"
            )

        cache_path = spec.merged_cache_path
        if cache_path is None:
            cache_path = str(Path(spec.lora_adapter_path).parent / "_merged_for_inference.pth")
        merged_ckpt = Path(cache_path)
        if merged_ckpt.exists():
            if not self._merge_inputs_newer_than(merged_ckpt, spec):
                return str(merged_ckpt)

        logger.info("Merging GroundingDINO base + LoRA for inference...")
        from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora

        merged_model = load_grounding_dino_with_lora(
            base_checkpoint=spec.base_checkpoint,
            lora_adapter_path=spec.lora_adapter_path,
            merge=True,
        )
        merged_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": merged_model.model.state_dict()}, str(merged_ckpt))
        del merged_model
        logger.info("Saved merged checkpoint to %s", merged_ckpt)
        return str(merged_ckpt)
