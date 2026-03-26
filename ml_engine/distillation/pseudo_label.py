"""
Pseudo-label generation using fine-tuned teacher models.

Runs fine-tuned GroundingDINO + SAM-HQ on unlabeled images to produce
COCO-format annotations for student training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from ml_engine.inference.auto_labeler import AutoLabeler
from ml_engine.inference.config import (
    AutoLabelerConfig,
    DetectionThresholds,
    GroundingDINOModelSpec,
    SegmenterModelSpec,
    DETECTOR_SOURCE_BASE_LORA,
    OUTPUT_BOTH,
    OUTPUT_MASKS_ONLY,
    SEGMENTER_SAM_HQ,
)
from ml_engine.inference.exporters.coco import COCOExporter
from ml_engine.artifacts import ResolvedArtifacts

logger = logging.getLogger(__name__)


def _build_autolabeler_config(
    artifacts: ResolvedArtifacts,
    distill_cfg: Dict[str, Any],
) -> AutoLabelerConfig:
    """Build AutoLabelerConfig from distillation policy + teacher artifacts."""
    pseudo_cfg = distill_cfg.get('pseudo_label', {})
    thr_cfg = pseudo_cfg.get('thresholds', {})

    thresholds = DetectionThresholds(
        box=thr_cfg.get('box', 0.3),
        text=thr_cfg.get('text', 0.3),
        nms=thr_cfg.get('nms', 0.7),
    )
    output_mode = pseudo_cfg.get('output_mode', OUTPUT_BOTH)

    if output_mode in (OUTPUT_BOTH, OUTPUT_MASKS_ONLY) and not artifacts.has_segmenter:
        raise ValueError(
            "Segmentation output mode requires a fine-tuned model. "
            "Please train a segmentation model first."
        )

    detector_spec = None
    segmenter_spec = None

    if artifacts.detector_adapter_dir:
        manifest = artifacts.detector_manifest
        detector_spec = GroundingDINOModelSpec(
            source=DETECTOR_SOURCE_BASE_LORA,
            config_path=manifest.base_model.config_path,
            base_checkpoint=manifest.base_model.checkpoint_path,
            lora_adapter_path=str(artifacts.detector_adapter_dir),
            merged_cache_path=str(artifacts.detector_merged) if artifacts.detector_merged else None,
        )

        logger.info("Using fine-tuned GroundingDINO model with LoRA adapter: %s", artifacts.detector_adapter_dir)

    if artifacts.segmenter_adapter_dir:
        manifest = artifacts.segmenter_manifest
        segmenter_spec = SegmenterModelSpec(
            backend=SEGMENTER_SAM_HQ,
            base_checkpoint=manifest.base_model.checkpoint_path,
            lora_adapter_path=str(artifacts.segmenter_adapter_dir),
            model_type=manifest.base_model.model_type,
        )

        logger.info("Using fine-tuned SAM-HQ model with LoRA adapter: %s", artifacts.segmenter_adapter_dir)

    return AutoLabelerConfig(
        detector=detector_spec,
        segmenter=segmenter_spec or SegmenterModelSpec(),
        thresholds=thresholds,
        output_mode=output_mode,
    )


def generate_pseudo_labels(
    image_paths: List[str],
    class_names: List[str],
    teacher_dir: str,
    output_path: str,
    distillation_cfg: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Generate COCO-format pseudo-labels for unlabeled images using fine-tuned teachers.

    Detects which teachers exist in teacher_dir and configures the AutoLabeler
    to use fine-tuned checkpoints.

    Args:
        image_paths: List of absolute paths to unlabeled images
        class_names: Class names for detection prompts
        teacher_dir: Path to teacher training output (contains lora_adapters/)
        output_path: Where to save the resulting COCO JSON
        progress_callback: Optional callback(current, total, message)

    Returns:
        COCO-format dictionary with images, annotations, categories
    """

    from ml_engine.artifacts import resolve_teacher_artifacts
    artifacts = resolve_teacher_artifacts(teacher_dir)
    if not artifacts.has_detector and not artifacts.has_segmenter:
        raise ValueError(
            f"No fine-tuned teachers found in {teacher_dir}. "
        )

    config = _build_autolabeler_config(artifacts, distillation_cfg)
    labeler = AutoLabeler(config)

    logger.info("Generating pseudo-labels for %d images...", len(image_paths))
    results = labeler.label_images(
        image_paths=image_paths,
        class_prompts=class_names,
        progress_callback=progress_callback,
    )

    coco_output = COCOExporter.export(
        results=results,
        class_prompts=class_names,
        output_mode=config.output_mode,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f)

    logger.info("Saved %d pseudo-labeled annotations to %s",
                len(coco_output.get('annotations', [])), output_path)

    return coco_output
