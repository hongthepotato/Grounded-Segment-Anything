---
description: "Strategy for the auto-labeling stage (GroundingDINO + SAM)."
tools:
  - dispatch_stage
  - inspect_status
  - read_memory
  - advance_gate
---

## Goal

Generate pseudo-labels for unlabeled images using the teacher model (GroundingDINO) and SAM.
Quality bar: enough annotations to train a student, not a publication-quality dataset.

## Decision framework

**Before dispatching:**
- Confirm teacher_training_complete is in stage_summaries. Do not run auto_label before teacher passes its gate.
- Check contract.data.image_paths length. If < 50 images, auto_label is low value -- consider skipping and advancing directly to student_distillation with manual labels if available.
- Read memory for prior auto_label failures (OOM, bad confidence threshold).

**Config overrides:**
- `confidence_threshold`: default 0.3. If teacher val_mAP50 > 0.7, can raise to 0.4 for cleaner labels.
- `iou_threshold`: default 0.5. Raise to 0.6 if prior auto_label produced overlapping boxes.
- `sam_model_type`: "vit_b" for GPU < 8GB, "vit_h" for larger datasets.

**After job_completed:**
- Check `outcome.metrics.labeled_count` and `outcome.metrics.label_rate`.
- If `label_rate < 0.5` (less than half of images got labels): escalate with reason.
- If `label_rate >= 0.5`: advance to `auto_label_complete`.

## What not to do

- Never treat confidence_threshold as a quality gate by itself. Low confidence can still produce good segmentation masks.
- Never dispatch auto_label without a completed teacher model. The labels will be garbage.
- Never retry auto_label more than once for the same dataset -- if it fails twice, the issue is the teacher, not the labeler.

## Failure modes

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| label_rate = 0 | Wrong data_path or image format | Escalate, check paths |
| OOM in logs | SAM vit_h on small GPU | Retry with sam_model_type=vit_b |
| label_rate = 1.0 but boxes look wrong | Threshold too low | Flag in memory, raise threshold on retry |
