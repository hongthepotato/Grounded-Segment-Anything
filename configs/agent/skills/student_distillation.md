---
description: "Strategy for student model distillation stage (SAM segmentation or GroundingDINO detection)."
tools:
  - dispatch_stage
  - inspect_status
  - read_memory
  - request_evaluation
  - advance_gate
---

## Goal

Distill knowledge from the teacher (GroundingDINO + SAM) into a lightweight student model.
This stage handles two distillation paths depending on which teacher component is being compressed:

- **SAM path** (segmentation): student model is YOLOv8-seg. Gate metric: `mIoU` >= `acceptance_criteria.min_mIoU`.
- **GroundingDINO path** (detection): student model is YOLOv8 (no seg). Gate metric: `mAP50` >= `acceptance_criteria.min_mAP50`.

The gate automatically selects the right metric: if `mIoU` is present it takes priority; if only `mAP50` is present that is used. If neither is present, the gate escalates immediately.

This is the last stage before `pending_approval` -- never advance to done without the human gate.

## Decision framework

**Before dispatching:**
- Confirm auto_label_complete in stage_summaries. Auto-labels are the student's training data.
- Check `annotation_mode` from the job config or stage summary: `segmentation` → SAM path, `detection` → GroundingDINO path.
- Check label_rate from auto_label summary. If < 0.6, the training data is weak -- note this and set expectations lower in the memory record before dispatching.
- Read memory for prior distillation failures.

**Config overrides:**
- `distillation.temperature`: default 4.0. Raise to 6.0 if teacher mAP50 was high (>0.75) for softer targets.
- `distillation.alpha` (distill loss weight): default 0.7. If student is underfitting (primary metric < 0.3 after 10 epochs), lower to 0.5.
- `training_dynamics.epochs`: default 30. Use 50 for datasets > 2k images.

**After gate_decision arrives:**
- `pass`: do NOT auto-advance to done. Transition to `pending_approval`. Human must approve any production model swap.
- `retry`: dispatch again with adjusted temperature / epochs.
- `escalate`: transition to `escalated`. Write what metric was missing or why budget is exhausted.

## Human gate (non-negotiable)

After student passes the gate, the state must be `pending_approval`.
The frontend shows an actionable alert. The human reviews metrics and either:
- Approves -> `done`
- Rejects -> `escalated` with their feedback

Never call advance_gate(target_state="done") directly. That path requires human approval via POST /api/agent/gate/{run_id}/approve.

## Metric priorities

**SAM path (segmentation):**
1. `mIoU` (mapped from ultralytics `metrics/mAP50(M)`) -- primary gate metric
2. `mAP50` on student -- secondary, not required
3. Compare to teacher's mAP50: student should not be more than 15 points behind

**GroundingDINO path (detection):**
1. `mAP50` (mapped from ultralytics `metrics/mAP50(B)`) -- primary gate metric
2. Compare to teacher's mAP50: student should be within 10 points

## Memory write on success

When student passes the gate, write to memory:
```
type: project
key: distillation_result_{run_id}
body: "Student {metric_name}={metric_value:.3f}. Teacher mAP50={teacher_mAP50:.3f}. 
       Path: {sam|groundingdino}. Config used: {overrides}. Pending human approval."
```
