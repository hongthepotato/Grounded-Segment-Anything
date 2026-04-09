---
description: "Strategy for student model (lightweight segmentation) distillation stage."
tools:
  - dispatch_stage
  - inspect_status
  - read_memory
  - request_evaluation
  - advance_gate
---

## Goal

Distill knowledge from the teacher (GroundingDINO + SAM) into a lightweight student model.
Gate metric: val_mIoU >= acceptance_criteria.min_mIoU.

This is the last stage before `pending_approval` -- never advance to done without the human gate.

## Decision framework

**Before dispatching:**
- Confirm auto_label_complete in stage_summaries. Auto-labels are the student's training data.
- Check label_rate from auto_label summary. If < 0.6, the training data is weak -- note this and set expectations lower in the memory record before dispatching.
- Read memory for prior distillation failures.

**Config overrides:**
- `distillation.temperature`: default 4.0. Raise to 6.0 if teacher mAP50 was high (>0.75) for softer targets.
- `distillation.alpha` (distill loss weight): default 0.7. If student is underfitting (mIoU < 0.3 after 10 epochs), lower to 0.5.
- `training_dynamics.epochs`: default 30. Use 50 for datasets > 2k images.

**After gate_decision arrives:**
- `pass`: do NOT auto-advance to done. Transition to `pending_approval`. Human must approve any production model swap.
- `retry`: dispatch again with adjusted temperature / epochs.
- `escalate`: transition to `escalated`. Write what metric was missing or why budget is exhausted.

## Human gate (non-negotiable)

After student passes the mIoU gate, the state must be `pending_approval`.
The frontend shows an actionable alert. The human reviews metrics and either:
- Approves -> `done`
- Rejects -> `escalated` with their feedback

Never call advance_gate(target_state="done") directly. That path requires human approval via POST /api/agent/gate/{run_id}/approve.

## Metric priorities

1. `val_mIoU` -- primary gate metric
2. `val_mAP50` on student -- secondary, not required
3. Compare to teacher's mAP50: student should not be more than 15 points behind

## Memory write on success

When student passes mIoU gate, write to memory:
```
type: project
key: distillation_result_{run_id}
body: "Student mIoU={mIoU:.3f}. Teacher mAP50={teacher_mAP50:.3f}. 
       Config used: {overrides}. Pending human approval."
```
