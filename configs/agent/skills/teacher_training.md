---
description: "Strategy for the teacher model (GroundingDINO) training stage."
tools:
  - dispatch_stage
  - inspect_status
  - read_memory
  - advance_gate
---

## Goal

Train the GroundingDINO teacher model to detect the target classes with val_mAP50 >= acceptance_criteria.min_mAP50.

## Decision framework

**Before dispatching:**
- Read memory for prior teacher_training feedback (failed runs, known bad configs).
- Check contract.budget.max_teacher_epochs. If already at limit, escalate.
- If a previous trial reached val_mAP50 >= 0.50, that config is a warm start -- use it.

**Config overrides to consider:**
- `training_dynamics.learning_rate`: if prior run hit NaN loss, halve it.
- `training_dynamics.batch_size`: start at 8 for datasets < 500 images.
- `experiment_loop.max_trials`: set to 3 for first run, 5 if dataset > 1k images.

**After gate decision arrives:**
- `pass`: advance to `teacher_training_complete`, then dispatch student_distillation.
- `retry` (mAP50 below threshold, retries left): dispatch again with tighter LR (× 0.5).
- `escalate` (retries exhausted or missing metrics): transition to `escalated`, write a clear reason to memory so the human knows exactly what to fix.

## Metric priorities

1. `val_mAP50` -- primary gate metric (from evaluation report, not training loss)
2. `val_grounding_dino_total_loss` -- useful for diagnosing divergence, not for gating
3. `best_metric` in outcome.json -- fallback if mAP50 field is missing

## What not to do

- Never gate on training loss alone. Loss going down does not mean mAP50 is acceptable.
- Never set epochs_per_trial < 5. Fewer epochs is false economy for LoRA convergence.
- Never dispatch a second stage concurrently with teacher_training.

## Escalation message template

When escalating, write to memory:
```
type: feedback
key: teacher_training_escalation_{run_id}
body: "Teacher training exhausted {n} retries. Best val_mAP50={best}. 
       Suggested fix: {specific recommendation}. 
       Contract required: {threshold}."
```
