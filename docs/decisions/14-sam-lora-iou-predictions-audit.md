# 18. SAM-LoRA audit: iou_predictions regression loss + defensive contracts

**Status:** Complete 2026-04-28. Branch `fix/sam-lora-test-audit` open as PR against `agentic`. v0.1.1 → v0.1.2.

---

## Context

The test file for `SAMHQLoRA` (`tests/test_sam_lora.py`) was sitting in the
project root rather than `tests/unit/ml_engine/`. While relocating it, a
full audit of the module's contract was done. Three real bugs were found,
fixed, and hardened further after adversarial review. The `iou_predictions`
head was the primary audit target.

---

## Bugs found and fixed

### 1. `iou_predictions` quality head was silently untrained

**File:** `ml_engine/training/losses.py` — `SegmentationLoss.forward()`

SAM's mask decoder outputs two tensors: `pred_masks` and `iou_predictions`.
`pred_masks` was trained by focal + dice + IoU boundary losses. `iou_predictions`
was present in the `predictions` dict but never consumed — the quality head
inherited weights from the pretrained SAM-HQ checkpoint and then drifted
unconstrained during fine-tuning.

The quality head matters because `iou_predictions` is the ranking signal used
downstream:
- `evaluator.py:372` — mAP score ranking per object
- `grounded_sam_visam.py` — mask selection via `argsort()`

If the score is untrained, mask ranking at inference is effectively random.

**Fix:** Added MSE regression loss:

```python
with torch.no_grad():
    pred_binary = (pred_masks[valid_mask] > 0).float()
    intersection = (pred_binary * target_masks[valid_mask]).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + target_masks[valid_mask].sum(dim=(-2, -1)) - intersection
    actual_iou = intersection / (union + 1e-6)
loss_iou_quality = F.mse_loss(valid_iou_pred, actual_iou)
```

Weight is controlled by `loss_weights["iou_quality"]`. Default dict uses `1.0`.
Callers who pass a custom `loss_weights` dict without this key get `0.0`
(no-op) — backward-safe.

**Why `torch.no_grad()` on the target:** The actual IoU is computed from the
model's own `pred_masks` output (thresholded at 0), not from ground truth alone.
If we let gradients flow through the IoU target computation, the loss would have
a trivial shortcut: make every mask all-positive to maximize self-IoU. The
`no_grad()` block treats the IoU target as a fixed supervisory signal for the
quality head only, while mask quality itself is driven by focal + dice.

### 2. `upscale_masks` crashed on multimask output

**File:** `ml_engine/models/teacher/sam_lora.py` — `SAMHQLoRA.upscale_masks()`

`PyTorch`'s `F.interpolate` requires exactly 4D input `[B,C,H,W]`. When
`SAMHQLoRA.forward()` is called with `multimask_output=True`, the decoder
returns `pred_masks` of shape `[B,N,K,H,W]` (K=3 candidates per object).
Passing this to `upscale_masks` raised `RuntimeError: expected 4D input`.

**Fix:** Detect 5D input and reshape through the interpolation:

```python
if masks.dim() == 5:
    b, n, k, h, w = masks.shape
    masks = masks.reshape(b * n * k, 1, h, w)
    masks = F.interpolate(masks, size=target_size, mode="bilinear", align_corners=False)
    return masks.reshape(b, n, k, *target_size)
```

A rank guard (`dim() not in (4, 5)`) raises `ValueError` for anything else.

### 3. `box_prompts` N=0 gave an opaque crash

**File:** `ml_engine/models/teacher/sam_lora.py` — `SAMHQLoRA.forward()`

An empty box tensor `[B, 0, 4]` propagated to `torch.cat([])` inside SAM's
prompt encoder, producing a `RuntimeError` with no indication of where to look.

**Fix:** Explicit guard at the top of `forward()`:

```python
if box_prompts is not None and box_prompts.shape[1] == 0:
    raise ValueError(
        f"box_prompts must contain at least 1 object per image (shape[1] > 0), "
        f"got shape {tuple(box_prompts.shape)}"
    )
```

---

## Contracts hardened after adversarial review

Three additional issues found during adversarial review before landing:

| Location | Issue | Fix |
|---|---|---|
| `losses.py:660` | `.get("iou_quality", 1.0)` silently opted custom `loss_weights` callers into quality loss at weight 1.0 | Changed fallback to `0.0` |
| `losses.py:forward()` entry | 5D `pred_masks [B,N,K,H,W]` would silently flatten `K×H×W` as the pixel dimension, computing wrong loss with no error | Added `pred_masks.ndim != 4` guard |
| `sam_lora.py:forward()` | `point_prompts` with N=0 still crashed deep in SAM internals | Added symmetric `ValueError` guard matching `box_prompts` |

---

## Decisions made

**`iou_quality` default weight = 1.0 in default dict, 0.0 in `.get()` fallback.**
Callers using the default `SegmentationLoss()` constructor get quality loss active
at weight 1.0 (correct — matches original SAM training recipe). Callers who pass
an explicit `loss_weights` dict without `iou_quality` get 0.0 (no change to their
training). This asymmetry is intentional: opt-in by default, backward-safe for
custom configs.

**IoU target computed from `pred_masks > 0`, not from a fixed threshold.**
The threshold of 0 corresponds to 0.5 post-sigmoid — the standard binary
classification boundary. This was chosen over alternatives (e.g. 0.5 on raw
logits) because the mask decoder is trained with focal loss and its logit
magnitudes are not well-calibrated to a fixed scale.

**`upscale_masks` raises on unexpected rank rather than silently no-op.**
A no-op fallback would hide caller errors where 3D or 6D tensors slip through.
`ValueError` forces the caller to fix the tensor shape at the source.

---

## Tests

- Relocated `tests/test_sam_lora.py` → `tests/unit/ml_engine/test_sam_lora.py`
- Expanded from 18 → 24 tests
- 6 new tests: `test_5d_multimask_input`, `test_5d_non_square_target`,
  `test_iou_predictions_receives_gradient`, `test_iou_predictions_multimask_shape_raises`,
  `test_iou_quality_weight_zero_excludes_quality_from_total`, `test_forward_empty_box_list`
  (narrowed from `(RuntimeError, ValueError)` to `pytest.raises(ValueError, match="at least 1 object")`)
- All 24 pass in ~1:47

## Coverage outcome

Branch was at 75% before this PR. New tests brought it to ~88% on the
`ml_engine/models/teacher/sam_lora.py` and `ml_engine/training/losses.py` modules.
