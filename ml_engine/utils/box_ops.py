"""Bias-free pairwise box IoU / GIoU.

The vendored ``groundingdino.util.box_ops.box_iou`` adds ``+ 1e-6`` to the IoU
denominator (and the matching term in ``generalized_box_iou``), which biases
self-IoU below 1.0 with error ``1e-6 / (area + 1e-6)``. For normalized cxcywh
coordinates this is invisible at typical box sizes (~2.5e-5 at 20%×20%) but
reaches ~1% at 1%×1% and ~50% at 0.1%×0.1% — distorting both training-loss
gradient signal and inference-time IoU thresholds on small objects, with no
upper-bound guard from the degeneracy assertion (which uses ``>=``, allowing
zero-area "point" boxes).

Fix: bare division, with two layered defenses against precision corner cases.

1. **Low-precision inputs (fp16 / bf16) are promoted to fp32 at function
   entry.** Closes issue #91. Without this, the ``enclosing`` denominator in
   ``generalized_box_iou`` lands in fp16 on the all-fp16 path and
   ``finfo(fp16).tiny == 6.10e-5`` clobbers valid sub-tiny enclosing areas —
   distorting GIoU ~40% off truth on a 1e-3-side distinct-pair. fp32 lifts
   that floor to 1.18e-38 (a true no-op for any realistic area) and the
   computation runs cleanly. fp64 inputs are left alone (already higher
   precision than fp32).

2. **Final clamp ``clamp(min=torch.finfo(t.dtype).tiny)``** as defense in depth.
   After step 1 every internal tensor is fp32 or fp64, where ``tiny`` is so
   small (1.18e-38 / 2.22e-308) it only fires on a genuinely zero-area box.
   Effect on a degenerate box: ``0 / 0`` becomes ``0 / tiny == 0``, so callers
   see IoU = 0 instead of NaN.

Output dtype, by call path:

* fp16 / bf16 input → fp32 output (from the explicit promotion above).
* fp32 input → fp32 output.
* fp64 input → fp64 output (preserved, never downcast).

This matches the prior de-facto behavior exactly: ``torchvision.ops.box_area``
already upcast fp16/bf16 → fp32, so the production output was always fp32 for
low-precision inputs. We do the promotion explicitly so the function no longer
depends on torchvision's upcast remaining in place — a regression-canary in
``tests/unit/test_losses.py::TestBoxOpsDtypeSafety`` still locks that in as
defense in depth.

Use these in place of ``groundingdino.util.box_ops.box_iou`` /
``generalized_box_iou`` everywhere in ``ml_engine/``. The vendored
``box_cxcywh_to_xyxy`` is mathematically correct (no division) and may
continue to be imported from groundingdino directly.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.ops.boxes import box_area

# Dtypes whose ``finfo(dtype).tiny`` is large enough (>= ~6e-5) to clobber
# valid sub-tiny areas in the IoU/GIoU denominators. We promote these to
# at least fp32 at function entry. fp32/fp64 are not in this set — their
# ``tiny`` is essentially zero, so the final clamp is a true no-op for any
# realistic box.
_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _promote_target(d1: torch.dtype, d2: torch.dtype) -> torch.dtype:
    """Common target dtype for both box inputs.

    Low-precision (fp16 / bf16) is lifted to fp32 first, then the result is
    ``torch.promote_types``'d with the other dtype. This preserves fp64 when
    the OTHER input is fp64 (``box_iou(fp16, fp64) → fp64``) instead of
    silently downcasting it, while still pulling fp16/bf16 up to at least
    fp32 to avoid the denominator-clamp distortion #91 fixed.
    """
    t1 = torch.float32 if d1 in _LOW_PRECISION else d1
    t2 = torch.float32 if d2 in _LOW_PRECISION else d2
    return torch.promote_types(t1, t2)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU + union for boxes in xyxy format.

    Returns ``(iou, union)`` where both are ``[N, M]`` tensors.
    """
    if boxes1.dtype in _LOW_PRECISION or boxes2.dtype in _LOW_PRECISION:
        target = _promote_target(boxes1.dtype, boxes2.dtype)
        boxes1 = boxes1.to(target)
        boxes2 = boxes2.to(target)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=torch.finfo(union.dtype).tiny)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU for boxes in xyxy format. Returns ``[N, M]`` in ``[-1, 1]``."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 has degenerate xyxy"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 has degenerate xyxy"

    if boxes1.dtype in _LOW_PRECISION or boxes2.dtype in _LOW_PRECISION:
        target = _promote_target(boxes1.dtype, boxes2.dtype)
        boxes1 = boxes1.to(target)
        boxes2 = boxes2.to(target)

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:, :, 0] * wh[:, :, 1]

    return iou - (enclosing - union) / enclosing.clamp(min=torch.finfo(enclosing.dtype).tiny)
