"""Bias-free pairwise box IoU / GIoU.

The vendored ``groundingdino.util.box_ops.box_iou`` adds ``+ 1e-6`` to the IoU
denominator (and the matching term in ``generalized_box_iou``), which biases
self-IoU below 1.0 with error ``1e-6 / (area + 1e-6)``. For normalized cxcywh
coordinates this is invisible at typical box sizes (~2.5e-5 at 20%×20%) but
reaches ~1% at 1%×1% and ~50% at 0.1%×0.1% — distorting both training-loss
gradient signal and inference-time IoU thresholds on small objects, with no
upper-bound guard from the degeneracy assertion (which uses ``>=``, allowing
zero-area "point" boxes).

Fix: bare division, with ``clamp(min=torch.finfo(t.dtype).tiny)`` as defense
in depth. ``finfo(...).tiny`` is the smallest positive normal value for the
tensor's dtype: 1.18e-38 in fp32, 6.10e-5 in fp16, 2.22e-308 in fp64. This
floors the denominator at "the smallest non-zero value this dtype can hold"
without distorting valid IoU values — strictly a no-op for any non-degenerate
box, and the only effect on a degenerate (zero-area) box is to turn ``0 / 0``
into ``0 / tiny == 0`` so callers see IoU = 0 instead of NaN.

In current PyTorch + torchvision, the fp16/bf16 NaN path is also shielded
structurally:

* ``torchvision.ops.box_area`` upcasts fp16/bf16 → fp32 internally, so
  ``area1`` is fp32 even if ``boxes1`` is low-precision. ``union = area1
  + area2 - inter`` then mixes fp32 + low-precision → fp32 by promotion,
  so the union (and thus the union clamp) lands in fp32 regardless of
  input. Locked in by a regression-canary in
  ``tests/unit/test_losses.py::TestBoxOpsDtypeSafety``.
* ``torch.amp.autocast(fp16/bf16)`` casts matmul/conv-class ops only, not
  the element-wise ``max``/``clamp``/``+``/``*``/``/`` used here.
* The criterion mixes low-precision model predictions with fp32 dataloader
  targets, which promotes everything to fp32 anyway.

**Caveat — small-fp16-box GIoU distortion.** The ``enclosing`` denominator
in ``generalized_box_iou`` is computed directly from the input boxes
(``min``/``max``/``-``), NOT through ``box_area``. So on the all-fp16 path
its dtype is fp16, not fp32. ``finfo(fp16).tiny == 6.10e-5`` is large
enough to clobber valid sub-tiny enclosing areas — e.g., a 1e-3-side
distinct-pair gives an enclosing area of ~3.7e-5, which the clamp lifts
to 6.10e-5, distorting GIoU ~40% off truth. The prior ``1e-12`` literal
underflowed to 0 in fp16 and was a no-op there, so it actually gave
correct values for this case (~1% off, dominated by fp16 representation
noise) at the cost of NaN on degenerate boxes. The trade is intentional:
production paths never hit this case (mixed-dtype + box_area upcast →
fp32 throughout), and the dtype-aware clamp prevents NaN where the prior
clamp couldn't. The principled fix — fp32 promotion inside box_iou /
generalized_box_iou — is tracked as TODO #42.

Use these in place of ``groundingdino.util.box_ops.box_iou`` /
``generalized_box_iou`` everywhere in ``ml_engine/``. The vendored
``box_cxcywh_to_xyxy`` is mathematically correct (no division) and may
continue to be imported from groundingdino directly.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.ops.boxes import box_area


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU + union for boxes in xyxy format.

    Returns ``(iou, union)`` where both are ``[N, M]`` tensors.
    """
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
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:, :, 0] * wh[:, :, 1]

    return iou - (enclosing - union) / enclosing.clamp(min=torch.finfo(enclosing.dtype).tiny)
