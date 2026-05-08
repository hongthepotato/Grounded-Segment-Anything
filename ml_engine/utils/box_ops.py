"""Bias-free pairwise box IoU / GIoU.

The vendored ``groundingdino.util.box_ops.box_iou`` adds ``+ 1e-6`` to the IoU
denominator (and the matching term in ``generalized_box_iou``), which biases
self-IoU below 1.0 with error ``1e-6 / (area + 1e-6)``. For normalized cxcywh
coordinates this is invisible at typical box sizes (~2.5e-5 at 20%×20%) but
reaches ~1% at 1%×1% and ~50% at 0.1%×0.1% — distorting both training-loss
gradient signal and inference-time IoU thresholds on small objects, with no
upper-bound guard from the degeneracy assertion (which uses ``>=``, allowing
zero-area "point" boxes).

Fix: bare division, with ``clamp(min=1e-12)`` as defense in depth. The clamp
threshold is many orders of magnitude smaller than any realistic normalized
area (a 1px box in a 4K image is ~6e-8) so it is a strict no-op for live
inputs and only activates for genuinely degenerate boxes — preventing 0/0
without distorting valid IoU values the way ``+ 1e-6`` did.

fp16 caveat: ``1e-12`` underflows to 0 in fp16, making the clamp a no-op
there. In practice the criterion mixes fp16 pred boxes with fp32 target
boxes from the dataloader, which promotes the IoU computation to fp32
(making the clamp effective). A fully-fp16 pipeline (uncommon) would
re-expose 0/0 NaN risk on degenerate boxes; tracked in TODOS #39.

Use these in place of ``groundingdino.util.box_ops.box_iou`` /
``generalized_box_iou`` everywhere in ``ml_engine/``. The vendored
``box_cxcywh_to_xyxy`` is mathematically correct (no division) and may
continue to be imported from groundingdino directly.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchvision.ops.boxes import box_area

DENOM_EPS = 1e-12


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
    iou = inter / union.clamp(min=DENOM_EPS)
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

    return iou - (enclosing - union) / enclosing.clamp(min=DENOM_EPS)
