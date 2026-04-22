"""
Loss functions for teacher model fine-tuning.

This module provides loss functions for:
- Grounding DINO: Detection loss (classification + box regression)
- SAM: Segmentation loss (mask IoU + focal loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from groundingdino.util.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and ground truths.
    
    This module computes an assignment between targets and predictions using the
    Hungarian algorithm (linear_sum_assignment from scipy).
    
    The matching cost has three components:
    1. Classification cost (focal loss cost)
    2. L1 cost between boxes
    3. GIoU cost between boxes
    
    Costs from Grounding DINO paper (Table in appendix):
    - cost_class: 1.0 (for matching), 2.0 (for loss)
    - cost_bbox: 5.0
    - cost_giou: 2.0
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        """
        Args:
            cost_class: Weight for classification cost in matching
            cost_bbox: Weight for L1 box cost in matching
            cost_giou: Weight for GIoU cost in matching
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching with proper -inf filtering.

        Args:
            outputs: Dict with:
                - 'pred_logits': [B, N, num_tokens] token-level similarities
                - 'pred_boxes': [B, N, 4] predicted boxes in [cx, cy, w, h] format
                - 'text_token_mask': [B, num_valid_tokens] optional mask
            targets: List of dicts (len=B), each with:
                - 'labels': [M] class labels (0-indexed)
                - 'boxes': [M, 4] boxes in [cx, cy, w, h] format
                - 'token_labels': [M, num_tokens] token-level targets

        Returns:
            List of (pred_idx, tgt_idx) tuples, one per batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten batch dimension for cost computation
        out_logits = outputs["pred_logits"].flatten(0, 1)  # [B*N, num_tokens]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 4]

        # Concatenate targets across batch
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [total_targets, 4]

        # Get token-level targets
        assert "token_labels" in targets[0], "token_labels required for matching!"
        # [total_targets, num_tokens]
        tgt_token_labels = torch.cat([v["token_labels"] for v in targets])

        # Classification cost: token-level focal loss (sigmoid, not softmax).
        # Grounding DINO uses multi-label token prediction — softmax is wrong here.
        # Filter valid token positions only
        text_token_mask = outputs.get('text_token_mask', None)
        if text_token_mask is not None:
            # Create mask [B*N, num_tokens]
            B, num_valid = text_token_mask.shape
            num_tokens = out_logits.shape[-1]
            text_mask = torch.zeros((B, num_tokens), dtype=torch.bool, device=out_logits.device)
            text_mask[:, :num_valid] = text_token_mask
            text_mask = text_mask.unsqueeze(1).expand(-1, num_queries, -1).reshape(B * num_queries, -1)  # [B*N, num_tokens]
        else:
            # Fallback: use -inf detection
            text_mask = ~torch.isinf(out_logits)

        # Only compute focal cost on valid positions
        out_prob = out_logits.sigmoid()  # [B*N, num_tokens]

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # Mask out invalid positions via torch.where, NOT a post-hoc multiply.
        # GroundingDINO's ContrastiveEmbed fills padded text positions with -inf,
        # and sigmoid + log(p + eps) can produce ±inf at those positions under
        # reduced precision (FP16's 1e-8 epsilon underflows to 0, log(0) = -inf).
        # A subsequent `cost * mask.float()` would turn those inf values into
        # NaN via 0 * inf, regardless of dtype. torch.where selects the zero
        # branch directly at masked positions, so the NaN class is eliminated
        # independently of the numerical properties of the arithmetic above.
        neg_cost_class = torch.where(text_mask, neg_cost_class, 0.0)
        pos_cost_class = torch.where(text_mask, pos_cost_class, 0.0)

        # Compute cost for each query-target pair via matmul
        # tgt_token_labels: [total_targets, num_tokens]
        # pos/neg_cost_class: [B*N, num_tokens]
        cost_class = (
            pos_cost_class @ tgt_token_labels.T
            - neg_cost_class @ (1 - tgt_token_labels).T
        )  # [B*N, total_targets]

        # L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*N, total_targets]

        # GIoU cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )  # [B*N, total_targets]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()  # [B, N, total_targets]

        # Split by batch and compute Hungarian matching
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return as list of (pred_idx, tgt_idx) tensors
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


class GroundingDINOCriterion(nn.Module):
    """
    Complete loss computation for Grounding DINO.
    
    This implements the full DETR-style training pipeline:
    1. Hungarian matching between predictions and targets
    2. Token-level focal loss for classification (following GLIP)
    3. L1 + GIoU loss for box regression
    4. Auxiliary losses from all decoder layers
    5. Encoder auxiliary loss
    
    Loss weights from Grounding DINO paper:
    - Matching costs: class=1.0, bbox=5.0, giou=2.0
    - Loss weights: class=2.0, bbox=5.0, giou=2.0
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        losses: Optional[List[str]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            num_classes: Number of object classes
            matcher: HungarianMatcher instance
            weight_dict: Dict with loss weights, e.g.:
                {
                    'loss_ce': 2.0,
                    'loss_bbox': 5.0,
                    'loss_giou': 2.0,
                    'loss_ce_0': 2.0,  # Auxiliary losses
                    'loss_bbox_0': 5.0,
                    ...
                }
            losses: List of losses to compute. Defaults to ['labels', 'boxes'].
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = list(losses) if losses is not None else ['labels', 'boxes']
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        log: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Token-level classification loss using focal loss.
        
        CRITICAL: Compute loss for ALL queries, not just matched ones!
        - Matched queries: target is positive_map (binary labels for each token)
        - Unmatched queries: target is all zeros (background)
        
        This follows MMDetection's approach where all 900 queries participate
        in training, and unmatched queries learn to output "no object".
        
        Loss normalization follows MMDetection:
        - avg_factor = num_total_pos + num_total_neg * bg_cls_weight
        - Where num_total_pos/neg are QUERY counts, not token counts
        - This ensures proper gradient scaling
        
        Args:
            outputs: Model outputs with:
                - 'pred_logits': [B, N, num_tokens]
                - 'text_token_mask': [B, num_valid_tokens] boolean mask
            targets: List of target dicts with 'token_labels'
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
            log: Whether to log metrics
        
        Returns:
            Dict with 'loss_ce' and optionally 'class_error'
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, N, num_tokens]
        batch_size, num_queries, max_text_len = src_logits.shape

        # Create target labels for ALL queries
        target_labels = torch.zeros_like(src_logits)  # [B, N, max_text_len]

        # Count matched and unmatched queries for avg_factor
        num_total_pos = 0  # Number of matched queries
        num_total_neg = 0  # Number of unmatched queries

        # Only matched queries get their positive_map assigned
        assert "token_labels" in targets[0], "token_labels must be provided!"
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                # src_idx: queries are matched
                # tgt_idx: ground truth boxes they match to
                # fill-in a 256 length vector of token labels for each matched query.
                # .to(target_labels.dtype) handles the case where target_labels
                # inherited the autocast dtype (fp16/bf16) from src_logits but
                # token_labels is built as fp32 in dino_utils.py — index_put_
                # is strict about dtype match and would otherwise raise.
                target_labels[batch_idx, src_idx] = (
                    targets[batch_idx]['token_labels'][tgt_idx].to(target_labels.dtype)
                )
                num_total_pos += len(src_idx)
            num_total_neg += num_queries - len(src_idx)

        # ===== Build text mask to filter padded tokens =====
        text_token_mask = outputs.get('text_token_mask', None)

        if text_token_mask is not None:
            # Pad text_token_mask to max_text_len
            text_masks = torch.zeros((batch_size, max_text_len), dtype=torch.bool, device=src_logits.device)
            text_masks[:, :text_token_mask.shape[1]] = text_token_mask
            # Expand to [B, N, max_text_len]
            text_mask = text_masks.unsqueeze(1).expand(-1, num_queries, -1)
        else:
            # Fallback: detect valid positions using -inf
            text_mask = ~torch.isinf(src_logits)

        # ===== Filter padded tokens using masked_select =====
        src_logits_valid = torch.masked_select(src_logits, text_mask).contiguous()
        target_labels_valid = torch.masked_select(target_labels, text_mask).contiguous()

        # ===== Compute focal loss following MMDetection's normalization =====
        # avg_factor = num_total_pos + num_total_neg * bg_cls_weight
        # bg_cls_weight is typically 0.1 in DETR/DINO
        bg_cls_weight = 0.1
        avg_factor = num_total_pos * 1.0 + num_total_neg * bg_cls_weight
        avg_factor = max(avg_factor, 1.0)  # Prevent division by zero

        if src_logits_valid.numel() > 0:
            # Compute focal loss per element (no reduction)
            loss_ce = self._focal_loss(
                src_logits_valid,
                target_labels_valid,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
            # Sum and normalize by avg_factor (MMDetection approach)
            loss_ce = loss_ce.sum() / avg_factor
        else:
            loss_ce = torch.tensor(0.0, device=src_logits.device, requires_grad=True)

        losses = {'loss_ce': loss_ce}

        if log:
            # Recall on positive tokens — "what fraction of object tokens did we fire on?"
            # Token-level accuracy is useless here: with ~6 positive tokens out of ~6300,
            # a model that always predicts 0 scores 99.9% accuracy.
            pos_mask = target_labels_valid > 0.5
            if pos_mask.any():
                pos_recall = (src_logits_valid[pos_mask].sigmoid() > 0.5).float().mean() * 100
                losses['class_error'] = 100 - pos_recall
            else:
                losses['class_error'] = torch.tensor(100.0, device=src_logits.device)

        return losses

    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss per element (no reduction).
        
        Matches MMDetection's py_sigmoid_focal_loss implementation.
        
        Args:
            pred: Predictions (logits) [N]
            target: Targets (0 or 1) [N]
            alpha: Balancing factor
            gamma: Focusing parameter
        
        Returns:
            Per-element focal loss [N]
        """
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)

        # pt = p if target=1, else (1-p)
        # Actually, pt here denotes (1 - pt) in the Focal Loss paper
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

        # Focal weight: alpha * (1-pt)^gamma for positive, (1-alpha) * pt^gamma for negative
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)

        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight

        return loss

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Box regression loss: L1 + GIoU.
        
        Args:
            outputs: Model outputs with 'pred_boxes' [B, N, 4]
            targets: List of target dicts with 'boxes'
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
        
        Returns:
            Dict with 'loss_bbox' and 'loss_giou'
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Get matched predictions and targets
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes

        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following matched indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to appropriate loss function."""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs with:
                - 'pred_logits': [B, N, num_tokens] final predictions
                - 'pred_boxes': [B, N, 4] final predictions
                - 'aux_outputs': List of dicts with intermediate predictions (optional)
                - 'enc_outputs': Dict with encoder predictions (optional)
            targets: List of target dicts (len=B) with:
                - 'labels': [M] class labels
                - 'boxes': [M, 4] boxes
                - 'token_labels': [M, num_tokens] (optional)
        
        Returns:
            Dict with all loss components
        """
        # Separate auxiliary outputs
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k not in ['aux_outputs', 'enc_outputs']}

        # 1. Match final layer predictions to targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                   device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 2. Compute losses for final layer
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # 3. Compute auxiliary losses from intermediate decoder layers
        # Note: Propagate text_token_mask to auxiliary outputs for proper masking
        if 'aux_outputs' in outputs:
            text_token_mask = outputs.get('text_token_mask', None)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # Add text_token_mask to aux_outputs if available
                if text_token_mask is not None:
                    aux_outputs = {**aux_outputs, 'text_token_mask': text_token_mask}
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 4. Compute encoder auxiliary loss (binary classification)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # Add text_token_mask to enc_outputs if available
            text_token_mask = outputs.get('text_token_mask', None)
            if text_token_mask is not None:
                enc_outputs = {**enc_outputs, 'text_token_mask': text_token_mask}
            # For encoder, use binary objectness targets: "something is here, class unknown".
            # token_labels are set to all-ones so every valid token position is activated
            # uniformly — loss_labels will mask out padding via text_token_mask.
            # Using class-specific token_labels here would contradict the zeroed class labels
            # and train the encoder with the same signal as the final decoder layer.
            bin_targets = [{'labels': torch.zeros_like(t['labels']),
                           'boxes': t['boxes'],
                           'token_labels': torch.ones_like(t['token_labels'])} for t in targets]
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build_criterion(
    num_classes: int,
    num_decoder_layers: int = 6,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> GroundingDINOCriterion:
    """
    Build the Grounding DINO criterion with proper weights.
    
    Args:
        num_classes: Number of object classes
        num_decoder_layers: Number of decoder layers (default: 6)
        focal_alpha: Alpha for focal loss (default: 0.25)
        focal_gamma: Gamma for focal loss (default: 2.0)
    
    Returns:
        GroundingDINOCriterion instance
    """
    # Matching costs from paper
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
    )

    # Loss weights from paper
    weight_dict = {
        'loss_ce': 2.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }

    # Auxiliary loss weights (same as main loss)
    aux_weight_dict = {}
    for i in range(num_decoder_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    
    # Encoder loss weights
    aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()})
    
    weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    criterion = GroundingDINOCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )

    return criterion


class SegmentationLoss(nn.Module):
    """
    Segmentation loss for SAM fine-tuning.
    
    Combines:
    - Focal loss for binary mask prediction
    - Dice loss for overlap optimization
    - IoU loss for mask quality
    
    Example:
        >>> criterion = SegmentationLoss()
        >>> loss_dict = criterion(pred_masks, target_masks)
        >>> total_loss = loss_dict['loss']
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            loss_weights: Optional dict with weights for each loss component
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'focal': 20.0,
                'dice': 1.0,
                'iou': 1.0
            }
        self.loss_weights = loss_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss with padding mask support.
        
        Args:
            predictions: Dict with key 'pred_masks': [B, N, H, W]
            targets: Dict with keys:
                - 'masks': [B, N, H, W]
                - 'valid_mask': [B, N] boolean mask (True = valid, False = padding)
        
        Returns:
            Dict with loss components and total loss
        """
        pred_masks = predictions['pred_masks']
        target_masks = targets['masks']
        valid_mask = targets.get('valid_mask', torch.ones(target_masks.shape[:2], 
                                                          dtype=torch.bool, 
                                                          device=target_masks.device))
        
        # Apply valid mask to select only valid masks
        if valid_mask.any():
            # Flatten and select valid masks
            b, n = valid_mask.shape
            valid_indices = valid_mask.view(-1)  # [B*N]
            
            pred_masks_flat = pred_masks.view(b * n, -1)  # [B*N, H*W]
            target_masks_flat = target_masks.view(b * n, -1)  # [B*N, H*W]
            
            valid_pred = pred_masks_flat[valid_indices]  # [num_valid, H*W]
            valid_target = target_masks_flat[valid_indices]  # [num_valid, H*W]
            
            if len(valid_pred) > 0:
                # Focal loss
                loss_focal = self.sigmoid_focal_loss(valid_pred, valid_target)
                
                # Dice loss
                loss_dice = self.dice_loss(valid_pred, valid_target)
                
                # IoU loss
                loss_iou = self.iou_loss(valid_pred, valid_target)
            else:
                loss_focal = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
                loss_dice = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
                loss_iou = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        else:
            loss_focal = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
            loss_dice = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
            loss_iou = torch.tensor(0.0, device=pred_masks.device, requires_grad=True)
        
        # Total loss
        total_loss = (
            self.loss_weights['focal'] * loss_focal +
            self.loss_weights['dice'] * loss_dice +
            self.loss_weights['iou'] * loss_iou
        )
        
        return {
            'loss': total_loss,
            'loss_focal': loss_focal.detach(),
            'loss_dice': loss_dice.detach(),
            'loss_iou': loss_iou.detach()
        }
    
    def sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Sigmoid focal loss for binary masks.
        
        Args:
            inputs: Predicted masks (logits) [num_valid, H*W] — already flattened spatially
            targets: Target masks (binary) [num_valid, H*W]

        Returns:
            Mean focal loss across all valid masks
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Dice loss for segmentation, computed per mask and averaged.

        Note: called from forward() with pre-selected valid masks.

        Args:
            inputs: Predicted masks (logits) [num_valid, H*W] — already flattened spatially
            targets: Target masks (binary) [num_valid, H*W]
            smooth: Smoothing factor

        Returns:
            Mean dice loss across all valid masks
        """
        inputs = inputs.sigmoid()  # [num_valid, H*W]

        # Sum over spatial dimension (dim=1) to get per-mask intersection/area
        intersection = (inputs * targets).sum(dim=1)        # [num_valid]
        dice = (2. * intersection + smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + smooth  # [num_valid]
        )
        return (1 - dice).mean()

    def iou_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        IoU loss for segmentation, computed per mask and averaged.

        Note: called from forward() with pre-selected valid masks.

        Args:
            inputs: Predicted masks (logits) [num_valid, H*W] — already flattened spatially
            targets: Target masks (binary) [num_valid, H*W]
            smooth: Smoothing factor

        Returns:
            Mean IoU loss across all valid masks
        """
        inputs = inputs.sigmoid()  # [num_valid, H*W]

        # Per-mask intersection and union
        intersection = (inputs * targets).sum(dim=1)        # [num_valid]
        union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection  # [num_valid]
        iou = (intersection + smooth) / (union + smooth)    # [num_valid]
        return (1 - iou).mean()
