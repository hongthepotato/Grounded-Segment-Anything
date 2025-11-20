"""
Loss functions for teacher model fine-tuning.

This module provides loss functions for:
- Grounding DINO: Detection loss (classification + box regression)
- SAM: Segmentation loss (mask IoU + focal loss)
- Combined losses for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

# Import box utilities from GroundingDINO
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GroundingDINO"))
from groundingdino.util.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss

"""
Proper Grounding DINO Loss Implementation
Based on DETR (https://github.com/facebookresearch/detr) and Grounding DINO paper.

Key components:
1. HungarianMatcher for bipartite matching
2. SetCriterion for loss computation
3. Focal loss for token-level contrastive classification
4. L1 + GIoU for box regression
5. Auxiliary losses from all decoder layers + encoder
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

# Import box utilities from GroundingDINO
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GroundingDINO"))
from groundingdino.util.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss


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
        use_focal: bool = True
    ):
        """
        Args:
            cost_class: Weight for classification cost in matching
            cost_bbox: Weight for L1 box cost in matching
            cost_giou: Weight for GIoU cost in matching
            use_focal: Whether to use focal loss for classification cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: List[Dict[str, torch.Tensor]],
        tokenizer=None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.
        
        Args:
            outputs: Dict with:
                - 'pred_logits': [B, N, num_tokens] token-level similarities
                - 'pred_boxes': [B, N, 4] predicted boxes in [cx, cy, w, h] format
            targets: List of dicts (len=B), each with:
                - 'labels': [M] class labels (0-indexed)
                - 'boxes': [M, 4] boxes in [cx, cy, w, h] format
                - 'token_labels': [M, num_tokens] (optional) token-level targets
            tokenizer: Optional tokenizer for token-to-class mapping
        
        Returns:
            List of (pred_idx, tgt_idx) tuples, one per batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch dimension for cost computation
        # [B, N, num_tokens] -> [B*N, num_tokens]
        out_logits = outputs["pred_logits"].flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 4]

        # Concatenate targets across batch
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [total_targets]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [total_targets, 4]
        
        # Get token-level targets if available
        if "token_labels" in targets[0]:
            tgt_token_labels = torch.cat([v["token_labels"] for v in targets])  # [total_targets, num_tokens]
        else:
            # Fallback: create one-hot-like token labels
            # This is simplified - proper implementation needs tokenizer
            num_tokens = out_logits.shape[-1]
            tgt_token_labels = torch.zeros(len(tgt_ids), num_tokens, device=out_logits.device)
            # Simple strategy: assume uniform token distribution per class
            # In practice, you'd map class labels to their token spans

        # Classification cost using focal loss
        if self.use_focal:
            out_prob = out_logits.sigmoid()  # [B*N, num_tokens]
            
            # Focal loss cost computation
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            
            # Compute cost for each query-target pair
            # We need to expand targets to compute cost matrix
            # cost_class: [B*N, total_targets]
            cost_class = []
            for token_label in tgt_token_labels:
                # token_label: [num_tokens]
                pos_cost = (pos_cost_class * token_label.unsqueeze(0)).sum(-1)  # [B*N]
                neg_cost = (neg_cost_class * (1 - token_label.unsqueeze(0))).sum(-1)  # [B*N]
                cost_class.append(pos_cost - neg_cost)
            cost_class = torch.stack(cost_class, dim=1)  # [B*N, total_targets]
        else:
            # Softmax-based cost (not used in Grounding DINO)
            out_prob = out_logits.softmax(-1)
            cost_class = -out_prob[:, tgt_ids]

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
        losses: List[str] = ['labels', 'boxes'],
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
            losses: List of losses to compute ['labels', 'boxes']
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
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
        
        Args:
            outputs: Model outputs with 'pred_logits' [B, N, num_tokens]
            targets: List of target dicts
            indices: Matching indices from Hungarian matcher
            num_boxes: Number of boxes for normalization
            log: Whether to log metrics
        
        Returns:
            Dict with 'loss_ce' and optionally 'class_error'
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, N, num_tokens]

        # Get matched predictions
        idx = self._get_src_permutation_idx(indices)
        
        # Get target token labels for matched boxes
        if "token_labels" in targets[0]:
            target_token_labels = torch.cat([t["token_labels"][J] for t, (_, J) in zip(targets, indices)])
        else:
            # Fallback: create token labels from class labels
            target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            # Simplified: would need proper token mapping here
            num_tokens = src_logits.shape[-1]
            target_token_labels = torch.zeros(len(target_classes), num_tokens, device=src_logits.device)

        # Extract matched predictions
        src_logits_matched = src_logits[idx]  # [num_matched, num_tokens]
        
        # Compute focal loss on matched pairs
        loss_ce = sigmoid_focal_loss(
            src_logits_matched, 
            target_token_labels, 
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma
        ) * src_logits.shape[1]  # Scale by num_queries
        
        losses = {'loss_ce': loss_ce}

        if log:
            # Compute classification accuracy for logging
            # Take max prob token as prediction
            pred_classes = src_logits_matched.sigmoid().max(-1)[1]
            tgt_classes = target_token_labels.max(-1)[1]
            losses['class_error'] = 100 - (pred_classes == tgt_classes).float().mean() * 100

        return losses

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

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following matched indices."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

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
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 4. Compute encoder auxiliary loss (binary classification)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # For encoder, use binary targets (objectness)
            bin_targets = [{'labels': torch.zeros_like(t['labels']), 
                           'boxes': t['boxes']} for t in targets]
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
        use_focal=True
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


# # Example usage
# if __name__ == "__main__":
#     # Build criterion
#     criterion = build_criterion(num_classes=80, num_decoder_layers=6)
    
#     # Dummy data
#     B, N, num_tokens = 2, 900, 256
#     outputs = {
#         'pred_logits': torch.randn(B, N, num_tokens),
#         'pred_boxes': torch.rand(B, N, 4),
#         'aux_outputs': [
#             {'pred_logits': torch.randn(B, N, num_tokens), 'pred_boxes': torch.rand(B, N, 4)}
#             for _ in range(5)
#         ],
#         'enc_outputs': {
#             'pred_logits': torch.randn(B, N, num_tokens),
#             'pred_boxes': torch.rand(B, N, 4)
#         }
#     }
    
#     targets = [
#         {'labels': torch.randint(0, 80, (5,)), 'boxes': torch.rand(5, 4)},
#         {'labels': torch.randint(0, 80, (3,)), 'boxes': torch.rand(3, 4)}
#     ]
    
#     # Compute losses
#     losses = criterion(outputs, targets)
    
#     # Total loss
#     total_loss = sum(losses[k] * criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    
#     print(f"Total loss: {total_loss.item():.4f}")
#     print(f"Loss components: {list(losses.keys())}")


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
                loss_focal = torch.tensor(0.0, device=pred_masks.device)
                loss_dice = torch.tensor(0.0, device=pred_masks.device)
                loss_iou = torch.tensor(0.0, device=pred_masks.device)
        else:
            loss_focal = torch.tensor(0.0, device=pred_masks.device)
            loss_dice = torch.tensor(0.0, device=pred_masks.device)
            loss_iou = torch.tensor(0.0, device=pred_masks.device)
        
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
            inputs: Predicted masks (logits) [B, N, H, W]
            targets: Target masks (binary) [B, N, H, W]
        
        Returns:
            Focal loss value
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
        Dice loss for segmentation.
        
        Args:
            inputs: Predicted masks (logits) [B, N, H, W]
            targets: Target masks (binary) [B, N, H, W]
            smooth: Smoothing factor
        
        Returns:
            Dice loss value
        """
        inputs = inputs.sigmoid()
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice
    
    def iou_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        IoU loss for segmentation.
        
        Args:
            inputs: Predicted masks (logits) [B, N, H, W]
            targets: Target masks (binary) [B, N, H, W]
            smooth: Smoothing factor
        
        Returns:
            IoU loss value
        """
        inputs = inputs.sigmoid()
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return 1 - iou


class CombinedTeacherLoss(nn.Module):
    """
    Combined loss for multi-task teacher training.
    
    When training both detection and segmentation together.
    
    Example:
        >>> criterion = CombinedTeacherLoss(num_classes=3)
        >>> loss_dict = criterion(predictions, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        task_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            num_classes: Number of classes
            task_weights: Weights for each task (detection, segmentation)
        """
        super().__init__()
        
        self.detection_loss = DetectionLoss(num_classes)
        self.segmentation_loss = SegmentationLoss()
        
        if task_weights is None:
            task_weights = {'detection': 1.0, 'segmentation': 1.0}
        self.task_weights = task_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with detection and segmentation predictions
            targets: Dict with detection and segmentation targets
        
        Returns:
            Dict with all loss components
        """
        loss_dict = {}
        total_loss = 0
        
        # Detection loss
        if 'pred_logits' in predictions and 'labels' in targets:
            det_losses = self.detection_loss(predictions, targets)
            for k, v in det_losses.items():
                if k != 'loss':
                    loss_dict[f'det_{k}'] = v
            total_loss += self.task_weights['detection'] * det_losses['loss']
        
        # Segmentation loss
        if 'pred_masks' in predictions and 'masks' in targets:
            seg_losses = self.segmentation_loss(predictions, targets)
            for k, v in seg_losses.items():
                if k != 'loss':
                    loss_dict[f'seg_{k}'] = v
            total_loss += self.task_weights['segmentation'] * seg_losses['loss']
        
        loss_dict['loss'] = total_loss
        return loss_dict
