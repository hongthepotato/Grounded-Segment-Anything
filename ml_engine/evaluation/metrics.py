"""
Metric computation for model evaluation.

This module provides:
- DetectionMetrics: COCO-style mAP for object detection (Grounding DINO)
- SegmentationMetrics: IoU/Dice for instance segmentation (SAM)
- SimpleMetricsConverter: Convert technical metrics to rookie-friendly format

Uses torchmetrics for standardized COCO-style evaluation.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import numpy as np

# Use torchmetrics for standard COCO-style mAP computation
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)


class DetectionMetrics:
    """
    COCO-style detection metrics using torchmetrics.MeanAveragePrecision.
    
    Computes:
    - mAP@50: Mean Average Precision at IoU threshold 0.5
    - mAP@50-95: Mean AP across IoU thresholds 0.5 to 0.95
    - Per-class AP
    
    Example:
        >>> metrics = DetectionMetrics(num_classes=3, class_names=['dog', 'cat', 'car'])
        >>> metrics.update(predictions, targets)
        >>> results = metrics.compute()
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            class_metrics=True
        )

        # Per-class GT counts (for reporting)
        self.class_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.map_metric.reset()
        self.class_counts = {i: 0 for i in range(self.num_classes)}

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ):
        """
        Update with batch of predictions and targets.
        
        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
        """
        preds_formatted = []
        targets_formatted = []

        for pred, tgt in zip(predictions, targets):
            pred_dict = {
                'boxes': pred['boxes'].cpu() if torch.is_tensor(pred['boxes']) else torch.tensor(pred['boxes']),
                'scores': pred['scores'].cpu() if torch.is_tensor(pred['scores']) else torch.tensor(pred['scores']),
                'labels': pred['labels'].cpu().long() if torch.is_tensor(pred['labels']) else torch.tensor(pred['labels']).long()
            }
            preds_formatted.append(pred_dict)

            tgt_dict = {
                'boxes': tgt['boxes'].cpu() if torch.is_tensor(tgt['boxes']) else torch.tensor(tgt['boxes']),
                'labels': tgt['labels'].cpu().long() if torch.is_tensor(tgt['labels']) else torch.tensor(tgt['labels']).long()
            }
            targets_formatted.append(tgt_dict)

            # Count GT objects per class
            for label in tgt_dict['labels'].tolist():
                if 0 <= label < self.num_classes:
                    self.class_counts[label] += 1

        self.map_metric.update(preds_formatted, targets_formatted)

    def compute(self) -> Dict[str, Any]:
        """Compute all detection metrics."""
        results = self.map_metric.compute()

        mAP50 = float(results.get('map_50'))
        mAP50_95 = float(results.get('map'))

        # Per-class AP (mAP@[.5:.95] per class - the primary COCO metric)
        per_class_ap = {}
        if 'map_per_class' in results and results['map_per_class'] is not None:
            for i, ap in enumerate(results['map_per_class']):
                if i < len(self.class_names):
                    per_class_ap[self.class_names[i]] = float(ap) if not torch.isnan(ap) else 0.0

        return {
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'per_class_ap': per_class_ap,
            'per_class_counts': {self.class_names[i]: self.class_counts[i] for i in range(self.num_classes)}
        }


class SegmentationMetrics:
    """
    Segmentation metrics computation for instance segmentation.
    
    Computes:
    - mIoU: Mean Intersection over Union
    - Dice: Dice coefficient (F1 score for segmentation)
    - Per-class IoU
    
    Uses vectorized operations for efficient computation.
    
    Example:
        >>> metrics = SegmentationMetrics(num_classes=3, class_names=['dog', 'cat', 'car'])
        >>> metrics.update(predictions, targets)
        >>> results = metrics.compute()
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5
    ):
        """
        Args:
            num_classes: Number of object classes
            class_names: Optional list of class names
            iou_threshold: IoU threshold for matching predictions to GT
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.iou_threshold = iou_threshold
        
        # Accumulate IoU/Dice values per class
        self.class_ious: Dict[int, List[float]] = {i: [] for i in range(num_classes)}
        self.class_dice: Dict[int, List[float]] = {i: [] for i in range(num_classes)}
        self.class_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
        
        # Overall statistics
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
    
    def reset(self):
        """Reset accumulated metrics."""
        self.class_ious = {i: [] for i in range(self.num_classes)}
        self.class_dice = {i: [] for i in range(self.num_classes)}
        self.class_counts = {i: 0 for i in range(self.num_classes)}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
    
    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ):
        """
        Update with batch of predictions and targets.
        
        Args:
            predictions: List of dicts with 'masks', 'scores', 'labels'
            targets: List of dicts with 'masks', 'labels'
        """
        for pred, tgt in zip(predictions, targets):
            self._update_single(pred, tgt)
    
    def _update_single(self, pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
        """Update metrics for a single image."""
        pred_masks = pred['masks'].cpu().float() if torch.is_tensor(pred['masks']) else torch.tensor(pred['masks']).float()
        pred_labels = pred['labels'].cpu() if torch.is_tensor(pred['labels']) else torch.tensor(pred['labels'])
        pred_scores = pred.get('scores', torch.ones(len(pred_labels)))
        if torch.is_tensor(pred_scores):
            pred_scores = pred_scores.cpu()
        
        tgt_masks = tgt['masks'].cpu().float() if torch.is_tensor(tgt['masks']) else torch.tensor(tgt['masks']).float()
        tgt_labels = tgt['labels'].cpu() if torch.is_tensor(tgt['labels']) else torch.tensor(tgt['labels'])
        
        # Count GT per class
        for label in tgt_labels.tolist():
            if 0 <= label < self.num_classes:
                self.class_counts[label] += 1
        
        if len(pred_masks) == 0:
            self.total_fn += len(tgt_masks)
            return
        
        if len(tgt_masks) == 0:
            self.total_fp += len(pred_masks)
            return
        
        # Compute IoU matrix between all pred and gt masks (vectorized)
        # Flatten masks: [N, H, W] -> [N, H*W]
        pred_flat = pred_masks.view(len(pred_masks), -1)
        tgt_flat = tgt_masks.view(len(tgt_masks), -1)
        
        # Intersection: [N_pred, N_gt]
        intersection = torch.mm(pred_flat, tgt_flat.t())
        
        # Areas
        pred_areas = pred_flat.sum(dim=1, keepdim=True)  # [N_pred, 1]
        tgt_areas = tgt_flat.sum(dim=1, keepdim=True).t()  # [1, N_gt]
        
        # Union and IoU
        union = pred_areas + tgt_areas - intersection
        iou_matrix = intersection / (union + 1e-10)
        
        # Dice: 2*intersection / (pred + gt)
        dice_matrix = 2 * intersection / (pred_areas + tgt_areas + 1e-10)
        
        # Match predictions to GT (greedy matching by score)
        matched_gt = set()
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for pred_idx in sorted_indices:
            pred_label = pred_labels[pred_idx].item()
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx in range(len(tgt_masks)):
                if gt_idx in matched_gt:
                    continue
                if tgt_labels[gt_idx].item() != pred_label:
                    continue
                
                iou = iou_matrix[pred_idx, gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                self.total_tp += 1
                matched_gt.add(best_gt_idx)
                
                if 0 <= pred_label < self.num_classes:
                    self.class_ious[pred_label].append(best_iou)
                    self.class_dice[pred_label].append(dice_matrix[pred_idx, best_gt_idx].item())
            else:
                self.total_fp += 1
        
        self.total_fn += len(tgt_masks) - len(matched_gt)
    
    def compute(self) -> Dict[str, Any]:
        """Compute all segmentation metrics."""
        per_class_iou = {}
        per_class_dice = {}
        
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            per_class_iou[class_name] = float(np.mean(self.class_ious[class_idx])) if self.class_ious[class_idx] else 0.0
            per_class_dice[class_name] = float(np.mean(self.class_dice[class_idx])) if self.class_dice[class_idx] else 0.0
        
        # Mean across classes with GT
        valid_ious = [v for i, v in enumerate(per_class_iou.values()) if self.class_counts[i] > 0]
        valid_dice = [v for i, v in enumerate(per_class_dice.values()) if self.class_counts[i] > 0]
        
        mIoU = float(np.mean(valid_ious)) if valid_ious else 0.0
        mean_dice = float(np.mean(valid_dice)) if valid_dice else 0.0
        
        precision = self.total_tp / (self.total_tp + self.total_fp + 1e-10)
        recall = self.total_tp / (self.total_tp + self.total_fn + 1e-10)
        
        return {
            'mIoU': mIoU,
            'mean_dice': mean_dice,
            'precision': float(precision),
            'recall': float(recall),
            'per_class_iou': per_class_iou,
            'per_class_dice': per_class_dice,
            'per_class_counts': {self.class_names[i]: self.class_counts[i] for i in range(self.num_classes)}
        }


class SimpleMetricsConverter:
    """
    Convert technical metrics to rookie-friendly format.
    
    Maps complex metrics to simple, understandable values:
    - Overall Score (0-100)
    - Grade (Excellent, Very Good, Good, etc.)
    - Detection Rate
    - Accuracy Rate
    - Plain English summary
    
    Example:
        >>> converter = SimpleMetricsConverter()
        >>> simple = converter.convert_detection(technical_metrics)
        >>> print(simple['summary'])
        "Your model finds 85% of objects and is correct 92% of the time."
    """

    # Grade thresholds
    GRADE_THRESHOLDS = [
        (90, "Excellent"),
        (80, "Very Good"),
        (70, "Good"),
        (60, "Average"),
        (50, "Needs Improvement"),
        (0, "Poor")
    ]

    def convert_detection(self, technical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert detection metrics to simple format.
        
        Args:
            technical_metrics: Dict with mAP50, mAP50_95, per_class_ap, etc.
        
        Returns:
            Dict with simple metrics for rookies
        """
        # Overall score is mAP@50 scaled to 0-100
        overall_score = technical_metrics.get('mAP50') * 100

        # Grade based on overall score
        grade = self._get_grade(overall_score)

        # Generate simple summary
        summary = self._generate_detection_summary(overall_score, grade)

        # Per-class simple metrics
        per_class = []
        per_class_ap = technical_metrics.get('per_class_ap', {})
        per_class_counts = technical_metrics.get('per_class_counts', {})

        for class_name, ap in per_class_ap.items():
            class_score = ap * 100
            class_grade = self._get_grade(class_score)
            sample_count = per_class_counts.get(class_name, 0)

            class_info = {
                'class': class_name,
                'score': round(class_score, 1),
                'grade': class_grade,
                'sample_count': sample_count
            }

            # Add warning for low sample count or low performance
            if sample_count < 50:
                class_info['warning'] = 'Low sample count - results may be unreliable'
            elif class_score < 50:
                class_info['warning'] = 'Low accuracy - consider adding more training examples'

            per_class.append(class_info)

        # Sort by score descending
        per_class.sort(key=lambda x: x['score'], reverse=True)

        return {
            'overall_score': round(overall_score, 1),
            'grade': grade,
            'summary': summary,
            'per_class': per_class
        }
    
    def convert_segmentation(self, technical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert segmentation metrics to simple format.
        
        Args:
            technical_metrics: Dict with mIoU, mean_dice, etc.
        
        Returns:
            Dict with simple metrics
        """
        # Overall score is mIoU scaled to 0-100
        overall_score = technical_metrics.get('mIoU', 0) * 100
        
        # Coverage rate is recall (what % of objects were segmented)
        coverage_rate = technical_metrics.get('recall', 0) * 100
        
        # Quality rate is mean Dice
        quality_rate = technical_metrics.get('mean_dice', 0) * 100
        
        # Grade based on overall score
        grade = self._get_grade(overall_score)
        
        # Generate summary
        summary = self._generate_segmentation_summary(
            overall_score, coverage_rate, quality_rate
        )
        
        # Per-class simple metrics
        per_class = []
        per_class_iou = technical_metrics.get('per_class_iou', {})
        per_class_counts = technical_metrics.get('per_class_counts', {})
        
        for class_name, iou in per_class_iou.items():
            class_score = iou * 100
            class_grade = self._get_grade(class_score)
            sample_count = per_class_counts.get(class_name, 0)
            
            class_info = {
                'class': class_name,
                'score': round(class_score, 1),
                'grade': class_grade,
                'sample_count': sample_count
            }
            
            # Add warning for low sample count or low performance
            if sample_count < 50:
                class_info['warning'] = 'Low sample count - results may be unreliable'
            elif class_score < 50:
                class_info['warning'] = 'Low quality - consider adding more training examples'
            
            per_class.append(class_info)
        
        # Sort by score descending
        per_class.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'overall_score': round(overall_score, 1),
            'grade': grade,
            'coverage_rate': round(coverage_rate, 1),
            'quality_rate': round(quality_rate, 1),
            'summary': summary,
            'per_class': per_class
        }

    def _get_grade(self, score: float) -> str:
        """Convert score to human-readable grade."""
        for threshold, grade in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "Poor"
    
    def _generate_detection_summary(self, overall_score: float, grade: str) -> str:
        """Generate plain English summary for detection."""
        if overall_score >= 90:
            return f"Excellent! Your model scores {overall_score:.0f}/100. It detects objects very accurately."
        if overall_score >= 80:
            return f"Very good! Your model scores {overall_score:.0f}/100. Detection quality is high."
        if overall_score >= 70:
            return f"Good. Your model scores {overall_score:.0f}/100. Solid detection performance."
        if overall_score >= 50:
            return f"Your model scores {overall_score:.0f}/100. Consider more training data to improve."
        return f"Your model scores {overall_score:.0f}/100. Needs more training or data."
    
    def _generate_segmentation_summary(
        self,
        overall_score: float,
        coverage_rate: float,
        quality_rate: float
    ) -> str:
        """Generate plain English summary for segmentation."""
        parts = []
        
        # Overall assessment
        grade = self._get_grade(overall_score)
        parts.append(f"Your model scores {overall_score:.0f}/100 ({grade}).")
        
        # Coverage rate
        parts.append(f"It segments {coverage_rate:.0f}% of objects in images.")
        
        # Quality rate
        parts.append(f"The mask quality is {quality_rate:.0f}% accurate on average.")
        
        return " ".join(parts)
