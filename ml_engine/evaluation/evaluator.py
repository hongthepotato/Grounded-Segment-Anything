"""
Model Evaluator for running evaluation on test sets.

This module provides:
- ModelEvaluator: Main evaluation orchestrator
- Supports both Grounding DINO (detection) and SAM (segmentation)
- Handles inference, metric computation, and result aggregation
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add GroundingDINO to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GroundingDINO"))
from groundingdino.util.box_ops import box_iou, box_cxcywh_to_xyxy

from ml_engine.evaluation.metrics import (
    DetectionMetrics,
    SegmentationMetrics,
    SimpleMetricsConverter
)

logger = logging.getLogger(__name__)


def _cxcywh_norm_to_xyxy_pixel(boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Convert boxes from normalized cxcywh to pixel xyxy."""
    if len(boxes) == 0:
        return torch.zeros((0, 4), device=boxes.device)
    # cxcywh to xyxy (still normalized)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    # Scale to pixel coordinates
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    return boxes_xyxy * scale


class ModelEvaluator:
    """
    Main evaluator for teacher models.
    
    Handles:
    - Running inference on test set
    - Computing technical metrics (mAP, IoU, etc.)
    - Converting to simple metrics
    - Collecting success/failure samples for visualization
    
    Example:
        >>> evaluator = ModelEvaluator(device='cuda')
        >>> 
        >>> # Evaluate detection model
        >>> results = evaluator.evaluate_detection(
        ...     model=grounding_dino,
        ...     dataloader=test_loader,
        ...     class_names=['dog', 'cat', 'car'],
        ...     dataset_info=dataset_info
        ... )
        >>> 
        >>> # Evaluate segmentation model
        >>> results = evaluator.evaluate_segmentation(
        ...     model=sam,
        ...     dataloader=test_loader,
        ...     class_names=['dog', 'cat', 'car'],
        ...     dataset_info=dataset_info
        ... )
    """

    def __init__(
        self,
        device: str = 'cuda',
        confidence_threshold: float = 0.3,
        max_samples_for_viz: int = 20
    ):
        """
        Args:
            device: Device to run inference on
            confidence_threshold: Threshold for filtering predictions
            max_samples_for_viz: Maximum samples to collect for visualization
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.max_samples_for_viz = max_samples_for_viz
        self.simple_converter = SimpleMetricsConverter()

    @torch.no_grad()
    def evaluate_detection(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        class_names: List[str],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate detection model (Grounding DINO).
        
        Args:
            model: Grounding DINO model
            dataloader: Test dataloader
            class_names: List of class names
            dataset_info: Dataset information dict
        
        Returns:
            Dictionary with:
            - technical_metrics: Full COCO-style metrics
            - simple_metrics: Rookie-friendly metrics
            - samples: Success/failure samples for visualization
        """
        model.eval()
        model.to(self.device)

        num_classes = len(class_names)
        metrics = DetectionMetrics(num_classes=num_classes, class_names=class_names)

        # Samples for visualization
        success_samples = []
        failure_samples = []

        logger.info("Running detection evaluation on %d batches...", len(dataloader))

        for batch in tqdm(dataloader, desc="Evaluating detection"):
            batch_preds, batch_targets = self._run_detection_inference(
                model, batch, class_names, dataset_info
            )

            # Update metrics
            metrics.update(batch_preds, batch_targets)

            # Collect samples for visualization
            self._collect_samples(
                batch, batch_preds, batch_targets,
                success_samples, failure_samples,
                model_type='detection'
            )

        # Compute final metrics
        technical_metrics = metrics.compute()
        simple_metrics = self.simple_converter.convert_detection(technical_metrics)

        logger.info("Detection evaluation complete")
        logger.info("  mAP@50: %.3f", technical_metrics['mAP50'])
        logger.info("  mAP@50-95: %.3f", technical_metrics['mAP50_95'])

        return {
            'model_type': 'detection',
            'technical_metrics': technical_metrics,
            'simple_metrics': simple_metrics,
            'samples': {
                'success': success_samples[:self.max_samples_for_viz // 2],
                'failure': failure_samples[:self.max_samples_for_viz // 2]
            }
        }

    @torch.no_grad()
    def evaluate_segmentation(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        class_names: List[str],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate segmentation model (SAM).
        
        Args:
            model: SAM model
            dataloader: Test dataloader
            class_names: List of class names
            dataset_info: Dataset information dict
        
        Returns:
            Dictionary with:
            - technical_metrics: Full segmentation metrics
            - simple_metrics: Rookie-friendly metrics
            - samples: Success/failure samples for visualization
        """
        model.eval()
        model.to(self.device)
        
        num_classes = len(class_names)
        metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)
        
        # Samples for visualization
        success_samples = []
        failure_samples = []
        
        logger.info("Running segmentation evaluation on %d batches...", len(dataloader))
        
        for batch in tqdm(dataloader, desc="Evaluating segmentation"):
            batch_preds, batch_targets = self._run_segmentation_inference(
                model, batch, class_names, dataset_info
            )
            
            # Update metrics
            metrics.update(batch_preds, batch_targets)
            
            # Collect samples for visualization
            self._collect_samples(
                batch, batch_preds, batch_targets,
                success_samples, failure_samples,
                model_type='segmentation'
            )
        
        # Compute final metrics
        technical_metrics = metrics.compute()
        simple_metrics = self.simple_converter.convert_segmentation(technical_metrics)
        
        logger.info("Segmentation evaluation complete")
        logger.info("  mIoU: %.3f", technical_metrics['mIoU'])
        logger.info("  Mean Dice: %.3f", technical_metrics['mean_dice'])
        logger.info("  Precision: %.3f", technical_metrics['precision'])
        logger.info("  Recall: %.3f", technical_metrics['recall'])
        
        return {
            'model_type': 'segmentation',
            'technical_metrics': technical_metrics,
            'simple_metrics': simple_metrics,
            'samples': {
                'success': success_samples[:self.max_samples_for_viz // 2],
                'failure': failure_samples[:self.max_samples_for_viz // 2]
            }
        }
    
    def _run_detection_inference(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Any],
        class_names: List[str],
        dataset_info: Dict[str, Any]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run detection inference on a batch.
        
        Returns:
            Tuple of (predictions_list, targets_list)
        """
        dino_data = batch['preprocessed']['grounding_dino']
        images = dino_data['images'].to(self.device)
        gt_boxes = dino_data['boxes'].to(self.device)
        gt_labels = dino_data['labels'].to(self.device)
        metadata_list = dino_data['metadata']
        
        batch_size = gt_labels.shape[0]
        cat_id_to_idx = dataset_info['category_id_to_index']
        
        # Use model's predict() method - handles all token-to-class conversion internally
        raw_predictions = model.predict(images, class_names, self.confidence_threshold)
        
        # Convert to xyxy pixel coordinates and format for metrics
        predictions_list = []
        targets_list = []
        
        for b in range(batch_size):
            img_h, img_w = metadata_list[b]['final_size']
            pred = raw_predictions[b]
            
            # Convert prediction boxes to xyxy pixel
            predictions_list.append({
                'boxes': _cxcywh_norm_to_xyxy_pixel(pred['boxes'], img_h, img_w),
                'scores': pred['scores'],
                'labels': pred['labels']
            })
            
            # Format targets
            valid_mask = gt_labels[b] != -1
            valid_boxes = gt_boxes[b][valid_mask]
            valid_labels_raw = gt_labels[b][valid_mask]
            
            targets_list.append({
                'boxes': _cxcywh_norm_to_xyxy_pixel(valid_boxes, img_h, img_w),
                'labels': torch.tensor(
                    [cat_id_to_idx[int(cat_id.item())] for cat_id in valid_labels_raw],
                    device=self.device
                )
            })
        
        return predictions_list, targets_list
    
    def _run_segmentation_inference(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Any],
        _class_names: List[str],  # Reserved for future use
        dataset_info: Dict[str, Any]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run segmentation inference on a batch.
        
        Returns:
            Tuple of (predictions_list, targets_list)
        """
        # Get preprocessed data
        sam_data = batch['preprocessed']['sam']
        images = sam_data['images'].to(self.device)
        gt_boxes = sam_data['boxes'].to(self.device)
        gt_masks = sam_data['masks'].to(self.device)
        gt_labels = sam_data['labels'].to(self.device)
        
        batch_size = gt_labels.shape[0]
        
        cat_id_to_idx = dataset_info.get('category_id_to_index', {})
        
        predictions_list = []
        targets_list = []
        
        # Forward pass with box prompts
        # Note: SAM expects boxes in xyxy format
        outputs = model(images, box_prompts=gt_boxes)
        
        pred_masks = outputs['pred_masks']  # [B, N, H, W]
        iou_predictions = outputs.get('iou_predictions', torch.ones_like(gt_labels, dtype=torch.float))
        
        for b in range(batch_size):
            valid_mask = gt_labels[b] != -1
            
            # Process predictions - only for valid GT boxes
            if valid_mask.sum() > 0:
                valid_pred_masks = pred_masks[b][valid_mask]
                valid_iou_preds = iou_predictions[b][valid_mask] if len(iou_predictions.shape) > 1 else iou_predictions[valid_mask]
                valid_labels_raw = gt_labels[b][valid_mask]
                
                # Convert category_id to class index
                valid_label_indices = []
                for cat_id in valid_labels_raw:
                    cat_id_int = int(cat_id.item())
                    if cat_id_int not in cat_id_to_idx:
                        raise ValueError(
                            f"Unknown category_id {cat_id_int} in segmentation evaluation!\n"
                            f"Available category_ids: {list(cat_id_to_idx.keys())}\n"
                            f"This indicates corrupted annotations or dataset_info mismatch."
                        )
                    valid_label_indices.append(cat_id_to_idx[cat_id_int])
                valid_labels = torch.tensor(valid_label_indices, device=self.device)
                
                # Binarize masks
                binary_masks = (valid_pred_masks > 0).float()
            else:
                binary_masks = torch.zeros((0, pred_masks.shape[-2], pred_masks.shape[-1]), device=self.device)
                valid_iou_preds = torch.zeros((0,), device=self.device)
                valid_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
            
            predictions_list.append({
                'masks': binary_masks,
                'scores': valid_iou_preds,
                'labels': valid_labels
            })
            
            # Process targets
            if valid_mask.sum() > 0:
                valid_gt_masks = gt_masks[b][valid_mask]
                valid_labels_raw = gt_labels[b][valid_mask]
                valid_label_indices = []
                for cat_id in valid_labels_raw:
                    cat_id_int = int(cat_id.item())
                    if cat_id_int not in cat_id_to_idx:
                        raise ValueError(
                            f"Unknown category_id {cat_id_int} in segmentation evaluation!\n"
                            f"Available category_ids: {list(cat_id_to_idx.keys())}\n"
                            f"This indicates corrupted annotations or dataset_info mismatch."
                        )
                    valid_label_indices.append(cat_id_to_idx[cat_id_int])
                valid_labels = torch.tensor(valid_label_indices, device=self.device)
            else:
                valid_gt_masks = torch.zeros((0, gt_masks.shape[-2], gt_masks.shape[-1]), device=self.device)
                valid_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
            
            targets_list.append({
                'masks': valid_gt_masks,
                'labels': valid_labels
            })
        
        return predictions_list, targets_list
    
    def _collect_samples(
        self,
        batch: Dict[str, Any],
        predictions: List[Dict],
        targets: List[Dict],
        success_samples: List[Dict],
        failure_samples: List[Dict],
        model_type: str
    ):
        """
        Collect success and failure samples for visualization.
        
        Success: High confidence correct predictions
        Failure: Missed objects or wrong predictions
        """
        file_names = batch.get('file_names', [])
        
        for b, (pred, tgt) in enumerate(zip(predictions, targets)):
            if len(success_samples) >= self.max_samples_for_viz and len(failure_samples) >= self.max_samples_for_viz:
                return
            
            file_name = file_names[b] if b < len(file_names) else f'sample_{b}'
            
            sample_info = {
                'file_name': file_name,
                'predictions': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in pred.items()},
                'targets': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tgt.items()}
            }
            
            # Determine if success or failure
            if model_type == 'detection':
                is_success = self._is_detection_success(pred, tgt)
            else:
                is_success = self._is_segmentation_success(pred, tgt)
            
            if is_success and len(success_samples) < self.max_samples_for_viz:
                sample_info['status'] = 'success'
                success_samples.append(sample_info)
            elif not is_success and len(failure_samples) < self.max_samples_for_viz:
                sample_info['status'] = 'failure'
                failure_samples.append(sample_info)
    
    def _is_detection_success(self, pred: Dict, tgt: Dict) -> bool:
        """Check if detection was successful (>50% of GT found with correct class)."""
        if len(tgt['boxes']) == 0:
            return len(pred['boxes']) == 0
        
        if len(pred['boxes']) == 0:
            return False
        
        # Convert to tensors for box_iou
        pred_boxes = torch.tensor(pred['boxes']) if not torch.is_tensor(pred['boxes']) else pred['boxes']
        tgt_boxes = torch.tensor(tgt['boxes']) if not torch.is_tensor(tgt['boxes']) else tgt['boxes']
        
        if pred_boxes.device != tgt_boxes.device:
            tgt_boxes = tgt_boxes.to(pred_boxes.device)
        
        # Compute IoU matrix using groundingdino's box_iou
        iou_matrix, _ = box_iou(pred_boxes.float(), tgt_boxes.float())
        
        # Match predictions to GT
        matches = 0
        matched_gt = set()
        
        for pred_idx in range(len(pred['boxes'])):
            pred_label = pred['labels'][pred_idx].item() if torch.is_tensor(pred['labels'][pred_idx]) else pred['labels'][pred_idx]
            
            for gt_idx in range(len(tgt['boxes'])):
                if gt_idx in matched_gt:
                    continue
                
                gt_label = tgt['labels'][gt_idx].item() if torch.is_tensor(tgt['labels'][gt_idx]) else tgt['labels'][gt_idx]
                
                if pred_label != gt_label:
                    continue
                
                if iou_matrix[pred_idx, gt_idx] >= 0.5:
                    matches += 1
                    matched_gt.add(gt_idx)
                    break
        
        recall = matches / len(tgt['boxes'])
        return recall >= 0.5
    
    def _is_segmentation_success(self, pred: Dict, tgt: Dict) -> bool:
        """Check if segmentation was successful (mean IoU > 0.5)."""
        if len(tgt['masks']) == 0:
            return len(pred['masks']) == 0
        
        if len(pred['masks']) == 0:
            return False
        
        # Compute mean IoU between matched masks (vectorized)
        pred_masks = torch.tensor(pred['masks']).float() if not torch.is_tensor(pred['masks']) else pred['masks'].float()
        tgt_masks = torch.tensor(tgt['masks']).float() if not torch.is_tensor(tgt['masks']) else tgt['masks'].float()
        
        # Flatten and compute IoU
        n_pred, n_tgt = len(pred_masks), len(tgt_masks)
        pred_flat = pred_masks.view(n_pred, -1)
        tgt_flat = tgt_masks.view(n_tgt, -1)
        
        # For simplicity, compute IoU for corresponding pairs (assumes same order)
        n_pairs = min(n_pred, n_tgt)
        total_iou = 0.0
        
        for i in range(n_pairs):
            intersection = (pred_flat[i] * tgt_flat[i]).sum()
            union = pred_flat[i].sum() + tgt_flat[i].sum() - intersection
            iou = intersection / (union + 1e-10)
            total_iou += float(iou)
        
        mean_iou = total_iou / n_pairs if n_pairs > 0 else 0
        return mean_iou >= 0.5
