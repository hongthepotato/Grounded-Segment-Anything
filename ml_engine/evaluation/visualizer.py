"""
Simple prediction visualizer for debugging.

Saves side-by-side comparisons of predictions vs ground truth.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """
    Simple visualizer for debugging predictions during training.
    
    Saves prediction images with bounding boxes overlaid.
    Green = Ground Truth, Red = Prediction.
    
    Example:
        >>> visualizer = PredictionVisualizer(output_dir='experiments/test/predictions')
        >>> visualizer.save_batch(
        ...     epoch=5,
        ...     images=batch_images,
        ...     predictions={'boxes': pred_boxes, 'labels': pred_labels},
        ...     targets={'boxes': gt_boxes, 'labels': gt_labels},
        ...     class_names=['person', 'car']
        ... )
    """
    
    def __init__(
        self, 
        output_dir: str,
        max_samples_per_epoch: int = 8,
        enabled: bool = True
    ):
        """
        Args:
            output_dir: Directory to save visualizations
            max_samples_per_epoch: Maximum samples to save per epoch
            enabled: Whether visualization is enabled
        """
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples_per_epoch
        self.enabled = enabled
        self._samples_this_epoch = 0
        self._current_epoch = -1
        
        # Lazy import to avoid dependency issues
        self._plt = None
        self._Image = None
        
    def _lazy_import(self):
        """Import visualization libraries only when needed."""
        if self._plt is None:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from PIL import Image
                self._plt = plt
                self._patches = patches
                self._Image = Image
            except ImportError as e:
                logger.warning(f"Visualization disabled: {e}")
                self.enabled = False
                return False
        return True
    
    def save_batch(
        self,
        epoch: int,
        images: List[np.ndarray],
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        class_names: List[str],
        image_ids: Optional[List[int]] = None
    ) -> int:
        """
        Save visualizations for a batch of predictions.
        
        Args:
            epoch: Current epoch number
            images: List of images as numpy arrays [H, W, 3] (RGB, 0-255)
            predictions: Dict with 'boxes' [N, 4] in xyxy format, 'labels' [N]
            targets: Dict with 'boxes' [N, 4] in xyxy format, 'labels' [N]
            class_names: List of class names
            image_ids: Optional list of image IDs for naming
            
        Returns:
            Number of images saved
        """
        if not self.enabled:
            return 0
            
        if not self._lazy_import():
            return 0
        
        # Reset counter for new epoch
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            self._samples_this_epoch = 0
        
        # Check if we've saved enough this epoch
        if self._samples_this_epoch >= self.max_samples:
            return 0
        
        # Create epoch directory
        epoch_dir = self.output_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        batch_size = len(images)
        
        for i in range(batch_size):
            if self._samples_this_epoch >= self.max_samples:
                break
                
            img = images[i]
            pred_boxes = predictions.get('boxes', [[]])[i] if predictions.get('boxes') is not None else []
            pred_labels = predictions.get('labels', [[]])[i] if predictions.get('labels') is not None else []
            gt_boxes = targets.get('boxes', [[]])[i] if targets.get('boxes') is not None else []
            gt_labels = targets.get('labels', [[]])[i] if targets.get('labels') is not None else []
            
            # Generate filename
            img_id = image_ids[i] if image_ids else self._samples_this_epoch
            save_path = epoch_dir / f'sample_{img_id:04d}.png'
            
            self._save_single(
                image=img,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                save_path=save_path
            )
            
            self._samples_this_epoch += 1
            saved_count += 1
        
        if saved_count > 0:
            logger.debug(f"Saved {saved_count} prediction visualizations to {epoch_dir}")
        
        return saved_count
    
    def _save_single(
        self,
        image: np.ndarray,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        class_names: List[str],
        save_path: Path
    ):
        """Save a single visualization."""
        plt = self._plt
        patches = self._patches
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left: Ground Truth
        axes[0].imshow(image)
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        self._draw_boxes(axes[0], gt_boxes, gt_labels, class_names, color='green')
        axes[0].axis('off')
        
        # Right: Predictions
        axes[1].imshow(image)
        axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
        self._draw_boxes(axes[1], pred_boxes, pred_labels, class_names, color='red')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def _draw_boxes(
        self,
        ax,
        boxes: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        color: str = 'green'
    ):
        """Draw bounding boxes on an axis."""
        patches = self._patches
        
        if len(boxes) == 0:
            return
        
        boxes = np.atleast_2d(boxes)
        labels = np.atleast_1d(labels)
        
        for box, label in zip(boxes, labels):
            if len(box) < 4:
                continue
                
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_idx = int(label) if not np.isnan(label) and label >= 0 else -1
            if 0 <= label_idx < len(class_names):
                label_text = class_names[label_idx]
            else:
                label_text = f'cls_{label_idx}'
            
            ax.text(
                x1, y1 - 5,
                label_text,
                fontsize=8,
                color='white',
                backgroundcolor=color,
                fontweight='bold'
            )
    
    def cleanup_old_epochs(self, keep_last_n: int = 5):
        """Remove old epoch directories to save disk space."""
        if not self.output_dir.exists():
            return
            
        epoch_dirs = sorted(self.output_dir.glob('epoch_*'))
        
        if len(epoch_dirs) > keep_last_n:
            for old_dir in epoch_dirs[:-keep_last_n]:
                import shutil
                shutil.rmtree(old_dir)
                logger.debug(f"Removed old visualization directory: {old_dir}")


