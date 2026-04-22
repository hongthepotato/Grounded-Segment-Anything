"""
Grounding DINO training utilities.

This module provides utilities for:
- Token mapping: Building positive maps for token-level classification
- Target building: Creating DETR-format targets for Hungarian matching
"""

import logging
from typing import Dict, List

import torch
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span

logger = logging.getLogger(__name__)


def build_positive_map(
    tokenizer,
    class_names: List[str],
    max_text_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Build token-to-class positive map using official Grounding DINO utilities.
    
    This creates a mapping from class indices to BERT token positions,
    which is used for token-level contrastive classification.
    
    Args:
        tokenizer: BERT tokenizer from the model
        class_names: List of class names
        max_text_len: Maximum text length (typically 256)
        device: Device to place the tensor on
    
    Returns:
        Tensor of shape [num_classes, max_text_len] with 1.0 at relevant token positions
    """
    # Build caption and character-level token spans using official utility
    # This formats the caption exactly as Grounding DINO expects: "class1 . class2 . class3"
    caption, cat2tokenspan = build_captions_and_token_span(
        class_names,
        force_lowercase=False
    )
    
    # Tokenize using the model's tokenizer
    tokenized = tokenizer(
        caption,
        padding="longest",
        return_tensors="pt"
    ).to(device)
    
    # Build class_id -> token_span mapping
    class_id_to_name = {i: name for i, name in enumerate(class_names)}
    token_span_per_class = []
    
    for class_id in range(len(class_names)):
        class_name = class_id_to_name[class_id]
        if class_name not in cat2tokenspan:
            raise ValueError(
                f"Class '{class_name}' not found in cat2tokenspan!\n"
                f"Available classes: {list(cat2tokenspan.keys())}\n"
                f"This indicates a mismatch between class_names and caption tokenization.\n"
                f"Check if class names contain special characters or case mismatches."
            )
        token_span_per_class.append(cat2tokenspan[class_name])
    
    # Create positive map using official utility
    # This converts character spans to actual BERT token positions
    positive_map = create_positive_map_from_span(
        tokenized,
        token_span_per_class,
        max_text_len=max_text_len
    ).to(device)
    
    return positive_map


def build_detr_targets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    positive_map: torch.Tensor,
    category_id_to_index: Dict[int, int],
    device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """
    Build DETR-format targets for Hungarian matching loss.
    
    Args:
        boxes: Tensor of shape [B, max_objs, 4] in normalized [cx, cy, w, h] format
        labels: Tensor of shape [B, max_objs] with class IDs (-1 for padding)
        positive_map: Tensor of shape [num_classes, max_text_len]
        category_id_to_index: Mapping from category_id to 0-based index
        device: Device to place tensors on
    
    Returns:
        List of target dicts (one per batch element), each containing:
        - 'labels': [num_valid_objs] class IDs
        - 'boxes': [num_valid_objs, 4] in normalized [cx, cy, w, h]
        - 'token_labels': [num_valid_objs, max_text_len]
    """
    batch_size = labels.shape[0]
    targets = []
    
    for b in range(batch_size):
        # Get valid objects for this batch element
        valid_mask = labels[b] != -1
        valid_labels = labels[b][valid_mask]
        valid_boxes = boxes[b][valid_mask]
        
        # Sanity check boxes (should be normalized [0, 1])
        if len(valid_boxes) > 0:
            if (valid_boxes < 0).any() or (valid_boxes > 1).any():
                logger.warning(
                    "Batch %d: boxes not normalized! Range: [%.3f, %.3f]",
                    b, valid_boxes.min(), valid_boxes.max()
                )
        
        # Create token labels for each valid object using the positive map
        token_labels = torch.zeros(
            len(valid_labels),
            positive_map.shape[1],
            dtype=torch.float32,
            device=device
        )
        
        # Map category IDs to indices and assign token labels
        for obj_idx, class_id in enumerate(valid_labels):
            cat_id = int(class_id.item())
            if cat_id not in category_id_to_index:
                raise ValueError(
                    f"Unknown category_id {cat_id} not in category_id_to_index mapping!\n"
                    f"Available category_ids: {list(category_id_to_index.keys())}\n"
                    f"This indicates corrupted annotations or dataset_info mismatch.\n"
                    f"Batch {b}, object {obj_idx}."
                )
            class_idx = category_id_to_index[cat_id]
            token_labels[obj_idx] = positive_map[class_idx]
        
        targets.append({
            'labels': valid_labels,
            'boxes': valid_boxes,
            'token_labels': token_labels,
        })
    
    return targets

