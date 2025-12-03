# MMDetection GroundingDINO Fine-tuning Pipeline Analysis

## Overview

After examining the actual MMDetection implementation, I've identified the **correct** way to handle GroundingDINO fine-tuning. The solution is simpler and cleaner than I initially thought.

---

## Key Finding #1: The `-inf` Padding is By Design

**File**: `mmdet/models/dense_heads/grounding_dino_head.py:82`

```python
res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))
```

GroundingDINO **intentionally** pads text tokens to `max_text_len=256` with `-inf` to mask out invalid positions during contrastive matching. **This is NOT a bug**.

---

## Key Finding #2: Filter Before Loss, Not In Loss

**File**: `mmdet/models/dense_heads/grounding_dino_head.py:539-552`

```python
# ===== this change =====
# Loss is not computed for the padded regions of the text.
assert (self.text_masks.dim() == 2)
text_masks = self.text_masks.new_zeros(
    (self.text_masks.size(0), self.max_text_len))
text_masks[:, :self.text_masks.size(1)] = self.text_masks
text_mask = (text_masks > 0).unsqueeze(1)
text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

labels = torch.masked_select(labels, text_mask)
label_weights = label_weights[...,
                              None].repeat(1, 1, text_mask.size(-1))
label_weights = torch.masked_select(label_weights, text_mask)
```

**Critical insight**: They use `torch.masked_select` to **completely remove** the padded token positions BEFORE passing to the loss function. The loss function itself is just standard Focal Loss/BCE - no special handling needed.

---

## Key Finding #3: positive_maps Creation

**File**: `mmdet/models/detectors/glip.py:100-150`

```python
def create_positive_map(tokenized, tokens_positive: list, max_num_entities: int = 256) -> Tensor:
    """construct a map such that positive_map[i,j] = True if box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_num_entities), dtype=torch.float)
    
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            # ... error handling ...
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)
```

This is **almost identical** to your `create_positive_map_from_span()`. The key: it only fills positions where `text_token_mask == True` (i.e., not `-inf`).

---

## Key Finding #4: Training Pipeline

**File**: `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py`

```python
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),           # FREEZE BACKBONE
            'language_model': dict(lr_mult=0.0)      # FREEZE BERT
        }))
```

**For fine-tuning on small datasets**: Freeze backbone and language model, only train the detection head!

---

## The Complete Solution

### 1. Problem Diagnosis

Your current implementation has two issues:

1. **You're trying to fix focal loss to handle `-inf`** (lines 278-363 in `losses.py`)
   - This is the wrong approach
   - Focal loss with `-inf` logits will always produce NaN when target=1

2. **You're not filtering before loss**
   - You need to remove `-inf` positions BEFORE computing loss
   - Not during loss computation

### 2. The Fix

**Step 1**: Modify your `GroundingDINOCriterion.loss_labels()` to filter before loss:

```python
def loss_labels(
    self,
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    num_boxes: int,
    log: bool = True
) -> Dict[str, torch.Tensor]:
    """Token-level classification loss using focal loss."""
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']  # [B, N, num_tokens]
    assert 'text_token_mask' in outputs, "Need text_token_mask to filter padding!"
    text_token_mask = outputs['text_token_mask']  # [B, num_valid_tokens]

    # Get matched predictions
    idx = self._get_src_permutation_idx(indices)
    
    # Get target token labels for matched boxes
    assert "token_labels" in targets[0], "token_labels must be provided!"
    target_token_labels = torch.cat([t["token_labels"][J] for t, (_, J) in zip(targets, indices)])

    # Extract matched predictions
    src_logits_matched = src_logits[idx]  # [num_matched, num_tokens]
    
    # ===== CRITICAL: Filter out padded tokens BEFORE loss =====
    # Pad text_token_mask to max_text_len
    B = src_logits.shape[0]
    max_text_len = src_logits.shape[-1]
    text_masks = torch.zeros((B, max_text_len), dtype=torch.bool, device=src_logits.device)
    text_masks[:, :text_token_mask.shape[1]] = text_token_mask
    
    # Create mask for matched queries
    text_mask = text_masks[idx[0]]  # [num_matched, max_text_len]
    
    # Filter using torch.masked_select
    src_logits_valid = torch.masked_select(src_logits_matched, text_mask).contiguous()
    target_labels_valid = torch.masked_select(target_token_labels, text_mask).contiguous()
    
    # Compute focal loss on VALID positions only (no -inf!)
    loss_ce = sigmoid_focal_loss(
        src_logits_valid,
        target_labels_valid,
        num_boxes,
        alpha=self.focal_alpha,
        gamma=self.focal_gamma
    ) * src_logits.shape[1]  # Scale by num_queries
    
    losses = {'loss_ce': loss_ce}

    if log:
        # Accuracy on valid positions only
        pred_binary = (src_logits_valid.sigmoid() > 0.5).float()
        losses['class_error'] = 100 - (pred_binary == target_labels_valid).float().mean() * 100

    return losses
```

**Step 2**: Remove all the debug prints (lines 278-363 in your current `losses.py`)

**Step 3**: Keep your original `sigmoid_focal_loss` from GroundingDINO utils.py - it's fine!

### 3. Data Pipeline

Your positive_map creation is **already correct**:

```python
positive_map = create_positive_map_from_span(
    tokenized, 
    token_span_per_class,
    max_text_len=outputs['pred_logits'].shape[-1]  # 256
)
```

Just make sure you also pass `text_token_mask` in outputs:

```python
outputs = {
    'pred_logits': logits,
    'pred_boxes': boxes,
    'text_token_mask': text_token_mask,  # ADD THIS!
    'aux_outputs': aux_outputs,
    ...
}
```

---

## Training Configuration

Based on MMDetection's cat dataset example:

```yaml
# Optimizer
optimizer:
  type: AdamW
  lr: 0.0001
  weight_decay: 0.0001

# Freeze pretrained parts for small datasets
param_groups:
  - modules: ['backbone']
    lr_mult: 0.0  # Freeze
  - modules: ['language_model']
    lr_mult: 0.0  # Freeze
  - modules: ['neck', 'encoder', 'decoder', 'bbox_head']
    lr_mult: 1.0  # Train

# Loss weights (from MMDetection)
loss_cls: 2.0
loss_bbox: 5.0
loss_giou: 2.0

# Focal loss params
focal_alpha: 0.25
focal_gamma: 2.0
```

---

## Summary

**The MMDetection approach**:
1. ✅ Use `-inf` padding in ContrastiveEmbed (by design)
2. ✅ Create proper positive_maps from token spans
3. ✅ **Filter out `-inf` positions using `torch.masked_select` BEFORE loss**
4. ✅ Use standard Focal Loss on the filtered tensors
5. ✅ Freeze backbone and BERT for small datasets

**What you were doing wrong**:
1. ❌ Trying to handle `-inf` inside focal loss
2. ❌ Not filtering before loss computation
3. ❌ 80+ lines of debug prints cluttering the loss function

**The fix is simple**: Extract valid positions with `torch.masked_select`, then compute loss normally.

---

## References

- MMDetection GroundingDINO Head: `mmdet/models/dense_heads/grounding_dino_head.py`
- Fine-tuning Config: `configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py`
- Usage Guide: `configs/mm_grounding_dino/usage.md`




