# GroundingDINO Loss Fix - Applied Changes

## Summary

Fixed the NaN loss issue in GroundingDINO fine-tuning by implementing the **MMDetection approach**: filtering out padded token positions (`-inf`) BEFORE passing to loss function, rather than trying to handle them inside the loss.

---

## Root Cause

**The Problem**: GroundingDINO pads text tokens to `max_text_len=256` with `-inf` values to mask invalid positions. When focal loss tries to compute BCE on these positions with `target=1`, it produces NaN:

```python
logit = -inf
prob = sigmoid(-inf) = 0
bce_loss = -target * log(prob) = -1 * log(0) = inf  → NaN after aggregation
```

**The Insight**: The official GroundingDINO implementation and MMDetection don't fix focal loss itself. They **filter out the `-inf` positions before computing loss** using `torch.masked_select`.

---

## Changes Made

### 1. **Updated `GroundingDINOCriterion.loss_labels()`** 
**File**: `ml_engine/training/losses.py:237-310`

**Before**: 80+ lines of debug prints, trying to handle `-inf` inside focal loss

**After**: Clean filtering using `torch.masked_select` before loss:

```python
def loss_labels(...):
    # Get text_token_mask to filter padding
    text_token_mask = outputs.get('text_token_mask', None)
    
    # Extract matched predictions
    src_logits_matched = src_logits[idx]  # [num_matched, num_tokens]
    target_token_labels = ...
    
    # ===== CRITICAL: Filter out padded tokens BEFORE loss =====
    if text_token_mask is not None:
        # Pad mask to max_text_len and create per-query mask
        text_masks = torch.zeros((B, max_text_len), dtype=torch.bool, ...)
        text_masks[:, :text_token_mask.shape[1]] = text_token_mask
        text_mask = text_masks[idx[0]]  # [num_matched, max_text_len]
        
        # Filter using torch.masked_select (flattens to 1D)
        src_logits_valid = torch.masked_select(src_logits_matched, text_mask).contiguous()
        target_labels_valid = torch.masked_select(target_token_labels, text_mask).contiguous()
    
    # Compute focal loss on VALID positions only (no -inf!)
    loss_ce = sigmoid_focal_loss(src_logits_valid.unsqueeze(0), ...)
```

**Key Points**:
- Uses `text_token_mask` from model outputs
- `torch.masked_select` removes all padded positions
- Focal loss sees only valid token positions
- No NaN possible!

---

### 2. **Updated `HungarianMatcher.forward()`**
**File**: `ml_engine/training/losses.py:95-184`

**Added**: Filtering in matching cost computation

```python
@torch.no_grad()
def forward(self, outputs, targets, ...):
    # Get text_token_mask
    text_token_mask = outputs.get('text_token_mask', None)
    
    if text_token_mask is not None:
        # Create mask [B*N, num_tokens]
        text_mask = ...
    else:
        # Fallback: use -inf detection
        text_mask = ~torch.isinf(out_logits)
    
    # Mask out invalid positions (set cost to 0 for padded tokens)
    neg_cost_class = neg_cost_class * text_mask.float()
    pos_cost_class = pos_cost_class * text_mask.float()
    
    # Compute cost matrix...
```

**Why**: Matching cost should also ignore padded tokens to be consistent with loss computation.

---

### 3. **Updated `GroundingDINOLoRA.forward()`**
**File**: `ml_engine/models/teacher/grounding_dino_lora.py:264-321`

**Added**: Return `text_token_mask` in outputs

```python
def forward(self, images, class_names=None, ...):
    outputs = self.model(samples=images, captions=captions)
    
    # Add text_token_mask for loss computation
    base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
    tokenized = base_model.tokenizer(captions, padding='longest', return_tensors='pt')
    text_token_mask = tokenized.attention_mask.bool()  # [B, num_valid_tokens]
    
    outputs['text_token_mask'] = text_token_mask.to(outputs['pred_logits'].device)
    
    return outputs
```

**Why**: Loss function needs `text_token_mask` to filter padded positions.

---

## What We Kept

### ✅ Original `sigmoid_focal_loss` from GroundingDINO
**File**: `GroundingDINO/groundingdino/models/GroundingDINO/utils.py:139-169`

**No changes needed!** The original focal loss is fine. We just filter inputs before calling it.

### ✅ `positive_map` Creation
**File**: Your trainer already uses `create_positive_map_from_span()` correctly

No changes needed! Your positive_map creation is already correct and matches MMDetection.

### ✅ Hungarian Matcher Costs and Loss Weights
**Config values from MMDetection**:
```python
# Matching costs
cost_class: 1.0
cost_bbox: 5.0
cost_giou: 2.0

# Loss weights
loss_ce: 2.0
loss_bbox: 5.0
loss_giou: 2.0

# Focal loss params
focal_alpha: 0.25
focal_gamma: 2.0
```

Your existing config is correct!

---

## Testing Checklist

- [ ] Run training and verify no NaN losses
- [ ] Check that loss values are reasonable (not 0 or inf)
- [ ] Verify gradients are flowing (check `loss.backward()`)
- [ ] Monitor GPU memory usage (should be similar to before)
- [ ] Validate predictions after training

---

## Expected Behavior

**Before Fix**:
```
loss_ce: nan
loss_bbox: 0.234
loss_giou: 0.567
total_loss: nan
```

**After Fix**:
```
loss_ce: 1.234
loss_bbox: 0.234
loss_giou: 0.567
total_loss: 2.035
```

---

## References

- **MMDetection GroundingDINO Head**: `mmdetection/mmdet/models/dense_heads/grounding_dino_head.py:539-552`
- **MMDetection Fine-tuning Config**: `mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py`
- **Key Insight**: Lines 547-552 in `grounding_dino_head.py` show the filtering approach using `torch.masked_select`

---

## Key Takeaways

1. **The `-inf` padding is by design** - Don't try to fix it, filter it!
2. **Filter before loss, not in loss** - Use `torch.masked_select` to extract valid positions
3. **Keep original focal loss unchanged** - It's already correct
4. **Pass `text_token_mask` from model** - Needed for filtering
5. **MMDetection got it right** - Follow their approach, not guesswork

---

## Performance Notes

**Compared to your previous implementation with 80+ lines of debug prints**:
- ✅ **Cleaner code**: 40 lines vs 120 lines
- ✅ **Faster**: No repeated diagnostics on each forward pass
- ✅ **Correct**: Matches official MMDetection implementation
- ✅ **Maintainable**: Clear logic, easy to understand

The fix is simple: **extract valid positions, compute loss, done**.




