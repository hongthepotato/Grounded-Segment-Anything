# Next Steps: Testing the Fixed GroundingDINO Fine-tuning

## What Was Fixed

I've implemented the **MMDetection approach** to fix your NaN loss issue. The key changes:

1. ✅ **Filter `-inf` tokens BEFORE loss** using `torch.masked_select`  
2. ✅ **Pass `text_token_mask` from model** to identify valid tokens
3. ✅ **Update Hungarian Matcher** to mask invalid tokens in cost computation
4. ✅ **Remove 80+ lines of debug prints** - replaced with clean filtering

---

## Quick Test

Run your training command to verify the fix:

```bash
cd /root/coding/platform/Grounded-Segment-Anything

# Test with your existing training script
python cli/train_teacher.py \
    --data data/your_dataset.json \
    --images data/images/ \
    --output experiments/test_fix \
    --experiment-name grounding_dino_loss_fix_test \
    --epochs 2 \
    --batch-size 2
```

**Expected output** (no more NaN!):

```
Epoch 1/2:
  loss_ce: 1.234    ← Should be a reasonable number, NOT nan
  loss_bbox: 0.567
  loss_giou: 0.432
  total_loss: 2.233 ← Should be finite
```

---

## Files Modified

1. **`ml_engine/training/losses.py`**
   - `HungarianMatcher.forward()`: Added text mask filtering in cost computation
   - `GroundingDINOCriterion.loss_labels()`: Filter padded tokens before focal loss

2. **`ml_engine/models/teacher/grounding_dino_lora.py`**
   - `GroundingDINOLoRA.forward()`: Return `text_token_mask` in outputs

3. **Documentation** (created):
   - `MMDETECTION_FINDINGS.md`: Complete analysis of MMDetection implementation
   - `LOSS_FIX_SUMMARY.md`: Summary of applied changes
   - `NEXT_STEPS.md`: This file

---

## Verification Steps

### 1. Check Loss Values

```python
# After training for 1-2 iterations, check logs:
# ✅ loss_ce should be finite (e.g., 0.5 - 3.0)
# ✅ loss_bbox should be finite (e.g., 0.1 - 1.0)
# ✅ loss_giou should be finite (e.g., 0.1 - 1.0)
# ✅ total_loss should be finite (sum of above)

# ❌ If ANY loss is nan or inf, something is wrong
```

### 2. Check Gradients

Add this debug code in your trainer (temporarily):

```python
# After loss.backward(), check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        has_nan = torch.isnan(param.grad).any()
        has_inf = torch.isinf(param.grad).any()
        if has_nan or has_inf:
            print(f"⚠️  Bad gradient in {name}: nan={has_nan}, inf={has_inf}")
```

All gradients should be finite!

### 3. Check Training Progress

```python
# After a few epochs:
# ✅ Loss should decrease (at least slightly)
# ✅ Model should be making predictions
# ✅ No memory leaks (stable GPU memory usage)
```

---

## If Issues Persist

### Issue 1: Still getting NaN loss

**Check**:
1. Is `text_token_mask` in model outputs?
   ```python
   outputs = model(images, class_names=class_names)
   print("Keys:", outputs.keys())  # Should include 'text_token_mask'
   ```

2. Are token_labels created correctly?
   ```python
   # In your trainer, check positive_map
   print(f"positive_map shape: {positive_map.shape}")  # [num_classes, 256]
   print(f"nonzero tokens: {(positive_map > 0).sum().item()}")  # Should be > 0
   ```

3. Check focal loss inputs:
   ```python
   # In losses.py, add temporary print:
   print(f"src_logits_valid shape: {src_logits_valid.shape}")
   print(f"Has -inf: {torch.isinf(src_logits_valid).any()}")  # Should be False!
   ```

### Issue 2: Loss is 0 or very small

This might indicate:
- No matches found by Hungarian matcher
- Wrong positive_map creation
- Check your `token_labels` in targets

**Debug**:
```python
# In your trainer, after creating targets:
for i, target in enumerate(targets):
    print(f"Batch {i}:")
    print(f"  labels: {target['labels']}")
    print(f"  boxes shape: {target['boxes'].shape}")
    print(f"  token_labels shape: {target['token_labels'].shape}")
    print(f"  token_labels nonzero: {(target['token_labels'] > 0).sum().item()}")
```

### Issue 3: Out of Memory

The filtering creates temporary tensors. If OOM:
```python
# Reduce batch size
batch_size: 2  → 1

# Or enable gradient checkpointing in config
gradient_checkpointing: true
```

---

## MMDetection Fine-tuning Best Practices

Based on their cat dataset example:

### Small Datasets (<1000 images):
```yaml
# Freeze pretrained parts
backbone_lr_mult: 0.0
language_model_lr_mult: 0.0

# Train only detection head
learning_rate: 0.0001
epochs: 20
batch_size: 4 (per GPU)
```

### Medium Datasets (1000-10000 images):
```yaml
# Partially train backbone
backbone_lr_mult: 0.1
language_model_lr_mult: 0.0

learning_rate: 0.0001
epochs: 12
batch_size: 2-4 (per GPU)
```

### Large Datasets (>10000 images):
```yaml
# Train everything
backbone_lr_mult: 1.0
language_model_lr_mult: 0.1  # Still keep BERT mostly frozen

learning_rate: 0.0001
epochs: 12
batch_size: 2 (per GPU)
```

---

## Performance Expectations

### Cat Dataset (144 images, 1 class)
- Zero-shot mAP: 88.1
- After 20 epochs fine-tuning: 90.1
- Training time: ~30 min (1x 3090Ti)

### Your Dataset
Expect similar improvements if:
- ✅ Dataset quality is good
- ✅ Annotations are accurate
- ✅ Class names are clear and specific
- ✅ Images are well-lit and diverse

---

## Useful Commands

### Training with Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 python cli/train_teacher.py \
    --data data/train.json \
    --images data/images/ \
    --output experiments/run1 \
    --gpus 2
```

### Resume from Checkpoint
```bash
python cli/train_teacher.py \
    --data data/train.json \
    --images data/images/ \
    --output experiments/run1 \
    --resume experiments/run1/checkpoints/last.pth
```

### Inference Only
```bash
python cli/test_teacher.py \
    --model experiments/run1/teachers/grounding_dino_lora \
    --images data/test/ \
    --output results/
```

---

## Key References

### MMDetection Files to Study:
1. `mmdetection/mmdet/models/dense_heads/grounding_dino_head.py`
   - Lines 539-552: The filtering approach
   - Lines 193-196: How positive_maps are used

2. `mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py`
   - Complete fine-tuning config example

3. `mmdetection/configs/mm_grounding_dino/usage.md`
   - Comprehensive usage guide

### GroundingDINO Files:
1. `GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py`
   - Original model implementation
   - Line 267: Where text_token_mask is created

2. `GroundingDINO/groundingdino/models/GroundingDINO/utils.py`
   - Line 82: ContrastiveEmbed applies -inf masking
   - Line 139: Original sigmoid_focal_loss (unchanged)

---

## Summary

**The fix is simple**: Extract valid token positions with `torch.masked_select`, then compute loss normally. No NaN possible!

**What you should see**:
- ✅ Finite losses from iteration 1
- ✅ Gradients flowing normally
- ✅ Training progress (loss decreasing)
- ✅ Reasonable mAP after fine-tuning

**If you still have issues**, share:
1. The loss values you're seeing
2. Your dataset size and class count
3. The exact error message

Good luck with your fine-tuning! 🎯




