# Practical Migration Guide: Custom → MMDetection Backend

## Quick Decision Matrix

**Should you migrate?**

| Your Priority | Use MMDet? | Reason |
|--------------|------------|--------|
| **Get working fine-tuning ASAP** | ✅ **YES** | Their code works, yours had bugs |
| **Minimal code maintenance** | ✅ **YES** | 2000 lines → 330 lines |
| **Distributed training** | ✅ **YES** | Built-in multi-GPU support |
| **Proven hyperparameters** | ✅ **YES** | Community-validated configs |
| **Keep 19MB LoRA checkpoints** | ❌ **NO** | MMDet doesn't support PEFT natively |
| **Extreme customization** | ❌ **NO** | Less flexible than custom code |
| **Learning budget available** | ✅ **YES** | 1-2 weeks to learn MMDet |

**If 4+ are YES**: Migrate to MMDetection  
**If 3+ are NO**: Keep custom implementation

---

## Migration Plan (Incremental, Low Risk)

### Phase 0: Validation (2 hours)

**Test MMDet with their example before committing:**

```bash
cd /root/coding/platform/Grounded-Segment-Anything/mmdetection

# Download cat dataset
wget https://download.openmmlab.com/mmyolo/data/cat_dataset.zip
unzip cat_dataset.zip -d data/cat/

# Test fine-tuning
python tools/train.py \
    configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py \
    --work-dir test_mmdet

# Expected result after 20 epochs:
# mAP50: ~90.1 (as documented in usage.md)
```

**If this works**: ✅ Proceed with migration  
**If this fails**: ❌ Debug or stick with custom

---

### Phase 1: Parallel Implementation (1 week)

**Keep both implementations running side-by-side for comparison.**

```python
# cli/train_teacher.py (updated)

def main(args):
    if args.use_mmdet:
        # New: MMDet backend
        train_with_mmdet(args)
    else:
        # Existing: Custom implementation
        train_with_custom(args)

def train_with_mmdet(args):
    """NEW: Train using MMDetection backend."""
    # Step 1: Your data inspection (keep this!)
    data_manager = DataManager(
        data_path=args.data,
        image_dir=args.images,
        split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
    )
    dataset_info = data_manager.get_dataset_info()
    
    # Step 2: Generate MMDet config
    from ml_engine.mmdet_integration import generate_config
    config_path = generate_config(dataset_info, args.output)
    
    # Step 3: Train with MMDet
    import subprocess
    subprocess.run([
        'python', 'mmdetection/tools/train.py',
        config_path,
        '--work-dir', args.output
    ])
    
    print(f"✅ Training complete! Checkpoints in {args.output}")

def train_with_custom(args):
    """EXISTING: Your custom trainer."""
    # ... your existing code ...
```

**Usage**:
```bash
# Test new MMDet backend
python cli/train_teacher.py --data data.json --use-mmdet --output exp_mmdet

# Compare with existing custom implementation
python cli/train_teacher.py --data data.json --output exp_custom

# Compare results (mAP, loss curves, training time)
```

---

### Phase 2: Config Generation (3 days)

Create a module that converts your data inspection into MMDet configs:

```python
# ml_engine/mmdet_integration/config_generator.py

from pathlib import Path
from mmengine.config import Config

def generate_mmdet_config(
    dataset_info: dict,
    output_dir: str,
    base_config: str = 'mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py'
) -> str:
    """
    Generate MMDetection config from your data inspection results.
    
    Args:
        dataset_info: From DataManager.get_dataset_info()
        output_dir: Where to save generated config
        base_config: MMDet base config to inherit from
    
    Returns:
        Path to generated config file
    """
    # Load base config
    cfg = Config.fromfile(base_config)
    
    # Override with your dataset
    cfg.data_root = dataset_info['image_dir']
    cfg.class_name = tuple(dataset_info['class_names'])
    cfg.num_classes = dataset_info['num_classes']
    cfg.metainfo = dict(
        classes=cfg.class_name,
        palette=[(220, 20, 60)] * cfg.num_classes  # Default colors
    )
    
    # Update model config
    cfg.model.bbox_head.num_classes = cfg.num_classes
    
    # Update dataset paths
    cfg.train_dataloader.dataset.data_root = dataset_info['image_dir']
    cfg.train_dataloader.dataset.ann_file = dataset_info['train_ann_file']
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    
    cfg.val_dataloader.dataset.data_root = dataset_info['image_dir']
    cfg.val_dataloader.dataset.ann_file = dataset_info['val_ann_file']
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    
    # Freeze strategy (layer freezing, not LoRA)
    cfg.optim_wrapper.optimizer.lr = 0.0001
    cfg.optim_wrapper.paramwise_cfg = dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),  # Freeze Swin Transformer
            'language_model': dict(lr_mult=0.0)  # Freeze BERT
        })
    
    # Save generated config
    output_path = Path(output_dir) / 'mmdet_config.py'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output_path))
    
    print(f"✅ Generated MMDet config: {output_path}")
    print(f"   Classes: {cfg.class_name}")
    print(f"   Train data: {cfg.train_dataloader.dataset.ann_file}")
    
    return str(output_path)


# Example usage
dataset_info = {
    'num_classes': 2,
    'class_names': ['invalid', 'valid'],
    'class_mapping': {0: 'invalid', 1: 'valid'},
    'image_dir': 'cli/images/',
    'train_ann_file': 'cli/output.json',
    'val_ann_file': 'cli/output.json',  # Same file for demo
    'has_boxes': True,
    'has_masks': False
}

config_path = generate_mmdet_config(dataset_info, 'experiments/test_mmdet')
```

---

### Phase 3: Integration Testing (2 days)

**Test on your actual dataset:**

```bash
# Generate config
python cli/train_teacher.py \
    --data cli/output.json \
    --images cli/images/ \
    --output experiments/mmdet_test \
    --use-mmdet \
    --epochs 5

# Compare with custom implementation
python cli/train_teacher.py \
    --data cli/output.json \
    --images cli/images/ \
    --output experiments/custom_test \
    --epochs 5

# Compare metrics
python tools/compare_experiments.py \
    experiments/mmdet_test \
    experiments/custom_test
```

**What to compare**:
- Loss values (should be similar)
- mAP scores (MMDet might be higher)
- Training time (MMDet might be faster)
- Memory usage
- Checkpoint size

---

### Phase 4: Full Migration (1 week)

**If Phase 3 results are good, replace custom implementation:**

1. **Update default configs**:
   ```yaml
   # configs/defaults/teacher_grounding_dino.yaml (simplified)
   
   backend: "mmdetection"  # vs "custom"
   
   mmdet:
     base_config: "mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py"
     
   # Freezing strategy
   freeze:
     backbone: true
     language_model: true
   
   # Training params
   epochs: 20
   batch_size: 4
   learning_rate: 0.0001
   ```

2. **Remove old code** (keep as backup):
   ```bash
   # Archive old implementation
   mkdir ml_engine/legacy/
   mv ml_engine/training/teacher_trainer.py ml_engine/legacy/
   mv ml_engine/training/losses.py ml_engine/legacy/
   
   # Keep only MMDet integration
   ml_engine/mmdet_integration/
   ├── __init__.py
   ├── config_generator.py     # 150 lines
   ├── dataset_adapter.py      # 80 lines
   └── runner.py               # 100 lines
   ```

3. **Update documentation**:
   ```markdown
   # TECHNICAL_APPROACH.md
   
   ## Training Backend: MMDetection
   
   We use MMDetection for GroundingDINO fine-tuning because:
   - Proven implementation (higher mAP than official)
   - Extensive tooling
   - Active community support
   
   Our platform adds:
   - Data-driven config generation
   - Simple CLI interface
   - Automatic model selection
   ```

---

## What You Keep vs What You Delegate

### ✅ Keep (Your Unique Value):

```python
# YOUR CODE (data-driven intelligence)
DataManager         # Inspect dataset, determine what to train
ConfigGenerator     # Auto-fill configs from data
CLI                 # Simple user interface
DistillationPipeline  # Not in MMDet, keep custom
EdgeOptimizer       # Not in MMDet, keep custom
```

### ✅ Delegate (Commodity):

```python
# MMDETECTION CODE (proven implementations)
GroundingDINOHead   # Loss functions (no bugs!)
GroundingDINO       # Model architecture
Runner              # Training loop
Hooks               # Logging, checkpointing
Datasets            # Data loading
```

---

## Estimated Effort vs Value

| Task | Effort | Value | Priority |
|------|--------|-------|----------|
| Test MMDet cat example | 2 hours | High (validates feasibility) | P0 |
| Config generation | 2 days | High (auto-config still works) | P0 |
| Dataset adapter | 1 day | Medium (might work as-is) | P1 |
| Parallel testing | 2 days | High (compare results) | P0 |
| Full migration | 1 week | High (eliminate bugs) | P1 |
| LoRA integration | 2 weeks | Low (freezing good enough) | P3 |

**Total for basic migration**: 1-2 weeks  
**Payoff**: Proven training code, higher mAP, less maintenance

---

## Quick Start: Test Right Now

Want to test immediately? Run this:

```bash
cd /root/coding/platform/Grounded-Segment-Anything

# Train with MMDet on your data (manual config creation)
cat > mmdet_test_config.py << 'EOF'
_base_ = 'mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py'

# Your dataset
data_root = 'cli/images/'
class_name = ('invalid', 'valid')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (0, 255, 0)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='../cli/output.json',  # Your COCO file
        data_prefix=dict(img='./')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='../cli/output.json',
        data_prefix=dict(img='./')))

# Smaller training for testing
max_epoch = 5
train_cfg = dict(max_epochs=max_epoch, val_interval=1)
EOF

# Train
cd mmdetection
python tools/train.py ../mmdet_test_config.py --work-dir ../experiments/mmdet_quick_test

# Check results
ls ../experiments/mmdet_quick_test/
```

**This takes 30 minutes**. If it works and produces good loss values (~1-3 for loss_ce), you have your answer.

---

## Bottom Line

**Use MMDetection.** Your current losses are working now, but you found 3 bugs in the custom implementation already:

1. ❌ NaN losses (fixed)
2. ❌ Incorrect scaling by `num_queries` (fixed)
3. ❌ Missing `token_labels` in encoder targets (fixed)

MMDetection has **NONE of these bugs** because they're already debugged by the community.

**Focus your effort on what makes your platform unique:**
- Data-driven pipeline
- Auto-config generation
- Simple CLI
- Distillation pipeline (not in MMDet)
- Edge optimization (not in MMDet)

**Let MMDetection handle the commodity part: training.**

This is exactly what Linus would do - **use proven components, focus on real innovation**. 🎯




