# Configuration Architecture - Three-Tier System

This document explains the clean three-tier configuration system for teacher model training.

## Design Philosophy

**Goal**: User provides ONE dataset file, platform handles ALL config generation automatically.

**Problem**: Training two models (DINO + SAM) simultaneously with potentially conflicting settings.

**Solution**: Three-tier config system with clear separation of concerns.

## Three-Tier System

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 1: Shared Training Config (teacher_training.yaml)          │
│ ─────────────────────────────────────────────────────────────── │
│ Parameters that MUST be the same for all models                 │
│ (because they share the same training loop and dataloader)      │
│                                                                  │
│ - batch_size: 8           ← Same dataloader                     │
│ - epochs: 50              ← Same training loop                  │
│ - num_workers: 4          ← Same dataloader                     │
│ - optimizer: AdamW        ← Can be same or different            │
│ - augmentation: {...}     ← Applied to all models               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Tier 2: Model-Specific Configs (data-driven)                    │
│ ─────────────────────────────────────────────────────────────── │
│ Parameters unique to each model                                 │
│                                                                  │
│ teacher_grounding_dino_lora.yaml   │  teacher_sam_lora.yaml     │
│ ─────────────────────────────────  │  ─────────────────────     │
│ - base_checkpoint (DINO path)      │  - base_checkpoint (SAM)   │
│ - lora.r: 16                       │  - lora.r: 8               │
│ - lora_alpha: 32                   │  - lora_alpha: 16          │
│ - target_modules: [DINO layers]    │  - target_modules: [SAM]   │
│ - learning_rate: 1e-4              │  - learning_rate: 5e-4     │
│ - freeze_backbone: true            │  - freeze_image_encoder    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Tier 3: Runtime Auto-Fill + CLI Overrides                       │
│ ─────────────────────────────────────────────────────────────── │
│ - Dataset info (from COCO JSON)                                 │
│   - num_classes: 3                                              │
│   - class_names: ['ear', 'defect', 'label']                     │
│   - class_mapping: {0: 'ear', 1: 'defect', 2: 'label'}         │
│                                                                  │
│ - CLI overrides (optional)                                      │
│   --batch-size 16  → overrides tier 1                           │
│   --lora-r 32      → overrides tier 2                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Final Merged Config
```

## File Responsibilities

### Tier 1: `teacher_training.yaml` (Shared)

**Contains**: Parameters that MUST be identical for all teacher models

**Why**: Both models train in the same loop with the same dataloader

**Parameters**:
- `batch_size` - Physically impossible to have different values (same dataloader)
- `epochs` - Same training duration for both models
- `num_workers` - Same dataloader configuration
- `optimizer` - Can be shared or overridden per model
- `augmentation` - Applied to all training data
- `checkpointing` - Same saving strategy
- `evaluation` - Same validation interval

### Tier 2: Model-Specific Configs

#### `teacher_grounding_dino_lora.yaml`

**Contains**: ONLY Grounding DINO-specific settings

**Parameters**:
- `model.base_checkpoint` - Path to DINO pretrained weights
- `lora.*` - DINO-specific LoRA configuration (rank=16, alpha=32)
- `learning_rate` - DINO has separate optimizer (can differ from SAM)
- `freeze_backbone` - DINO-specific freezing strategy
- `evaluation.metric` - "mAP50" for detection

#### `teacher_sam_lora.yaml`

**Contains**: ONLY SAM-specific settings

**Parameters**:
- `model.base_checkpoint` - Path to SAM pretrained weights
- `model.model_type` - SAM variant (vit_h, vit_l, vit_b)
- `lora.*` - SAM-specific LoRA configuration (rank=8, alpha=16)
- `learning_rate` - SAM has separate optimizer (can differ from DINO)
- `freeze_image_encoder` - SAM-specific freezing
- `prompt_type` - SAM-specific (boxes)
- `evaluation.metric` - "mask_IoU" for segmentation

### Tier 3: Runtime (Automatic)

**Auto-filled from dataset**:
- `num_classes` - From `len(coco_data['categories'])`
- `class_names` - From `categories[*]['name']`
- `class_mapping` - From category ID to name mapping

**CLI overrides** (optional):
```bash
python cli/train_teacher.py \
    --data annotations.json \
    --images images/ \
    --output exp1 \
    --batch-size 16 \        # Override tier 1
    --lora-r 32              # Override tier 2 for all models
```

## Merge Priority (Highest to Lowest)

```
4. CLI Overrides         (--batch-size 16)
   ↓
3. Dataset Info          (num_classes from COCO)
   ↓
2. Model-Specific        (teacher_grounding_dino_lora.yaml)
   ↓
1. Shared Training       (teacher_training.yaml)
```

## What Gets Saved

After merging, the platform saves a unified config to `experiments/exp1/teacher_config.yaml`:

```yaml
# experiments/exp1/teacher_config.yaml (GENERATED)

# From Tier 1 (shared)
batch_size: 8
epochs: 50
num_workers: 4
optimizer: "AdamW"
weight_decay: 1.0e-4

# From Tier 3 (dataset)
num_classes: 3
class_names: ['ear', 'defect', 'label']
class_mapping: {0: 'ear', 1: 'defect', 2: 'label'}

# From Tier 2 (model-specific)
models:
  grounding_dino:
    base_checkpoint: "data/models/pretrained/groundingdino_swint_ogc.pth"
    learning_rate: 1.0e-4    # DINO-specific
    lora:
      r: 16
      lora_alpha: 32
  
  sam:
    base_checkpoint: "data/models/pretrained/sam_vit_h_4b8939.pth"
    learning_rate: 5.0e-4    # SAM-specific (different from DINO!)
    lora:
      r: 8
      lora_alpha: 16
```

## Benefits

| Aspect | Old (Single-Tier) | New (Three-Tier) |
|--------|------------------|------------------|
| **Conflicts** | ❌ batch_size differs → which to use? | ✅ Impossible - in separate file |
| **User Understanding** | ❌ "Why did my batch_size change?" | ✅ Clear hierarchy |
| **Maintainability** | ❌ Duplicate params everywhere | ✅ Each param defined once |
| **Debugging** | ❌ Hard to find which config won | ✅ Clear merge order |
| **Extensibility** | ❌ Add model = check for conflicts | ✅ Just add model config file |

## Example User Workflow

```bash
# User just provides data - platform handles config complexity
$ python cli/train_teacher.py \
    --data data/raw/annotations.json \
    --images data/raw/images/ \
    --output experiments/my_exp

# Platform automatically:
# [Tier 1] Loads teacher_training.yaml (batch_size=8, epochs=50)
# [Tier 2] Detects has_boxes=True, has_masks=True
#          Loads teacher_grounding_dino_lora.yaml (DINO LoRA settings)
#          Loads teacher_sam_lora.yaml (SAM LoRA settings)
# [Tier 3] Auto-fills num_classes=3, class_names from COCO
# Merges all → Saves to experiments/my_exp/teacher_config.yaml
# Trains both models with consistent batch_size=8, epochs=50
# Each model uses its own learning_rate (DINO: 1e-4, SAM: 5e-4)
```

## How to Override

```bash
# Override shared param (affects all models)
--batch-size 16          # Changes tier 1

# Override model-specific param for all models
--lora-r 32              # Changes tier 2 for both DINO and SAM

# Override for specific model (future feature)
--models.grounding_dino.learning_rate 2e-4  # Only DINO
```

## Key Insight

**Shared params in separate file = Impossible to have conflicts**

- ❌ Old way: batch_size defined in both configs → which wins?
- ✅ New way: batch_size defined only in `teacher_training.yaml` → no ambiguity!

This follows your principle: **"Eliminate special cases" and "Single source of truth"**

