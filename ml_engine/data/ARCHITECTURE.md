# Data Module Architecture - Clean Design

## Design Principles (Linus Style)

1. **"Single Source of Truth"** - DataManager owns ALL data operations
2. **"No Double Work"** - Load once, inspect once, cache everything
3. **"Data-Driven"** - Data structure determines behavior (no mode enums)
4. **"Eliminate Special Cases"** - Enforce sequential IDs, no remapping needed
5. **"Simplicity"** - Each module does ONE thing well

## Module Responsibilities

```
ml_engine/data/
├── manager.py          # CENTRAL ORCHESTRATOR (owns everything)
│   ├─ Load JSON once
│   ├─ Inspect once (cache results)
│   ├─ Validate once
│   ├─ Preprocess once (auto-gen bbox from masks)
│   ├─ Split once (train/val/test)
│   └─ Create PyTorch datasets (pass pre-loaded data)
│
├── inspection.py       # PURE ANALYSIS FUNCTIONS (no I/O, no state)
│   ├─ inspect_dataset(coco_data) → stats
│   ├─ get_required_models(dataset_info) → model list
│   └─ print_dataset_report(dataset_info) → pretty print
│
├── validators.py       # PURE VALIDATION FUNCTIONS (no I/O, no state)
│   ├─ validate_coco_format(coco_data) → (is_valid, errors)
│   ├─ preprocess_coco_dataset(coco_data) → auto-fixed data
│   ├─ split_dataset(coco_data) → splits
│   └─ check_data_quality(coco_data) → quality report
│
├── loaders.py          # PYTORCH DATASET WRAPPERS (receives data, loads images only)
│   ├─ COCODataset(coco_data, image_dir) → base dataset
│   └─ TeacherDataset(coco_data, ...) → teacher-specific dataset
│
└── preprocessing.py    # MODEL-SPECIFIC IMAGE PREPROCESSING (different concern)
    └─ MultiModelPreprocessor → DINO/SAM/YOLO preprocessing
```

## Critical Design Decisions

### Decision 1: Sequential Category IDs (0, 1, 2, ...)

**Enforced**: Category IDs must be sequential starting from 0

```python
# ✅ VALID
categories = [
    {"id": 0, "name": "ear"},
    {"id": 1, "name": "defect"},
    {"id": 2, "name": "label"}
]

# ❌ INVALID (will be rejected)
categories = [
    {"id": 1, "name": "ear"},     # Doesn't start from 0
    {"id": 5, "name": "defect"},  # Not sequential
    {"id": 10, "name": "label"}
]
```

**Benefits**:
- No ID→Index remapping needed
- Simpler code (2 dicts eliminated)
- Clearer for users (class 0, 1, 2 vs class 1, 5, 10)
- Direct array indexing (faster)

**Frontend Requirement**:
```javascript
// Frontend must export sequential IDs
const categories = classDefinitions.map((cls, idx) => ({
    id: idx,  // 0, 1, 2, 3, ...
    name: cls.name
}));
```

### Decision 2: Platform Handles Splitting

**User provides**: ONE annotations.json file

**Platform handles**: Automatic 70/20/10 split with stratification

```python
# User workflow (simplified)
python train_teacher.py --data annotations.json --images images/ --output exp1

# Platform automatically:
# 1. Validates data
# 2. Auto-generates missing bbox from masks
# 3. Splits into train (70%), val (20%), test (10%)
# 4. Detects which models to train
# 5. Trains models
```

**Benefits**:
- User doesn't need ML expertise
- Consistent splitting across users
- Reproducible (fixed random seed)
- Proper stratification (maintains class distribution)

### Decision 3: Three-Tier Config System

**Prevents conflicts** by separating shared vs model-specific parameters:

```
Tier 1: configs/defaults/teacher_training.yaml
  ├─ batch_size: 8       (MUST be same for all models)
  ├─ epochs: 50          (MUST be same - same training loop)
  └─ num_workers: 4      (MUST be same - same dataloader)

Tier 2: configs/defaults/teacher_grounding_dino_lora.yaml
  ├─ learning_rate: 1e-4  (CAN differ - separate optimizer)
  └─ lora: {r: 16, ...}   (DINO-specific)

Tier 2: configs/defaults/teacher_sam_lora.yaml
  ├─ learning_rate: 5e-4  (CAN differ from DINO)
  └─ lora: {r: 8, ...}    (SAM-specific)
```

**Result**: Impossible to have conflicts on shared parameters!

## Data Flow Diagram

```
User provides: annotations.json + images/
         ↓
┌────────────────────────────────────────────────────────────┐
│ DataManager.__init__() (Runs Once)                         │
│ ───────────────────────────────────────────────────────── │
│ 1. Load JSON        → self.raw_data                       │
│ 2. Validate         → Enforce sequential IDs              │
│ 3. Preprocess       → Auto-gen bbox from masks            │
│ 4. Inspect          → self.dataset_info (cached)          │
│ 5. Quality check    → self.quality_report (cached)        │
│ 6. Split            → self.splits['train'/'val'/'test']   │
│                                                             │
│ Everything in memory, nothing reloaded!                    │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ Trainer Initialization                                      │
│ ───────────────────────────────────────────────────────── │
│ trainer = TeacherTrainer(data_manager=manager, ...)        │
│                                                             │
│ Gets from manager (no loading):                            │
│ - dataset_info (cached)                                    │
│ - required_models (computed from dataset_info)             │
│ - train/val datasets (created from cached splits)          │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ Training Loop                                               │
│ ───────────────────────────────────────────────────────── │
│ for batch in train_loader:                                 │
│     # Batch created from manager's cached split            │
│     # No file I/O during training!                         │
│     train_step(batch)                                      │
└────────────────────────────────────────────────────────────┘
```

## Simplified Category Handling

### Before (Complex):
```python
# Three mappings
cat_id_to_idx = {1: 0, 5: 1, 10: 2}     # COCO ID → PyTorch Index
cat_idx_to_name = {0: 'ear', 1: 'defect', 2: 'label'}  # Index → Name
class_mapping = {1: 'ear', 5: 'defect', 10: 'label'}   # ID → Name

# Remapping needed
cat_id = ann['category_id']  # 5
cat_idx = cat_id_to_idx[cat_id]  # 0 → 1
sample['labels'].append(cat_idx)
```

### After (Simple):
```python
# One array
class_names = ['ear', 'defect', 'label']  # Index 0, 1, 2

# No remapping!
cat_id = ann['category_id']  # 1 (already sequential!)
sample['labels'].append(cat_id)  # Use directly
```

## Error Handling Strategy

### Validation Levels:

**Level 1: HARD ERRORS (Stop Training)**
- Missing required keys
- Invalid bbox format
- Non-sequential category IDs
- Missing images referenced in annotations

**Level 2: SOFT WARNINGS (Continue Training)**
- Images without annotations (ignored)
- Class imbalance (might affect quality)
- Small objects (might be hard to detect)
- Missing classes in split (from small dataset)

### Example:
```bash
# Valid but with warnings
✓ Dataset format is valid
⚠️  Data quality warnings:
  - 5 images have no annotations (will be ignored)
  - High class imbalance detected (ratio 15:1)
  - 1 classes have fewer than 10 samples (might overfit)

⚠️  Training will proceed but consider collecting more data
```

## Key Simplifications Achieved

| What | Before | After | Savings |
|------|--------|-------|---------|
| **JSON loads** | 3 times | 1 time | 66% reduction |
| **Dataset inspections** | 2 times | 1 time | 50% reduction |
| **Category mappings** | 3 dicts | 1 array | 2 structures removed |
| **User files** | 2 files (train.json + val.json) | 1 file | 50% simpler |
| **Config conflicts** | Possible | Impossible | 100% safer |
| **split_dataset complexity** | 140 lines | 8 functions | More maintainable |

## Testing Coverage

Created `tests/test_data_manager.py`:
- ✅ Verify JSON loaded only once
- ✅ Verify inspection happens only once
- ✅ Verify datasets receive pre-loaded data
- ✅ Verify no redundant operations

## Usage Example (Complete Workflow)

```python
from ml_engine.data.manager import DataManager
from ml_engine.training.teacher_trainer import TeacherTrainer

# Step 1: User provides ONE file
manager = DataManager(
    data_path='data/raw/annotations.json',  # ONE file with all data
    image_dir='data/raw/images/',
    split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
)

# Step 2: Manager automatically:
# - Validates (enforces sequential IDs)
# - Auto-generates bbox from masks
# - Inspects (has_boxes, has_masks, num_classes)
# - Splits (train/val/test with stratification)
# - Caches everything

# Step 3: Get info (cached, no re-computation)
info = manager.get_dataset_info()
# {'has_boxes': True, 'has_masks': True, 'num_classes': 3, ...}

# Step 4: Create trainer (gets data from manager)
trainer = TeacherTrainer(
    data_manager=manager,  # ONE manager with everything
    output_dir='experiments/exp1',
    config=config
)

# Step 5: Train (uses manager's cached splits)
trainer.train()
# Trainer internally calls:
# - manager.create_pytorch_dataset('train')
# - manager.create_pytorch_dataset('val')
# No file I/O during training!
```

## Migration from Old Code

### Old Way (❌ Don't use):
```python
# Multiple files
dataset = TeacherDataset(
    json_path='train.json',
    image_dir='images/',
    class_mapping={0: 'ear', 1: 'defect'}
)
```

### New Way (✅ Use this):
```python
# Single manager
manager = DataManager('annotations.json', 'images/')
dataset = manager.create_pytorch_dataset('train')
# class_names automatically in dataset.class_names
```

## Benefits Summary

**For Users**:
- ✅ Provide ONE file instead of multiple
- ✅ Platform handles all complexity
- ✅ Clear error messages and warnings
- ✅ No ML expertise required

**For Developers**:
- ✅ Cleaner architecture
- ✅ No redundant operations
- ✅ Easy to test (pure functions)
- ✅ Easy to extend (add new validation)
- ✅ Clear data flow

**For Performance**:
- ✅ Faster (load once, cache everything)
- ✅ Less memory (no duplicate data structures)
- ✅ Direct array indexing (no dict lookups)

This refactoring embodies the principle: **"Good programmers worry about data structures, not code."**


