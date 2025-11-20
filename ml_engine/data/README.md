# Data Module Architecture

This document explains the clean architecture of the `ml_engine/data` module.

## Core Design Principles

1. **Single Source of Truth**: DataManager owns ALL dataset operations
2. **No Double Loading**: JSON files loaded exactly once
3. **Pure Functions**: inspection.py and validators.py contain stateless functions
4. **Clean Interfaces**: PyTorch datasets receive data, never load files

## File Responsibilities

```
ml_engine/data/
├── manager.py          # CENTRAL ORCHESTRATOR (owns all data operations)
├── inspection.py       # Pure analysis functions (no I/O)
├── validators.py       # Validation and cleaning functions (no I/O)
├── loaders.py         # PyTorch Dataset wrappers (receives data from manager)
└── preprocessing.py   # Model-specific image preprocessing
```

### 1. manager.py - DataManager (Central Orchestrator)

**Purpose**: Single source of truth for all dataset operations.

**Responsibilities**:
- ✅ Load COCO JSON once (and only once)
- ✅ Inspect dataset (cache results)
- ✅ Validate and auto-fix data (bbox from masks, etc.)
- ✅ Split train/val/test (if needed)
- ✅ Create PyTorch datasets (pass data to loaders)
- ✅ Cache all results

**Example**:
```python
from ml_engine.data.manager import DataManager

# Create manager (loads everything once)
manager = DataManager(
    data_path='data/raw/train.json',
    image_dir='data/raw/images',
    split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
)

# Get inspection results (cached)
info = manager.get_dataset_info()
print(info['has_boxes'], info['has_masks'])

# Get required models (data-driven)
models = manager.get_required_models()
# ['grounding_dino', 'sam']

# Create PyTorch dataset (no JSON loading in dataset!)
train_dataset = manager.create_pytorch_dataset(
    split='train',
    preprocessor=preprocessor,
    augmentation_pipeline=aug_pipeline
)
```

### 2. inspection.py - Pure Analysis Functions

**Purpose**: Analyze COCO data structure without side effects.

**Characteristics**:
- ❌ No file I/O
- ❌ No state
- ✅ Pure functions only
- ✅ Called by DataManager

**Functions**:
- `inspect_dataset(coco_data)` → Returns dataset statistics
- `get_required_models(dataset_info)` → Returns list of model names
- `print_dataset_report(dataset_info)` → Pretty print report

**Example**:
```python
from ml_engine.data.inspection import inspect_dataset

# Pure function: dict → dict
info = inspect_dataset(coco_data)
# Returns: {'has_boxes': True, 'has_masks': True, ...}
```

### 3. validators.py - Data Cleaning Functions

**Purpose**: Validate and clean COCO data.

**Characteristics**:
- ❌ No file I/O (receives dict, returns dict)
- ❌ No state
- ✅ Pure functions only
- ✅ Called by DataManager

**Functions**:
- `validate_coco_format(coco_data)` → (is_valid, errors)
- `preprocess_coco_dataset(coco_data)` → Auto-generate bbox/area from masks
- `split_dataset(coco_data, splits)` → Split into train/val/test
- `check_data_quality(coco_data)` → Quality report

**Example**:
```python
from ml_engine.data.validators import preprocess_coco_dataset, split_dataset

# Auto-generate missing fields
coco_data = preprocess_coco_dataset(coco_data)

# Split dataset
splits = split_dataset(coco_data, splits={'train': 0.7, 'val': 0.3})
```

### 4. loaders.py - PyTorch Dataset Wrappers

**Purpose**: Wrap pre-loaded COCO data as PyTorch Datasets.

**Characteristics**:
- ❌ NEVER loads JSON files directly
- ✅ Receives pre-loaded data from DataManager
- ✅ Loads images from disk
- ✅ Applies transforms

**Classes**:
- `COCODataset` - Base dataset (receives coco_data dict)
- `TeacherDataset` - Teacher-specific dataset (receives coco_data dict)

**Example**:
```python
from ml_engine.data.loaders import TeacherDataset

# ❌ WRONG: Don't load JSON directly
# dataset = TeacherDataset(json_path='train.json', ...)

# ✅ CORRECT: Get data from DataManager
manager = DataManager('train.json', 'images/')
train_data = manager.get_split('train')
dataset = TeacherDataset(
    coco_data=train_data,  # ← Pre-loaded data
    image_dir='images/',
    preprocessor=preprocessor
)
```

### 5. preprocessing.py - Model-Specific Image Preprocessing

**Purpose**: Handle model-specific image preprocessing (DIFFERENT concern!).

**Characteristics**:
- ❌ No JSON loading (different concern)
- ✅ Handles image transformations
- ✅ Model-specific (DINO: 800×1333, SAM: 1024×1024, YOLO: 640×640)

**Classes**:
- `MultiModelPreprocessor` - Preprocesses for multiple models
- `SingleModelPreprocessor` - Preprocesses for one model

**This is CORRECT as-is** - It's a different concern from data management.

## Data Flow Diagram

```
User provides: train.json + images/
         ↓
┌────────────────────────────────────────────────────────────┐
│ DataManager (Single Entry Point)                           │
│   ├─ Load JSON once                                        │
│   ├─ Inspect once (has_boxes, has_masks, class_mapping)   │
│   ├─ Validate and auto-fix (bbox from masks)              │
│   ├─ Split if needed (train/val/test)                     │
│   └─ Cache everything                                      │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ Get PyTorch Dataset from Manager                           │
│   ├─ manager.create_pytorch_dataset('train')              │
│   ├─ Returns COCODataset with pre-loaded data             │
│   └─ No JSON loading in dataset!                          │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ Training Loop                                               │
│   ├─ DataLoader wraps COCODataset                         │
│   ├─ Preprocessing applies during __getitem__             │
│   └─ Models train on preprocessed data                    │
└────────────────────────────────────────────────────────────┘
```

## Before vs After Refactoring

### Before (❌ Bad - Scattered Responsibilities)

```python
# train_teacher.py - loads and inspects
dataset_info = load_and_inspect_dataset(args.data)

# teacher_trainer.py - loads and inspects AGAIN!
train_data = load_json(train_data_path)
self.dataset_info = inspect_dataset(train_data)

# loaders.py - loads JSON AGAIN!
with open(json_path) as f:
    self.coco_data = json.load(f)

# Result: JSON loaded 3 times, inspected 2 times!
```

### After (✅ Good - Single Source of Truth)

```python
# train_teacher.py - creates manager
data_manager = DataManager(args.data, args.images)
dataset_info = data_manager.get_dataset_info()  # Cached

# teacher_trainer.py - receives manager
def __init__(self, train_data_manager, val_data_manager, ...):
    self.dataset_info = train_data_manager.get_dataset_info()  # Cached

# loaders.py - receives pre-loaded data
def __init__(self, coco_data, image_dir, ...):
    self.coco_data = coco_data  # Already loaded by manager

# Result: JSON loaded once, inspected once, cached everywhere!
```

## Usage Examples

### Example 1: Training Teacher Models

```python
from ml_engine.data.manager import DataManager
from ml_engine.training.teacher_trainer import TeacherTrainer

# Step 1: Create ONE manager (user provides one dataset, platform splits it)
manager = DataManager(
    data_path='data/raw/annotations.json',  # User provides ONE file
    image_dir='data/raw/images/',
    split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}  # Platform splits!
)

# Step 2: Create trainer (passes ONE manager)
# Trainer automatically uses 'train' and 'val' splits
trainer = TeacherTrainer(
    data_manager=manager,  # ONE manager with all splits
    output_dir='experiments/exp1',
    config=config
)

# Step 3: Train
trainer.train()

# Design philosophy:
# - User provides ONE dataset file
# - Platform handles splitting (no manual train/val files)
# - Simpler for non-ML experts
```

### Example 2: Dataset Splitting

```python
from ml_engine.data.manager import DataManager

# Create manager with split config
manager = DataManager(
    data_path='data/raw/annotations.json',
    image_dir='data/raw/images/',
    split_config={'train': 0.7, 'val': 0.2, 'test': 0.1}
)

# Get splits
train_data = manager.get_split('train')
val_data = manager.get_split('val')
test_data = manager.get_split('test')

# Save splits to separate files (optional)
manager.save_splits('data/processed/')
# Creates: train.json, val.json, test.json
```

### Example 3: Data-Driven Model Selection

```python
from ml_engine.data.manager import DataManager

# Load dataset
manager = DataManager('data/raw/train.json', 'data/raw/images/')

# Get dataset info (cached)
info = manager.get_dataset_info()
print(f"Has boxes: {info['has_boxes']}")
print(f"Has masks: {info['has_masks']}")
print(f"Classes: {info['class_mapping']}")

# Get required models (data-driven!)
models = manager.get_required_models()
# If has_boxes and has_masks → ['grounding_dino', 'sam']
# If has_boxes only → ['grounding_dino']
# If has_masks only → ['sam']
```

## Benefits of This Architecture

1. ✅ **No Double Loading**: JSON loaded exactly once
2. ✅ **No Double Inspection**: Dataset inspected exactly once
3. ✅ **Single Source of Truth**: DataManager owns all data operations
4. ✅ **Clean Separation**: Each file has ONE clear responsibility
5. ✅ **Easy Testing**: Pure functions are easy to test
6. ✅ **Better Performance**: Caching avoids redundant operations
7. ✅ **Simpler Code**: Clear data flow, no confusion about who loads what

## Migration Guide

If you have existing code using the old pattern:

```python
# OLD WAY (don't use anymore)
from ml_engine.data.loaders import TeacherDataset
dataset = TeacherDataset(
    json_path='train.json',  # ❌ Old API
    image_dir='images/'
)

# NEW WAY (use this)
from ml_engine.data.manager import DataManager
manager = DataManager('train.json', 'images/')
dataset = manager.create_pytorch_dataset(
    split='all',
    preprocessor=preprocessor
)
```

## Summary

**The golden rule**: DataManager is the ONLY class that loads JSON files. Everyone else gets data FROM the manager.

**File responsibilities**:
- `manager.py` - Owns everything (loads, inspects, validates, splits, caches)
- `inspection.py` - Pure analysis (no I/O)
- `validators.py` - Pure validation (no I/O)
- `loaders.py` - PyTorch wrappers (receives data, loads images)
- `preprocessing.py` - Image preprocessing (different concern, already correct)

**Result**: Clean architecture with clear responsibilities and no redundancy.

