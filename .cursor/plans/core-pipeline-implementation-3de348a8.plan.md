<!-- 3de348a8-e48c-4fa9-8971-33f4a4c55af1 0cb64a90-ccda-49e5-846d-98dfe7895585 -->
# Core Pipeline Implementation Plan

## Week 1: Data Pipeline & Augmentation System

### Phase 1.1: Project Structure Setup

- Create directory structure as per `PLATFORM_ARCHITECTURE.md` lines 114-278
- Set up `requirements.txt` with core dependencies: PyTorch, PEFT, transformers, ultralytics, albumentations, onnx, tensorboard
- Create `core/` module with config manager, logger, and constants
- Initialize git repo with proper `.gitignore` (exclude `experiments/`, `data/models/`, etc.)

### Phase 1.2: Data Inspection & Validation (CLI)

**File**: `cli/validate_dataset.py`

- Implement `inspect_dataset()` function (per `TECHNICAL_APPROACH.md` lines 67-89)
- Returns: `{has_boxes, has_masks, num_classes, class_mapping}`
- NO mode enums - data structure is self-describing
- Implement auto-bbox generation from masks (lines 1169-1210)
- One tight box per mask annotation (NOT one box for all objects)
- Handle polygon and RLE formats
- Implement stratified dataset splitting (lines 960-999)
- Maintain class distribution across train/val/test
- Fixed random seed for reproducibility
- CLI prints validation report with recommended models

**File**: `ml_engine/data/validators.py`

- COCO format validation
- Image-annotation consistency checks
- Class distribution analysis
- Multiple objects per image handling

### Phase 1.3: Characteristic-Based Augmentation System

**File**: `augmentation/__init__.py` (Primary API)

- Export `get_augmentation_registry()` as main entry point

**File**: `augmentation/augmentation_registry.py`

- Singleton registry pattern
- Methods:
- `get_pipeline(characteristics, environment, intensity)` → returns albumentations pipeline
- `get_pipeline_info()` → preview augmentations before creating
- `get_available_characteristics()` → list all options
- `get_available_environments()` → dict of environment options

**File**: `augmentation/characteristic_translator.py`

- Map characteristics → augmentation rules (per lines 1401-1414):
- `changes_shape` → ElasticTransform, PiecewiseAffine
- `reflective_surface` → RandomSunFlare, RandomShadow, ColorJitter
- `low_contrast` → CLAHE, RandomBrightnessContrast, Sharpen
- `moves_or_vibrates` → MotionBlur, SafeRotate
- `semi_transparent` → RandomFog, GaussNoise, Blur
- (etc. - see table lines 1403-1414)
- Map environment → augmentations (lines 1416-1422):
- lighting: stable/variable/poor
- camera: fixed/moving/shaky
- background: clean/busy/changing
- distance: fixed/variable/close
- Intensity scaling (lines 1488-1493): low/medium/high
- Automatic deduplication of overlapping augmentations

**File**: `augmentation/augmentation_factory.py`

- Convert augmentation rules → albumentations `Compose` pipeline
- Handle bbox/mask/keypoint transforms correctly
- Apply probability values based on intensity

**File**: `augmentation/parameter_system.py`

- `RangeParameter` class for value ranges (e.g., rotation: -15 to 15 degrees)
- `NestedParameter` class for complex augmentation configs
- Intensity-based parameter scaling

**File**: `augmentation/transform_builders.py`

- Transform-specific parameter builders for each albumentations transform
- Ensures correct parameter formats for library compatibility

**Config**: `configs/defaults/preprocessing.yaml`

- Model-specific preprocessing (lines 431-469):
- Grounding DINO: 800×1333, ImageNet normalization
- SAM: 1024×1024, SAM-specific normalization (NOT ImageNet!)
- YOLOv8: 640×640, [0,1] normalization

### Phase 1.4: Multi-Model Preprocessing Pipeline

**File**: `ml_engine/data/preprocessing.py`

- `MultiModelPreprocessor` class (lines 502-544)
- Initialize with list of active model names (data-driven)
- `preprocess_batch()` → preprocess for all active models
- `preprocess_for_model()` → preprocess for specific model
- `SingleModelPreprocessor` class (lines 554-673)
- Model-specific resize strategies (keep_aspect_ratio, resize_longest_side, letterbox)
- Model-specific normalization (handle different pixel ranges!)
- Padding strategies
- NO mode enums - just pass list of model names based on data inspection

**File**: `ml_engine/data/loaders.py`

- COCO dataset loader with multi-model preprocessing integration
- Batch collation that handles different resolutions
- Integration with characteristic-based augmentation system

---

## Week 2-3: Teacher Fine-Tuning with LoRA

### Phase 2.1: PEFT/LoRA Integration

**File**: `ml_engine/models/teacher/grounding_dino_lora.py`

- Load base Grounding DINO from `data/models/pretrained/groundingdino_swint_ogc.pth`
- Apply LoRA config (lines 1666-1689):
- `r=16, lora_alpha=32`
- Target modules: self_attn q/k/v/out projections
- Freeze backbone (Swin Transformer - 158M params)
- Verify only LoRA adapters are trainable (lines 1856-1873)
- Expected: 2.5M trainable params (~1.38% of model)

**File**: `ml_engine/models/teacher/sam_lora.py`

- Load base SAM from `data/models/pretrained/sam_vit_h_4b8939.pth`
- Partial freeze strategy (lines 1936-1981):
- Freeze image encoder (308M params)
- Freeze prompt encoder (3.8M params)
- Apply LoRA to mask decoder only (r=8)
- Expected: 0.4M trainable params (~0.13% of model)

**File**: `ml_engine/models/teacher/grounded_sam.py`

- Combine DINO + SAM as sequential pipeline
- Load base models + apply LoRA adapters
- Optional merge for faster inference (`merge_and_unload()`)
- Two-stage forward: text → DINO → boxes → SAM → masks

### Phase 2.2: Training Configuration & Management

**File**: `configs/defaults/teacher_grounding_dino_lora.yaml`

- Default hyperparameters (lines 1667-1700)
- LoRA config (r, alpha, target_modules, dropout)
- Placeholder for auto-filled fields: `num_classes: null, class_names: []`

**File**: `configs/defaults/teacher_sam_lora.yaml`

- SAM-specific defaults (lines 2758-2794)
- Partial freeze config
- Prompt strategy (use GT boxes)

**File**: `configs/defaults/training_dynamics.yaml`

- Gradient clipping config (lines 2084-2088): max_norm=0.1 for LoRA
- Mixed precision (FP16) config (lines 2091-2097)
- BatchNorm handling for LoRA (lines 2100-2108)
- LR warmup config (lines 2111-2117)

**File**: `configs/defaults/checkpoint_config.yaml`

- Checkpointing strategy (lines 2313-2344)
- Best model selection (monitor mAP50)
- Early stopping config (patience=15)
- RNG state saving for reproducibility

**File**: `core/config.py`

- Auto-config generation: `generate_config(default_path, dataset_info, cli_overrides)`
- Load defaults → fill num_classes/class_names from data → apply CLI overrides
- Save to `experiments/{name}/teacher_config.yaml`

### Phase 2.3: Training Manager & Trainer

**File**: `ml_engine/training/training_manager.py`

- `TrainingManager` class (lines 2130-2275)
- Handles gradient clipping, mixed precision, BN configuration
- `training_step()` method with AMP + gradient scaling
- Gradient statistics monitoring

**File**: `ml_engine/training/checkpoint_manager.py`

- `CheckpointManager` class (lines 2362-2615)
- Save checkpoints with optimizer/scheduler/scaler states
- Track best model based on validation metric
- Early stopping logic
- Automatic cleanup of old checkpoints (keep last 5)
- RNG state saving for reproducibility

**File**: `ml_engine/training/teacher_trainer.py`

- Data-driven teacher selection (lines 2817-2850):
- Inspect dataset → load only required teachers
- If has_boxes: train Grounding DINO
- If has_masks: train SAM
- If both: train both simultaneously (22GB on RTX 3090)
- Integration with TrainingManager and CheckpointManager
- TensorBoard logging
- Validation loop with mAP/IoU metrics

### Phase 2.4: Teacher Training CLI

**File**: `cli/train_teacher.py`

- Single command trains all required teachers (lines 374-408)
- Auto-detects from data (no manual selection)
- Steps:

1. Inspect dataset → get annotation types, num_classes, class_mapping
2. Load default configs
3. Auto-fill dataset-specific values
4. Determine which teachers to load (if has_boxes: DINO, if has_masks: SAM)
5. Save config to `experiments/{name}/teacher_config.yaml`
6. Start LoRA fine-tuning

- CLI overrides: `--batch_size`, `--epochs`, `--lora.r`, etc.
- Expected memory: 14.4GB (DINO) + 8GB (SAM) = 22.4GB (fits RTX 3090)

---

## Week 4-5: Student Distillation (Prompt-Free)

### Phase 3.1: Student Model Implementation

**File**: `ml_engine/models/student/yolov8_seg.py`

- YOLOv8-seg wrapper (detection + segmentation)
- Prompt-free interface: `forward(image)` → boxes, masks, class_ids
- NO text or box prompts in forward pass (lines 3609-3645)
- Output adapts to training: if only boxes in data → skip mask head

**File**: `ml_engine/models/student/yolov8.py`

- Detection-only variant (if only boxes in dataset)

**File**: `ml_engine/models/student/fastsam.py`

- Segmentation-focused variant (if only masks in dataset)

### Phase 3.2: Distillation Configuration

**File**: `configs/defaults/distillation.yaml`

- Unified config with conditional components (lines 3123-3179)
- Loss weights: detection, segmentation, logit, feature
- Weights auto-disabled if corresponding data not available
- Class mapping: `{}` (auto-filled from COCO categories)
- Training hyperparameters (epochs=300, lr=1e-3, SGD optimizer)
- Augmentation config (characteristic-based + distillation-specific)
- NO separate configs per annotation type

### Phase 3.3: Distillation Engine

**File**: `ml_engine/training/distillation.py`

- `DistillationTrainer` class (lines 2967-3089)
- Data-driven initialization:
- Load LoRA-adapted teachers (base + adapters)
- Inspect dataset → select student model
- Auto-fill class_mapping from data
- `training_step()`:
- Teacher forward (TWO-STAGE, WITH PROMPTS):
- For each class_id in class_mapping:
- DINO(image, text=class_name) → boxes
- SAM(image, boxes=prompts) → masks
- Run only loaded teachers (data-driven)
- Student forward (SINGLE-STAGE, NO PROMPTS):
- student(image) → boxes, masks, class_ids (direct prediction)
- Loss computation (lines 3028-3057):
- Compute loss for available components only
- If 'boxes' in outputs: detection_loss
- If 'masks' in outputs: segmentation_loss
- If DINO loaded: logit_loss
- If any teacher: feature_loss
- NO mode checking - just check presence

**File**: `ml_engine/training/losses.py`

- Detection loss (IoU + classification)
- Segmentation loss (Dice + BCE)
- Feature distillation loss (MSE on intermediate features)
- Logit distillation loss (KL divergence with temperature)

### Phase 3.4: Distillation Training CLI

**File**: `cli/train_student.py`

- Single command for prompt-free distillation (lines 410-449)
- Steps:

1. Inspect dataset → has_boxes, has_masks, class_mapping
2. Load LoRA-adapted teachers from `--teacher-dir`:

- Find base models in `data/models/pretrained/`
- Find adapters in `experiments/{name}/teachers/`
- Apply PEFT adapters, optionally merge

3. Select student model based on data:

- Both → YOLOv8s-seg
- Boxes only → YOLOv8s
- Masks only → FastSAM-s

4. Load default distillation config
5. Auto-fill class_mapping from COCO categories
6. Train prompt-free student

- Expected training time: 6-12 hours (300 epochs, RTX 4090)
- Expected results: mAP50 0.85-0.92, Mask IoU 0.86-0.90

### Phase 3.5: Prompt-Free Validation & Testing

**File**: `tests/unit/test_prompt_free.py`

- Verify student model interface (lines 3669-3700):
- Check input signature (only 'images', no prompts)
- Test inference without prompts
- Verify class_ids are integers in valid range
- ONNX export verification (lines 3648-3667):
- Export with single image input
- Verify ONNX model has only 1 input
- Automated test suite for CI/CD

---

## ONNX Export (Basic - Week 5 End)

### Phase 4.1: ONNX Export Implementation

**File**: `ml_engine/optimization/onnx_export.py`

- Export student to ONNX (lines 3417-3443)
- Dynamic batch size and resolution
- Input: images only (NO prompts)
- Output: boxes, masks, class_ids, scores
- Verify exported model is prompt-free

**File**: `cli/optimize_model.py`

- CLI for ONNX export
- Basic quantization support (INT8 for future)
- Save to `experiments/{name}/student/optimized/`

---

## Key Implementation Principles

1. **Data-Driven Architecture** (lines 42-70)

- NO mode enums - data structure determines behavior
- Direct inspection: `if 'bbox' in ann: load_model()`
- Stateless pipeline (no .mode_config.json)

2. **Auto-Config Generation** (lines 301-377)

- Read COCO → extract num_classes, class_mapping
- Load defaults → merge with data info
- Save to experiments/ for reproducibility
- Zero manual config editing

3. **LoRA Efficiency** (lines 1734-1773, 1109-1272)

- Freeze ALL base model parameters
- Train only adapter matrices (1-2% of params)
- Memory: 14.4GB (DINO) + 8GB (SAM) fits RTX 3090
- Save only adapters (19MB + 1.5MB)

4. **Prompt-Free Student** (lines 147-185, 3585-3775)

- Teacher uses prompts (from config class_mapping)
- Student learns WITHOUT prompts in single forward pass
- Class knowledge embedded in weights
- Deployment: `student(image)` → direct predictions

5. **Characteristic-Based Augmentation** (lines 1387-1643)

- User describes objects/environment (not augmentations)
- System selects appropriate transforms
- Intensity control (low/medium/high)
- Built on albumentations for production quality

---

## Success Criteria

- Dataset validation CLI works with COCO format (boxes/masks/both)
- Auto-bbox generation from masks (one tight box per object)
- Characteristic-based augmentation system functional
- LoRA teacher fine-tuning: 14.4GB DINO + 8GB SAM on RTX 3090
- Adapters saved: ~19MB (DINO) + ~1.5MB (SAM)
- Student distillation produces prompt-free model
- Student outputs adapt to data (boxes/masks/both + class_ids)
- ONNX export with single image input (no prompts)
- Expected accuracy: mAP50 0.85-0.92 (student vs teacher 0.90-0.95)

### To-dos

- [ ] Create directory structure, requirements.txt, core modules (config, logger, constants), initialize git with proper .gitignore
- [ ] Implement cli/validate_dataset.py with inspect_dataset(), auto-bbox generation from masks, stratified dataset splitting
- [ ] Implement characteristic-based augmentation system: registry, translator, factory, parameter system, transform builders
- [ ] Implement MultiModelPreprocessor with model-specific preprocessing (DINO 800x1333, SAM 1024x1024, YOLO 640x640)
- [ ] Implement COCO dataset loaders with multi-model preprocessing and augmentation integration
- [ ] Implement grounding_dino_lora.py and sam_lora.py with PEFT integration, partial freeze strategy
- [ ] Implement TrainingManager (gradient clipping, mixed precision, BN config) and CheckpointManager (best model tracking, early stopping)
- [ ] Implement data-driven teacher_trainer.py that loads only required teachers based on dataset inspection
- [ ] Implement cli/train_teacher.py with auto-config generation and data-driven teacher selection
- [ ] Implement prompt-free student models: yolov8_seg.py, yolov8.py, fastsam.py with correct interfaces
- [ ] Implement distillation loss functions: detection, segmentation, feature, logit losses
- [ ] Implement DistillationTrainer with data-driven loss computation and prompt-free student training
- [ ] Implement cli/train_student.py with LoRA teacher loading and auto-student selection
- [ ] Implement test suite to verify student is truly prompt-free (interface, ONNX export validation)
- [ ] Implement ONNX export in optimization/onnx_export.py and cli/optimize_model.py