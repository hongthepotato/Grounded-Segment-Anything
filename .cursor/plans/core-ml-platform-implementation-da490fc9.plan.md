<!-- da490fc9-60a9-4a86-a25f-8869249265d3 eeef9cac-daec-4923-be51-2c07a1c78c53 -->
# Core ML Platform Implementation Plan

## Overview

Build a complete ML platform that:

- Automatically detects annotation modes (boxes/masks/both)
- Fine-tunes teacher models (Grounding DINO + SAM) using PEFT/LoRA for memory efficiency
- Distills knowledge into prompt-free student models
- Exports optimized models for edge deployment
- Provides CLI tools for all operations

Target: RTX 3090/4090 (24GB), full pipeline from COCO dataset to optimized edge models.

---

## Phase 1: Project Foundation (Days 1-2)

### 1.1 Directory Structure & Dependencies

**Create complete directory structure:**

```
platform/
├── cli/                    # Command-line interface
├── augmentation/          # Characteristic-based augmentation
├── ml_engine/
│   ├── data/             # Data loading & preprocessing
│   ├── models/           # Teacher & student models
│   ├── training/         # Training logic
│   ├── optimization/     # Model optimization
│   ├── inference/        # Inference engines
│   ├── evaluation/       # Metrics & benchmarking
│   └── utils/
├── configs/
│   ├── templates/        # Template configs (committed)
│   ├── training/         # Active configs (gitignored)
│   ├── distillation/
│   └── optimization/
├── core/                 # Core utilities
├── scripts/             # Setup scripts
├── tests/               # Unit & integration tests
├── data/
│   ├── raw/
│   ├── models/
│   │   ├── pretrained/
│   │   ├── teachers/
│   │   └── students/
│   └── experiments/
└── logs/
```

**Key files:**

- `requirements.txt`: Add torch, transformers, peft, ultralytics, albumentations, onnx, tensorrt, opencv-python, pycocotools, pyyaml, tensorboard
- `setup.py`: Package setup with entry points for CLI
- `.gitignore`: Ignore active configs, data/, logs/, checkpoints
- `core/config.py`: YAML configuration loader
- `core/logger.py`: Logging setup with file + console handlers
- `core/constants.py`: Default paths, model URLs, constants

### 1.2 Core Configuration System

**Implement `core/config.py`:**

- Load YAML configs with validation
- Deep merge for config inheritance
- Environment variable substitution
- Config schema validation using pydantic

**Implement `core/logger.py`:**

- Structured logging with levels
- File rotation
- TensorBoard integration
- Context managers for logging scopes

---

## Phase 2: Data Pipeline (Days 3-5)

### 2.1 COCO Dataset Loading & Validation

**File: `ml_engine/data/loaders.py`**

Implement `COCODataset` class:

- Load COCO JSON (single file with all images)
- Support all annotation modes (boxes/masks/both)
- Memory-efficient loading with caching
- Image path resolution
- Multi-object handling (one annotation per object)

**File: `ml_engine/data/validators.py`**

Implement validation functions:

- `validate_coco_format()`: Check JSON schema
- `validate_images()`: Verify image files exist and are readable
- `detect_annotation_mode()`: Scan for bbox/segmentation fields
- `preprocess_annotations()`: Auto-generate bbox from masks if missing
- `compute_dataset_statistics()`: Class distribution, image sizes
- `validate_class_consistency()`: Check category IDs are valid

**Auto-preprocessing features:**

- Generate tight bounding boxes from segmentation masks (one box per mask)
- Compute area from masks if missing
- Handle multiple disconnected objects correctly

### 2.2 Dataset Splitting & Mode Detection

**File: `ml_engine/data/preprocessing.py`**

Implement:

- `split_dataset()`: Stratified train/val/test split with reproducible seeds
- `save_mode_config()`: Generate `.mode_config.json` with detected mode, recommended models
- `compute_bbox_from_mask()`: Polygon/RLE → tight bbox
- `compute_area_from_mask()`: Calculate segmentation area

**Mode config format:**

```json
{
  "annotation_mode": "detection_and_segmentation",
  "num_classes": 3,
  "class_mapping": {0: "class1", 1: "class2", 2: "class3"},
  "recommended_teachers": ["grounding_dino", "sam"],
  "recommended_student": "yolov8_seg",
  "distillation_config": "kd_both.yaml"
}
```

### 2.3 Mode-Aware Preprocessing Pipeline

**File: `ml_engine/data/preprocessing.py`**

Implement `ModeAwarePreprocessor` class:

- Automatically selects active models based on annotation mode
- Model-specific preprocessing (DINO: 800×1333, SAM: 1024×1024, YOLO: 640×640)
- Batch preprocessing for all active models
- Handles resolution mismatches elegantly
- Returns dict: `{'grounding_dino': (tensor, metadata), 'sam': (tensor, metadata), ...}`

Implement `SingleModelPreprocessor` class:

- Normalization (ImageNet for DINO, SAM-specific for SAM, YOLO-specific)
- Resizing strategies (keep_aspect_ratio, resize_longest_side, letterbox)
- Padding (zero padding for DINO/SAM, gray padding for YOLO)

### 2.4 Characteristic-Based Augmentation System

**File: `augmentation/augmentation_registry.py`**

Implement singleton registry:

- `get_pipeline()`: Build augmentation pipeline from characteristics
- `get_pipeline_info()`: Preview augmentations before building
- `get_available_characteristics()`: List all supported characteristics
- `get_available_environments()`: List environment options

**File: `augmentation/characteristic_translator.py`**

Implement translator:

- Map characteristics → augmentation rules
- Map environment conditions → augmentation rules
- Intensity-based parameter scaling (low/medium/high)
- Automatic deduplication of overlapping augmentations

**File: `augmentation/augmentation_factory.py`**

Build albumentations Compose pipelines:

- Convert parameter configs → albumentations transforms
- Handle bbox/mask/keypoint transforms
- Apply probability and intensity modulation

**Supported characteristics:**

- `changes_shape`, `changes_size`, `reflective_surface`, `low_contrast`, `moves_or_vibrates`, `semi_transparent`, `similar_to_background`, `multiple_objects`, `partially_hidden`

**Environment options:**

- `lighting`: stable/variable/poor
- `camera`: fixed/moving/shaky
- `background`: clean/busy/changing
- `distance`: fixed/variable/close

---

## Phase 3: Teacher Models with LoRA (Days 6-10)

### 3.1 Grounding DINO with PEFT/LoRA

**File: `ml_engine/models/teacher/grounding_dino.py`**

Implement `GroundingDINO` class:

- Load pretrained base model from checkpoint
- Text encoder for prompts
- Image encoder (Swin Transformer)
- Fusion module and detection head
- Feature extraction for distillation

**File: `ml_engine/models/teacher/grounding_dino_lora.py`**

Implement LoRA integration:

- `apply_lora_to_grounding_dino()`: Use PEFT to add LoRA adapters
- Target modules: `["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]`
- LoRA config: r=16, alpha=32, dropout=0.1
- Verify freezing: Only LoRA adapters trainable (~2.5M params)
- `merge_lora_weights()`: Merge adapters into base for inference

### 3.2 SAM with PEFT/LoRA

**File: `ml_engine/models/teacher/sam.py`**

Implement `SAM` class:

- Load pretrained SAM (ViT-H)
- Image encoder (frozen)
- Prompt encoder (frozen)
- Mask decoder
- Feature extraction for distillation

**File: `ml_engine/models/teacher/sam_lora.py`**

Implement LoRA integration:

- `apply_lora_to_sam()`: Add LoRA to mask decoder only
- Target modules: `["mask_decoder.transformer.layers.*.self_attn.*"]`
- LoRA config: r=8, alpha=16, dropout=0.05
- Freeze image encoder (308M params)
- Freeze prompt encoder (3.8M params)
- Only decoder LoRA trainable (~0.4M params)

### 3.3 Combined GroundedSAM Teacher

**File: `ml_engine/models/teacher/grounded_sam.py`**

Implement `GroundedSAM` wrapper:

- Load base models + LoRA adapters using PEFT
- `from_pretrained()`: Load base + apply LoRA adapters
- `predict()`: Sequential pipeline (DINO → SAM)
- Mode-aware: Only load models needed for annotation mode
- `use_merged` option: Merge LoRA weights for faster inference
- Feature extraction from both models for distillation

**Loading pattern:**

```python
base_dino = load_grounding_dino("pretrained/groundingdino.pth")
dino = PeftModel.from_pretrained(base_dino, "teachers/dino_lora/")

base_sam = load_sam("pretrained/sam_vit_h.pth")
sam = PeftModel.from_pretrained(base_sam, "teachers/sam_lora/")
```

### 3.4 Training Infrastructure

**File: `ml_engine/training/training_manager.py`**

Implement `TrainingManager`:

- Gradient clipping (max_norm=0.1 for LoRA)
- Mixed precision training with AMP
- Gradient scaler with dynamic scaling
- BatchNorm freezing for LoRA fine-tuning
- `training_step()`: Unified training step handler

**File: `ml_engine/training/checkpoint_manager.py`**

Implement `CheckpointManager`:

- Config-driven checkpoint strategy
- Best model selection based on metrics
- Early stopping with patience
- Automatic cleanup of old checkpoints
- Save optimizer/scheduler/scaler states
- RNG state saving for reproducibility

### 3.5 Teacher Trainer with LoRA

**File: `ml_engine/training/teacher_trainer.py`**

Implement `TeacherTrainer`:

- Mode-aware: Train only required teachers
- Apply LoRA using PEFT library
- Verify freezing correctness
- Training loop with validation
- TensorBoard logging
- Checkpoint management integration
- Loss computation (detection loss for DINO, segmentation loss for SAM)

**Key methods:**

- `train_epoch()`: Single epoch training
- `validate()`: Validation with metrics
- `_verify_lora_freezing()`: Safety check for correct freezing
- Save LoRA adapters only (not full model)

---

## Phase 4: Student Models (Days 11-12)

### 4.1 Student Model Implementations

**File: `ml_engine/models/student/yolov8.py`**

Implement `YOLOv8` (detection only):

- Load from ultralytics
- Detection head only (no segmentation)
- Output: boxes + class_ids
- NO prompt parameters in forward()

**File: `ml_engine/models/student/yolov8_seg.py`**

Implement `YOLOv8Seg` (detection + segmentation):

- Load YOLOv8-seg from ultralytics
- Detection + segmentation heads
- Output: boxes + masks + class_ids
- NO prompt parameters

**File: `ml_engine/models/student/fastsam.py`**

Implement `FastSAM` (segmentation only):

- Load FastSAM
- Segmentation-focused
- Output: masks + class_ids
- NO prompt parameters

**File: `ml_engine/models/registry.py`**

Implement model registry:

- `get_student_model()`: Factory function
- Mode-aware model selection
- Model configuration management

---

## Phase 5: Distillation Engine (Days 13-17)

### 5.1 Mode-Aware Distillation Trainer

**File: `ml_engine/training/distillation.py`**

Implement `DistillationTrainer`:

- Load LoRA-adapted teachers (base + adapters)
- Load class mapping from config (NOT hardcoded)
- Mode-aware loss computation
- Unified training loop (no special cases)

**Key architecture:**

```python
# Teacher: Sequential two-stage WITH prompts
for class_id, class_name in config['class_mapping'].items():
    dino_boxes = teacher.grounding_dino(image, text_prompt=class_name)
    sam_masks = teacher.sam(image, box_prompts=dino_boxes)

# Student: Single-stage NO prompts
student_output = student(image)  # Direct prediction
```

**Training loop:**

- Teacher inference (frozen, with prompts from config)
- Student inference (trainable, no prompts)
- Compute mode-aware loss (detection/segmentation/feature/logit)
- Backprop through student only

### 5.2 Multi-Component Loss Functions

**File: `ml_engine/training/losses.py`**

Implement distillation losses:

- `detection_loss()`: Box regression + classification (GIoU + cross-entropy)
- `segmentation_loss()`: Mask quality (BCE + Dice loss)
- `feature_distillation_loss()`: MSE between teacher/student features
- `logit_distillation_loss()`: KL divergence with temperature scaling

**Mode-aware loss computation:**

```python
def compute_distillation_loss(student_pred, teacher_out, gt, mode_config):
    total_loss = 0.0
    for loss_name in mode_config['loss_components']:
        if loss_name == 'detection_loss' and 'boxes' in student_pred:
            total_loss += weight * detection_loss(...)
        elif loss_name == 'segmentation_loss' and 'masks' in student_pred:
            total_loss += weight * segmentation_loss(...)
        # ... etc
    return total_loss
```

### 5.3 Helper Functions

**File: `ml_engine/training/distillation.py`**

Implement helpers:

- `_boxes_from_masks()`: Generate tight bbox from mask (one per mask annotation)
- `_align_teacher_student_outputs()`: Handle resolution differences
- `_resize_masks()`: Resize masks to common size for loss computation

---

## Phase 6: Optimization Pipeline (Days 18-20)

### 6.1 ONNX Export

**File: `ml_engine/optimization/onnx_export.py`**

Implement:

- `export_to_onnx()`: PyTorch → ONNX with dynamic axes
- Input validation (ensure single image input, no prompts)
- Opset version 13+ for compatibility
- Constant folding optimization

### 6.2 Quantization

**File: `ml_engine/optimization/quantization.py`**

Implement:

- `quantize_int8_dynamic()`: Dynamic quantization (easiest)
- `quantize_int8_static()`: Static quantization with calibration
- Calibration data loader
- Accuracy evaluation after quantization

### 6.3 TensorRT Conversion

**File: `ml_engine/optimization/tensorrt_export.py`**

Implement:

- `convert_to_tensorrt()`: ONNX → TensorRT engine
- FP16/INT8 precision support
- Optimization profiles for dynamic shapes
- Jetson-specific optimizations

---

## Phase 7: Inference & Evaluation (Days 21-23)

### 7.1 Inference Engines

**File: `ml_engine/inference/pytorch_engine.py`**

Implement `PyTorchEngine`:

- Load .pt checkpoint
- Preprocessing
- Forward pass
- Postprocessing (NMS, mask thresholding)

**File: `ml_engine/inference/onnx_engine.py`**

Implement `ONNXEngine`:

- Load ONNX model
- ONNXRuntime session
- Execution providers (CUDA/CPU)

**File: `ml_engine/inference/tensorrt_engine.py`**

Implement `TensorRTEngine`:

- Load TensorRT engine
- CUDA context management
- Inference with proper memory management

### 7.2 Evaluation Metrics

**File: `ml_engine/evaluation/metrics.py`**

Implement:

- `compute_map()`: Mean Average Precision (COCO metrics)
- `compute_iou()`: Intersection over Union for masks
- `compute_precision_recall()`: Precision/recall curves
- Mode-aware metrics (detection vs segmentation vs both)

**File: `ml_engine/evaluation/benchmark.py`**

Implement:

- `benchmark_speed()`: FPS, latency measurements
- `benchmark_memory()`: GPU/CPU memory usage
- `benchmark_accuracy()`: mAP, IoU, etc.
- Comparison reports (teacher vs student)

---

## Phase 8: CLI Tools (Days 24-27)

### 8.1 Dataset Validation CLI

**File: `cli/validate_dataset.py`**

Implement:

- Load COCO JSON
- Auto-preprocess: Generate bbox from masks if missing
- Detect annotation mode automatically
- Validate format and images
- Compute statistics
- Optional: Split dataset with stratification
- Generate `.mode_config.json`
- Optional: Auto-generate training configs

**Usage:**

```bash
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.7,val:0.15,test:0.15 \
    --generate-configs
```

### 8.2 Teacher Training CLI

**File: `cli/train_teacher.py`**

Implement:

- Load mode config
- Auto-select teachers based on mode
- Apply LoRA using PEFT
- Training loop with progress bars
- TensorBoard logging
- Save LoRA adapters only

**Usage:**

```bash
python cli/train_teacher.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --auto-detect \
    --output data/models/teachers/ \
    --gpu 0
```

### 8.3 Student Distillation CLI

**File: `cli/train_student.py`**

Implement:

- Load LoRA-adapted teachers (base + adapters)
- Load class mapping from config
- Mode-aware student selection
- Distillation training
- Save prompt-free student model

**Usage:**

```bash
python cli/train_student.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --auto-detect \
    --output data/models/students/ \
    --gpu 0
```

### 8.4 Optimization CLI

**File: `cli/optimize_model.py`**

Implement:

- ONNX export
- INT8 quantization
- TensorRT conversion
- Format validation

**Usage:**

```bash
python cli/optimize_model.py \
    --model data/models/students/best.pt \
    --format onnx \
    --quantize int8 \
    --output data/models/optimized/
```

### 8.5 Evaluation & Inference CLIs

**File: `cli/evaluate.py`**

Implement model evaluation with metrics reporting.

**File: `cli/inference.py`**

Implement batch inference with visualization.

---

## Phase 9: Configuration Templates (Days 28-29)

### 9.1 Create All Template Configs

Create in `configs/templates/`:

1. **Teacher configs:**

   - `teacher_grounding_dino_lora.template.yaml`: LoRA config, training hyperparameters
   - `teacher_sam_lora.template.yaml`: LoRA config, training hyperparameters

2. **Student configs:**

   - `student_yolov8.template.yaml`: Detection only
   - `student_yolov8_seg.template.yaml`: Detection + segmentation
   - `student_fastsam.template.yaml`: Segmentation only

3. **Distillation configs:**

   - `kd_detection_only.template.yaml`: Loss components, weights for detection mode
   - `kd_segmentation_only.template.yaml`: Loss components, weights for segmentation mode
   - `kd_both.template.yaml`: Loss components, weights for both mode

4. **Training dynamics:**

   - `training_dynamics.template.yaml`: Gradient clipping, mixed precision
   - `checkpoint_config.template.yaml`: Checkpoint strategy, early stopping
   - `preprocessing.template.yaml`: Model-specific preprocessing configs

5. **Augmentation:**

   - `augmentation.template.yaml`: Characteristic-based augmentation config

6. **Optimization:**

   - `quantization.template.yaml`: Quantization settings
   - `tensorrt.template.yaml`: TensorRT optimization settings

All templates should include:

- Detailed comments explaining each parameter
- Reasonable defaults
- Placeholders for dataset-specific values (num_classes, class_names)

### 9.2 Add .gitignore Files

Create `.gitignore` in:

- `configs/training/`: Ignore `*.yaml`
- `configs/distillation/`: Ignore `*.yaml`
- `configs/optimization/`: Ignore `*.yaml`
- `configs/deployment/`: Ignore `*.yaml`

Keep templates committed to git.

---

## Phase 10: Testing & Validation (Days 30-32)

### 10.1 Unit Tests

**File: `tests/unit/test_data_pipeline.py`**

Test:

- COCO loading with all annotation modes
- Bbox generation from masks
- Mode detection logic
- Dataset splitting with stratification

**File: `tests/unit/test_models.py`**

Test:

- Model loading
- LoRA application
- Forward passes
- Output shapes

**File: `tests/unit/test_lora.py`**

Test:

- LoRA freezing correctness
- Parameter counts
- Adapter loading/saving

**File: `tests/unit/test_prompt_free.py`**

Test:

- Student models have no prompt parameters
- ONNX exports have single image input
- Inference works without prompts

### 10.2 Integration Tests

**File: `tests/integration/test_training_pipeline.py`**

Test:

- End-to-end teacher training (small dataset)
- End-to-end distillation (small dataset)
- Mode-aware pipeline adaptation

**File: `tests/integration/test_optimization_pipeline.py`**

Test:

- ONNX export
- Quantization
- TensorRT conversion (if available)

---

## Phase 11: Documentation & Setup Scripts (Days 33-34)

### 11.1 Setup Scripts

**File: `scripts/setup_environment.sh`**

- Create directory structure
- Install dependencies
- Verify GPU availability

**File: `scripts/download_pretrained_models.sh`**

- Download Grounding DINO base model
- Download SAM base model
- Verify checksums

### 11.2 Documentation

Create:

- `docs/CLI_USAGE.md`: Complete CLI guide
- `docs/ANNOTATION_FORMATS.md`: COCO format examples
- `docs/LORA_GUIDE.md`: LoRA fine-tuning guide
- `README.md`: Quick start guide

---

## Phase 12: End-to-End Testing (Days 35-36)

Test complete pipeline with your COCO dataset:

1. Validate dataset with auto-preprocessing
2. Train teachers with LoRA (both DINO and SAM)
3. Distill student (YOLOv8-seg for both mode)
4. Optimize (ONNX + INT8)
5. Evaluate all models
6. Verify student is prompt-free
7. Benchmark performance

---

## Critical Implementation Notes

### Loading LoRA-Adapted Teachers

**Always load fine-tuned teachers for distillation:**

```python
from peft import PeftModel

# Load base + LoRA adapters
base_dino = load_grounding_dino("pretrained/groundingdino.pth")
teacher_dino = PeftModel.from_pretrained(base_dino, "teachers/dino_lora/")

base_sam = load_sam("pretrained/sam_vit_h.pth")
teacher_sam = PeftModel.from_pretrained(base_sam, "teachers/sam_lora/")
```

### Ensure Prompt-Free Student

Student models MUST NOT accept prompts:

```python
class StudentModel(nn.Module):
    def forward(self, image):  # Only image input!
        return boxes, masks, class_ids
```

### Load Class Mapping from Config

NEVER hardcode class names:

```python
config = load_config('configs/distillation/kd_config.yaml')
class_mapping = config['class_mapping']  # From YAML
```

### Verify LoRA Freezing

Always verify only LoRA parameters are trainable:

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
assert trainable / total < 0.02, "Too many trainable params for LoRA!"
```

---

## Expected Deliverables

1. ✅ Complete directory structure
2. ✅ All core ML modules implemented
3. ✅ LoRA integration with PEFT for both DINO and SAM
4. ✅ Mode-aware pipeline (detection/segmentation/both)
5. ✅ Characteristic-based augmentation system
6. ✅ All CLI tools functional
7. ✅ Configuration templates for all stages
8. ✅ Unit and integration tests
9. ✅ Documentation and setup scripts
10. ✅ Verified prompt-free student models
11. ✅ Tested on your COCO dataset

## Timeline Summary

- **Week 1-2** (Days 1-10): Foundation + Data + Teacher Models
- **Week 3** (Days 11-17): Student Models + Distillation
- **Week 4** (Days 18-24): Optimization + Inference + Evaluation
- **Week 5** (Days 25-32): CLI Tools + Testing
- **Week 6** (Days 33-36): Documentation + End-to-End Testing

**Total: ~6 weeks for complete implementation**

### To-dos

- [ ] Create directory structure, requirements.txt, setup.py, and core utilities (config, logger, constants)
- [ ] Implement COCO dataset loader supporting all annotation modes (boxes/masks/both)
- [ ] Implement dataset validation with auto-preprocessing (bbox from masks, mode detection)
- [ ] Implement mode-aware preprocessing pipeline with model-specific strategies
- [ ] Implement characteristic-based augmentation system with albumentations backend
- [ ] Implement Grounding DINO model with PEFT/LoRA integration
- [ ] Implement SAM model with PEFT/LoRA integration
- [ ] Implement GroundedSAM wrapper for combined teacher model
- [ ] Implement TrainingManager and CheckpointManager for training infrastructure
- [ ] Implement TeacherTrainer with LoRA fine-tuning support
- [ ] Implement student models (YOLOv8, YOLOv8-seg, FastSAM) with prompt-free architecture
- [ ] Implement multi-component distillation loss functions
- [ ] Implement mode-aware DistillationTrainer with config-based class mapping
- [ ] Implement ONNX export, INT8 quantization, and TensorRT conversion
- [ ] Implement inference engines (PyTorch, ONNX, TensorRT)
- [ ] Implement evaluation metrics (mAP, IoU) and benchmarking tools
- [ ] Implement CLI for dataset validation with auto-preprocessing
- [ ] Implement CLI for teacher training with auto-mode detection
- [ ] Implement CLI for student distillation with LoRA-adapted teachers
- [ ] Implement CLI for model optimization (ONNX, quantization, TensorRT)
- [ ] Implement CLI tools for evaluation and inference
- [ ] Create all configuration templates for training, distillation, and optimization
- [ ] Write unit tests for data pipeline, models, LoRA, and prompt-free validation
- [ ] Write integration tests for training and optimization pipelines
- [ ] Create documentation and setup scripts
- [ ] Run end-to-end test with actual COCO dataset on all stages