# Technical Approach: Grounded SAM to Edge Deployment

## Problem Statement

**Challenge**: Grounded SAM requires prompts (text for Grounding DINO, boxes/points for SAM) and is too large for edge devices.

**Solution**: Fine-tune teacher models on domain-specific data with LoRA (memory-efficient), then distill knowledge into lightweight, **prompt-free** models that can run on edge devices.

## Approach: Fine-tune Then Distill (Data-Driven Pipeline)

```
Step 1: Inspect dataset (check for bbox/segmentation fields, extract class_mapping)
Step 2: Fine-tune teacher model(s) based on available annotations:
        - Has bbox → Fine-tune Grounding DINO
        - Has segmentation → Fine-tune SAM (auto-generate boxes if needed)
        - Has both → Fine-tune both teachers
Step 3: Select student model based on available annotations:
        - Has bbox only → YOLOv8 (detection)
        - Has masks only → FastSAM/MobileSAM (segmentation)
        - Has both → YOLOv8-seg (detection + segmentation)
Step 4: Distill knowledge from teacher to student (data-driven)
        → CRITICAL: Student learns to predict classes WITHOUT prompts
Step 5: Optimize for edge deployment (quantization, TensorRT)
Step 6: Deploy to edge devices
```

### Why This Approach?

**Advantages:**
- ✅ Teacher models adapted to your domain (higher accuracy)
- ✅ Student model is **completely prompt-free** (no text/box input needed)
- ✅ Better performance on specialized tasks (e.g., industrial inspection)
- ✅ End-to-end trainable pipeline
- ✅ Suitable for production deployment

**Requirements:**
- Labeled dataset in COCO format (with bboxes, masks, or both)
- GPU resources for training (RTX 3090/4090 or better)
- Time investment: 1-2 weeks for full pipeline

### Quick Start: Complete Pipeline in 4 Commands

```bash
# 1. Validate dataset (auto-generates configs)
python cli/validate_dataset.py --data data/raw/annotations.json --split train:0.7,val:0.15,test:0.15

# 2. Fine-tune teachers (auto-detects which models needed from data)
python cli/train_teacher.py --data data/raw/train.json --val data/raw/val.json --output experiments/exp1

# 3. Distill student (auto-selects student model, auto-fills class_mapping)
python cli/train_student.py --data data/raw/train.json --teacher-dir experiments/exp1 --output experiments/exp1/student

# 4. Optimize for edge (quantize + export)
python cli/optimize_model.py --model experiments/exp1/student/best.pt --format onnx tensorrt --quantize int8

# Done! Deploy experiments/exp1/student/optimized/model_int8.engine to Jetson.
```

**That's it. Four commands. No config editing. No manual setup.**

## 🎯 Data-Driven Pipeline Design

**Core Principle**: The data structure itself determines pipeline behavior. No mode enums, no state files - just inspect the data and load corresponding models.

### Annotation Detection (Automatic, Stateless)

```python
def inspect_dataset(coco_data):
    """Inspect dataset to determine what annotations are available."""
    annotations = coco_data['annotations']
    
    return {
        'has_boxes': any('bbox' in ann for ann in annotations),
        'has_masks': any('segmentation' in ann for ann in annotations),
        'num_classes': len(coco_data['categories']),
        'class_mapping': {cat['id']: cat['name'] for cat in coco_data['categories']}
    }

# Usage: Data structure drives model selection directly
dataset_info = inspect_dataset(coco_data)

# Load models based on what data is available
models = []
if dataset_info['has_boxes']:
    models.append(GroundingDINO(...))  # Load DINO for box annotations
if dataset_info['has_masks']:
    models.append(SAM(...))             # Load SAM for mask annotations

# No enums, no lookup tables, no extra state - just data-driven logic
```

### Pipeline Adaptation Strategy (Data-Driven)

The platform uses **data structure inspection** to determine pipeline behavior:

```python
# ml_engine/training/pipeline.py

class TrainingPipeline:
    def __init__(self, train_data_path, config_defaults):
        # Inspect dataset once at initialization
        coco_data = load_json(train_data_path)
        self.dataset_info = inspect_dataset(coco_data)
        
        # Auto-fill config from defaults + dataset info
        self.config = self._build_config(config_defaults, self.dataset_info)
        
        # Load only required models based on available data
        self.models = self._load_models(self.dataset_info)
    
    def _load_models(self, dataset_info):
        """Load models based on what annotations are present in data."""
        models = {}
        
        # Data structure determines model loading - no mode enum!
        if dataset_info['has_boxes']:
            models['grounding_dino'] = self._load_grounding_dino()
        if dataset_info['has_masks']:
            models['sam'] = self._load_sam()
        
        return models
    
    def training_step(self, batch):
        """Training loop automatically adapts to available models."""
        # Run only loaded models - no if-else for modes
        teacher_outputs = {}
        
        if 'grounding_dino' in self.models:
            teacher_outputs['boxes'] = self.models['grounding_dino'](batch['image'])
        
        if 'sam' in self.models:
            box_prompts = teacher_outputs.get('boxes') or self._gen_boxes(batch['masks'])
            teacher_outputs['masks'] = self.models['sam'](batch['image'], box_prompts)
        
        # Loss automatically computed based on available outputs
        return self.compute_loss(student_pred, teacher_outputs)
```

### Benefits of Data-Driven Design

1. **No Extra State**: No mode enums, no .mode_config.json files
2. **Self-Describing Data**: COCO structure already tells you what's available
3. **Extensible**: Add keypoints? Just check `'keypoints' in ann` and load pose model
4. **Simple**: Data presence → load corresponding model. That's it.
5. **No Lookup Tables**: No PIPELINE_CONFIG dict needed

## 🔑 CRITICAL: How Student Becomes Prompt-Free

This is the **KEY INNOVATION** of the entire pipeline. Let me explain in detail:

### Teacher Model (Grounded SAM) - Sequential Pipeline with Prompts
```python
# Teacher is a TWO-STAGE SEQUENTIAL pipeline that needs prompts
# Stage 1: Grounding DINO (Text → Boxes)
# text_prompt loaded from config['class_mapping'] (e.g., "ear of bag")
input_stage1: image + text_prompt(class_name)
grounding_dino_output: boxes, confidence_scores
↓
# Stage 2: SAM (Boxes → Masks)  
input_stage2: image + boxes_from_stage1  # Boxes are prompts for SAM
sam_output: segmentation_masks
↓
# Combined output
output: boxes (from DINO), masks (from SAM), labels (class_id from prompt)

# Problem: 
# 1. Requires text prompt for DINO at inference time
# 2. Two-stage pipeline is slow (150ms total)
# 3. Cannot deploy to edge - needs prompts every time!
```

### Student Model (YOLOv8-seg) - Single-Stage End-to-End, Prompt-Free
```python
# Student is a SINGLE-STAGE END-TO-END model - NO prompts needed!
input: image ONLY (no text, no boxes, no prompts)
↓
Single forward pass through neural network
↓
output: boxes, masks, class_ids (all predicted simultaneously)

# Solution: 
# 1. No prompts needed - class info embedded in model weights!
# 2. Single-stage - much faster (8ms vs 150ms)
# 3. Perfect for edge deployment
```

### How Distillation Makes Student Prompt-Free

**The Magic: Class Information is "Baked Into" the Student**

```python
# Training Phase: Teacher provides class-specific knowledge

# Step 1: Teacher inference WITH prompts (SEQUENTIAL PIPELINE)
# Load class mapping from config (NOT hardcoded!)
class_mapping = config['class_mapping']  # e.g., {0: "ear", 1: "defect", 2: "label"}

for batch in training_data:
    # Teacher uses prompts to generate targets
    teacher_outputs = []
    for class_id, class_name in class_mapping.items():
        # === Teacher Stage 1: Grounding DINO (Text → Boxes) ===
        dino_boxes = teacher.grounding_dino.predict(
            image=batch.images,
            text_prompt=class_name  # ← Text prompt for DINO
        )
        
        # === Teacher Stage 2: SAM (Boxes → Masks) ===
        sam_masks = teacher.sam.predict(
            image=batch.images,
            box_prompts=dino_boxes  # ← Boxes as prompts for SAM
        )
        
        teacher_outputs.append({
            'class_id': class_id,
            'boxes': dino_boxes,           # From Grounding DINO
            'masks': sam_masks,            # From SAM
            'dino_features': teacher.grounding_dino.get_features(),
            'sam_features': teacher.sam.get_features(),
            'logits': teacher.grounding_dino.get_logits()
        })
    
    # Step 2: Student learns to predict SAME outputs WITHOUT prompts
    # Student is SINGLE-STAGE: direct prediction in one forward pass
    student_predictions = student(batch.images)  # ← NO prompts! Single pass!
    
    # Step 3: Distillation loss aligns student with teacher
    loss = (
        # Student learns to predict same boxes as teacher (from DINO)
        detection_loss(student_predictions.boxes, teacher_outputs.boxes) +
        
        # Student learns to predict same masks as teacher (from SAM)
        segmentation_loss(student_predictions.masks, teacher_outputs.masks) +
        
        # Student learns to predict same class IDs as teacher
        classification_loss(student_predictions.class_ids, teacher_outputs.class_ids) +
        
        # Feature-level knowledge transfer (from both DINO and SAM)
        feature_distillation_loss(student_features, teacher_dino_features, teacher_sam_features) +
        
        # Logit-level knowledge transfer (from DINO)
        kl_divergence(student_logits, teacher_logits)
    )
```

### Key Insight: Fixed Class Mapping

```python
# During fine-tuning & distillation, we establish FIXED mapping:
# Loaded from config file at training time
class_mapping = config['class_mapping']  # e.g., {0: "ear of bag", 1: "defect", 2: "label"}
# Student class 0 ← Teacher with prompt class_mapping[0]
# Student class 1 ← Teacher with prompt class_mapping[1]
# Student class 2 ← Teacher with prompt class_mapping[2]

# After training, student ONLY knows these classes
# No prompt needed - class info is in the weights!

# Inference (Edge Device):
image = load_image("test.jpg")
predictions = student(image)  # Direct prediction!
# Output: class_id=0, boxes=[...], masks=[...]
# (can map back: class_names[class_id] to get human-readable name)
```

### Architecture Comparison

| Aspect | Teacher (Grounded SAM) | Student (YOLOv8-seg) |
|--------|------------------------|----------------------|
| **Architecture** | Two-stage sequential (DINO→SAM) | Single-stage end-to-end |
| **Input** | Image + Text Prompt (DINO) + Boxes (SAM) | Image ONLY |
| **Classes** | Open-vocabulary (any text) | Fixed classes (0, 1, 2, ...) |
| **Flexibility** | Can detect anything | Only trained classes |
| **Inference** | Prompts required every time | No prompt needed |
| **Size** | 2.9 GB (both models) | 11.8 MB → 3MB |
| **Speed** | 150ms (GPU, sequential) | 8ms (GPU, single pass) |
| **Use Case** | Research, exploration | Production, edge deployment |

### Why This Works

1. **Training Time**: Teacher uses prompts to teach student what each class looks like
2. **Knowledge Transfer**: Student learns visual patterns associated with each class ID
3. **Weight Embedding**: Class-specific knowledge gets embedded in student's weights
4. **Inference Time**: Student directly outputs class predictions from visual features

### Example: Industrial Bag Inspection

```python
# Training: Teacher teaches student with prompts
# Class mapping loaded from config file
classes = load_config('distillation_config.yaml')['class_mapping']
# e.g., {0: "ear of flexible bag", 1: "surface defect", 2: "printed label", ...}

# After distillation: Student knows these classes by heart
# Deployment: No prompts needed!
edge_model = load_student_model("yolov8s_seg.onnx")
result = edge_model(camera_image)  # Instant classification!
# Returns: class_ids (integers), can map back to names using classes dict
```

## ⚙️ Configuration Management Strategy

### Auto-Generated Configs (Simplified Approach)

The platform uses **automatic configuration generation** from your dataset:

```
configs/
├── defaults/                           # Default configs with sensible values (committed to git)
│   ├── teacher_grounding_dino_lora.yaml
│   ├── teacher_sam_lora.yaml
│   ├── student_yolov8_seg.yaml
│   └── distillation.yaml
│
└── experiments/                        # Auto-generated per-experiment (NOT in git)
    └── {experiment_name}/              # Generated from: data + defaults + CLI overrides
        ├── teacher_config.yaml         # Auto-filled: num_classes, class_names from COCO
        ├── distillation_config.yaml    # Auto-filled: class_mapping from COCO
        └── metadata.json               # Dataset info, timestamp, git commit
```

### Simplified Workflow

**No Manual Config Editing Required!** The platform auto-generates everything from your data:

```bash
# Single command trains teacher models
# Platform automatically:
# 1. Inspects COCO dataset (gets num_classes, class_names, annotation type)
# 2. Generates config from defaults + dataset info
# 3. Starts training

python cli/train_teacher.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --output-dir experiments/my_experiment \
    --gpu 0

# Optional: Override specific hyperparameters via CLI
python cli/train_teacher.py \
    --data data/raw/train.json \
    --teacher.batch_size 16 \          # Override default
    --teacher.epochs 100 \              # Override default
    --output-dir experiments/my_experiment
```

### How Auto-Generation Works

```python
# Platform automatically:
# 1. Read COCO dataset
coco_data = load_json("train.json")
num_classes = len(coco_data['categories'])
class_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 2. Detect annotation type
has_boxes = any('bbox' in ann for ann in coco_data['annotations'])
has_masks = any('segmentation' in ann for ann in coco_data['annotations'])

# 3. Load default config
default_config = load_yaml("configs/defaults/teacher_grounding_dino_lora.yaml")

# 4. Auto-fill dataset-specific values
config = {
    **default_config,
    'num_classes': num_classes,
    'class_names': list(class_names.values()),
    'train_path': 'data/raw/train.json',
    'val_path': 'data/raw/val.json',
}

# 5. Apply CLI overrides
config.update(cli_args)

# 6. Save for reproducibility
save_yaml(f"experiments/{experiment_name}/teacher_config.yaml", config)
```

### Benefits

| Aspect | Old (Manual) | New (Auto-Generated) |
|--------|-------------|---------------------|
| **User Actions** | Copy template, edit file, fix typos | Just point to data |
| **Error Prone** | ✗ Easy to forget updating class_names | ✓ Automatic, no errors |
| **Reproducible** | ✗ Manual edits not tracked | ✓ All configs saved per experiment |
| **Fast** | ✗ 5-10 minutes setup | ✓ Instant |

### Example .gitignore

```gitignore
# configs/.gitignore
# Auto-generated experiment configs
experiments/*/

# Keep defaults
!defaults/*.yaml
```

### CLI Override System

Users can override any config value via CLI without editing files:

```bash
# Override LoRA rank
python cli/train_teacher.py --data train.json --lora.r 32

# Override training params
python cli/train_teacher.py --data train.json \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 2e-4

# Override multiple nested values
python cli/train_teacher.py --data train.json \
    --augmentation.intensity high \
    --training.optimizer SGD
```

## Detailed Technical Workflow

### Stage 0: Image Preprocessing & Normalization Standards

**🔑 CRITICAL: Different Models Use Different Preprocessing!**

Each model in our pipeline uses different normalization and resizing strategies. Understanding and correctly implementing these is **essential** for training success.

#### Preprocessing Configuration by Model

```python
# configs/defaults/preprocessing.yaml
preprocessing:
  # Grounding DINO preprocessing (Swin Transformer backbone)
  grounding_dino:
    input_size: 
      min_size: 800        # Resize shortest side to 800
      max_size: 1333       # Limit longest side to 1333
    normalization:
      mean: [0.485, 0.456, 0.406]  # ImageNet statistics
      std: [0.229, 0.224, 0.225]
      pixel_range: [0, 1]   # Normalize to [0,1] first, then standardize
    resize_mode: "keep_aspect_ratio"  # Maintain aspect ratio, pad to max_size
    padding_value: 0        # Zero padding
    pixel_format: "RGB"     # NOT BGR!
  
  # SAM preprocessing (ViT backbone)
  # ⚠️ CRITICAL: SAM uses DIFFERENT normalization than ImageNet!
  sam:
    input_size:
      height: 1024          # SAM requires square input
      width: 1024
    normalization:
      mean: [123.675, 116.28, 103.53]   # ← NOT ImageNet! Pixel-range values
      std: [58.395, 57.12, 57.375]       # ← Different from ImageNet!
      pixel_range: [0, 255]  # SAM expects [0,255] range, NOT [0,1]
    resize_mode: "resize_longest_side"   # SAM-specific: resize longest to 1024, pad rest
    padding_value: 0
    pixel_format: "RGB"
  
  # YOLOv8 student preprocessing
  student:
    input_size: 640         # Square input
    normalization:
      mean: [0.0, 0.0, 0.0]  # YOLO doesn't standardize
      std: [1.0, 1.0, 1.0]   # Just normalizes to [0,1]
      pixel_range: [0, 1]    # [0,1] range
    resize_mode: "letterbox"  # YOLO-specific: maintain aspect, gray padding
    padding_value: 114      # Gray padding (R=G=B=114)
    pixel_format: "RGB"
```

#### Why These Differences Matter

**Problem**: If you use wrong normalization, model performance drops dramatically (20-40% accuracy loss!).

**Example of what goes wrong**:
```python
# ❌ WRONG: Using ImageNet normalization for SAM
image = image / 255.0  # Convert to [0,1]
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
# Result: SAM gets completely wrong input distribution
# Expected mAP: 0.90 → Actual mAP: 0.52 (42% drop!)

# ✅ CORRECT: Using SAM's expected normalization
image = image  # Keep in [0,255] range
image = (image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
# Result: SAM performs as expected
```

#### Implementation: Data-Driven Preprocessing Pipeline

**Simplified Design Principle**: Preprocessing adapts based on which models are actually loaded (determined by data inspection).

```python
# ml_engine/data/preprocessing.py

from typing import Dict, Tuple, List
import torch
import torchvision.transforms.functional as F
from PIL import Image
import yaml

class MultiModelPreprocessor:
    """
    Preprocessor that handles multiple models with different input requirements.
    
    Architecture:
    1. Initialize with list of active model names (from data inspection)
    2. Each model gets its own preprocessor with correct settings
    3. Single call preprocesses for all models
    
    Why simpler?
    - No mode enums - just pass list of model names
    - No stage enums - models list already tells us what's needed
    - Data-driven - if you loaded DINO, preprocess for DINO
    """
    
    def __init__(self, active_models: List[str], config_path: str):
        """
        Args:
            active_models: List of model names to preprocess for (e.g., ['grounding_dino', 'sam'])
            config_path: Path to preprocessing config YAML
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['preprocessing']
        
        # Initialize preprocessors only for active models
        self.preprocessors = {}
        for model_name in active_models:
            if model_name in self.config:
                self.preprocessors[model_name] = SingleModelPreprocessor(
                    model_name, self.config[model_name]
                )
    
    def preprocess_batch(self, image: Image.Image) -> Dict[str, Tuple[torch.Tensor, Dict]]:
        """
        Preprocess image for all active models.
        
        Returns:
            Dict mapping model_name → (preprocessed_image, metadata)
        """
        results = {}
        for model_name, preprocessor in self.preprocessors.items():
            results[model_name] = preprocessor.preprocess(image)
        return results
    
    def preprocess_for_model(self, image: Image.Image, model_name: str) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image for a specific model."""
        if model_name not in self.preprocessors:
            raise ValueError(f"Model {model_name} not loaded. "
                           f"Available: {list(self.preprocessors.keys())}")
        return self.preprocessors[model_name].preprocess(image)


class SingleModelPreprocessor:
    """
    Handles preprocessing for a single model.
    Model-specific preprocessing encapsulated for reusability.
    """
    
    def __init__(self, model_name: str, model_config: Dict):
        """
        Args:
            model_name: Name of the model (for logging)
            model_config: Preprocessing config for this specific model
        """
        self.model_name = model_name
        self.config = model_config
        
        # Pre-compute tensors for normalization (efficiency)
        norm_cfg = self.config['normalization']
        self.mean = torch.tensor(norm_cfg['mean']).view(3, 1, 1)
        self.std = torch.tensor(norm_cfg['std']).view(3, 1, 1)
    
    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image according to model-specific requirements."""
        orig_size = image.size  # (width, height)
        
        # Step 1: Resize
        image, scale_factor = self._resize(image)
        
        # Step 2: Convert to tensor
        image_tensor = F.to_tensor(image)  # [0,1] range, [C,H,W]
        
        # Step 3: Normalize
        image_tensor = self._normalize(image_tensor)
        
        # Step 4: Pad
        image_tensor, pad_info = self._pad(image_tensor)
        
        metadata = {
            'original_size': orig_size,
            'scale_factor': scale_factor,
            'padding': pad_info,
            'final_size': image_tensor.shape[-2:],
            'model_name': self.model_name
        }
        
        return image_tensor, metadata
    
    def _resize(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """Apply model-specific resize strategy from config."""
        cfg = self.config['input_size']
        mode = self.config['resize_mode']
        w, h = image.size
        
        # Use strategy pattern based on config (no hardcoded if-else!)
        resize_strategies = {
            'keep_aspect_ratio': self._resize_keep_aspect,
            'resize_longest_side': self._resize_longest_side,
            'letterbox': self._resize_letterbox,
        }
        
        if mode not in resize_strategies:
            raise ValueError(f"Unknown resize mode: {mode}")
        
        return resize_strategies[mode](image, cfg, w, h)
    
    def _resize_keep_aspect(self, image, cfg, w, h):
        """Grounding DINO strategy: min_size on short side, limit max_size."""
        min_size, max_size = cfg['min_size'], cfg['max_size']
        scale = min_size / min(w, h)
        
        if max(w, h) * scale > max_size:
            scale = max_size / max(w, h)
        
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR), scale
    
    def _resize_longest_side(self, image, cfg, w, h):
        """SAM strategy: resize longest side to target."""
        target_size = cfg['height']
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR), scale
    
    def _resize_letterbox(self, image, cfg, w, h):
        """YOLO strategy: maintain aspect ratio, pad to square."""
        target_size = cfg
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR), scale
    
    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Apply model-specific normalization from config."""
        pixel_range = self.config['normalization']['pixel_range']
        
        # Adjust pixel range if needed
        if pixel_range == [0, 255] and image.max() <= 1.0:
            image = image * 255.0
        elif pixel_range == [0, 1] and image.max() > 1.0:
            image = image / 255.0
        
        # Apply standardization
        return (image - self.mean) / (self.std + 1e-8)  # Add epsilon for stability
    
    def _pad(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Add padding to meet model's input size requirements."""
        cfg = self.config['input_size']
        pad_value = self.config['padding_value']
        mode = self.config['resize_mode']
        
        if mode in ["resize_longest_side", "letterbox"]:
            _, h, w = image.shape
            target = cfg['height'] if isinstance(cfg, dict) else cfg
            
            pad_h, pad_w = target - h, target - w
            padding = (pad_w // 2, pad_w - pad_w // 2, 
                      pad_h // 2, pad_h - pad_h // 2)
            
            image = F.pad(image, padding, fill=pad_value)
            return image, {'padding': padding, 'target_size': target}
        
        return image, {'padding': (0, 0, 0, 0), 'target_size': image.shape[-2:]}


# ============================================================================
# USAGE EXAMPLES: Data-driven preprocessing
# ============================================================================

# Example 1: Teacher fine-tuning (boxes only in dataset)
dataset_info = inspect_dataset(coco_data)  # Returns: {has_boxes: True, has_masks: False}

# Load only required models
active_models = []
if dataset_info['has_boxes']:
    active_models.append('grounding_dino')
if dataset_info['has_masks']:
    active_models.append('sam')
# Result: active_models = ['grounding_dino']  ← Only DINO needed

preprocessor = MultiModelPreprocessor(
    active_models=active_models,
    config_path='configs/defaults/preprocessing.yaml'
)

# Preprocess image for all active models
preprocessed = preprocessor.preprocess_batch(pil_image)
# Returns: {'grounding_dino': (dino_tensor, dino_meta)}
# SAM preprocessing NOT performed - not in active_models!


# Example 2: Student distillation (boxes + masks in dataset)
dataset_info = inspect_dataset(coco_data)  # Returns: {has_boxes: True, has_masks: True}

# Determine student model from data
student_model = 'yolov8_seg' if (dataset_info['has_boxes'] and dataset_info['has_masks']) \
                else 'yolov8' if dataset_info['has_boxes'] \
                else 'fastsam'

# Load all models needed for distillation
active_models = ['grounding_dino', 'sam', student_model]

preprocessor = MultiModelPreprocessor(
    active_models=active_models,
    config_path='configs/defaults/preprocessing.yaml'
)

# Preprocess for all models
preprocessed = preprocessor.preprocess_batch(pil_image)
# Returns: {
#     'grounding_dino': (dino_tensor, dino_meta),    # 800×1333
#     'sam': (sam_tensor, sam_meta),                  # 1024×1024
#     'yolov8_seg': (student_tensor, student_meta)   # 640×640
# }


# Example 3: Inference (only student needed)
# Student model name stored during training
preprocessor = MultiModelPreprocessor(
    active_models=['yolov8_seg'],  # Just the student model
    config_path='configs/defaults/preprocessing.yaml'
)

image_tensor, metadata = preprocessor.preprocess_for_model(pil_image, 'yolov8_seg')
```

#### Resolution Mismatch During Distillation - Solved by Multi-Model Preprocessing

**Challenge**: Teacher uses 800×1333 (DINO) and 1024×1024 (SAM), but student uses 640×640.

**Solution**: Multi-model preprocessor handles each model's requirements automatically.

```python
# During distillation training loop
class DistillationTrainer:
    def __init__(self, teacher_models, student_model, config_path):
        # Determine active models from loaded teachers + student
        active_models = list(teacher_models.keys()) + [student_model]
        
        # Initialize preprocessor for all active models
        self.preprocessor = MultiModelPreprocessor(
            active_models=active_models,
            config_path=config_path
        )
        # If teacher_models = {'grounding_dino': <model>, 'sam': <model>}
        # and student_model = 'yolov8_seg'
        # Creates preprocessors for: grounding_dino, sam, yolov8_seg
        
    def training_step(self, batch):
        original_image = batch['image']  # PIL Image
        
        # Preprocess for ALL active models in one call!
        preprocessed = self.preprocessor.preprocess_batch(original_image)
        # Returns: {
        #     'grounding_dino': (dino_image, dino_meta),    # 800×1333
        #     'sam': (sam_image, sam_meta),                  # 1024×1024
        #     'yolov8_seg': (student_image, student_meta)   # 640×640
        # }
        
        # Teacher inference (frozen)
        with torch.no_grad():
            teacher_boxes = self.teachers['grounding_dino'](
                preprocessed['grounding_dino'][0],  # Correct size
                text_prompts=self.class_mapping
            )
            teacher_masks = self.teachers['sam'](
                preprocessed['sam'][0],  # Correct size
                box_prompts=teacher_boxes
            )
        
        # Student inference (trainable)
        student_output = self.student(
            preprocessed['yolov8_seg'][0]  # Correct size
        )
        
        # Loss computation in normalized coordinates
        loss = self.compute_loss(student_output, teacher_boxes, teacher_masks)
        return loss
```

**Why this works**:

1. **Data inspection → Model loading**: Check data, load corresponding models
2. **Model list → Preprocessing**: Initialize preprocessor with model names  
3. **Single method call**: `preprocess_batch()` handles all models
4. **Normalized coordinates**: Boxes in `[0,1]` range, resolution-independent
5. **Each model gets correct input**: No accuracy loss

**Simplified Approach**:

```python
# ✅ SIMPLE: Data-driven
dataset_info = inspect_dataset(coco_data)  # Check data structure
models = ['grounding_dino'] if dataset_info['has_boxes'] else []
models += ['sam'] if dataset_info['has_masks'] else []

preprocessor = MultiModelPreprocessor(active_models=models, config_path=config)
preprocessed = preprocessor.preprocess_batch(image)
# Done! Each model in 'models' list gets preprocessed correctly
```

#### Best Practices

1. **Always use config files** for preprocessing parameters (no hardcoding!)
2. **Verify preprocessing** with a few images before full training
3. **Log preprocessing params** in TensorBoard for reproducibility
4. **Test with known images** (e.g., from original papers) to verify correctness
5. **Document pixel range assumptions** clearly in config files

---

### Stage 1: Data Preparation

**Input Format: COCO JSON (Flexible Annotations)**

**📌 IMPORTANT: Single JSON File for ALL Images**

The COCO format uses **ONE JSON file** that contains annotations for **ALL images** in your dataset:

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg", "width": 1920, "height": 1080},
    {"id": 2, "file_name": "img002.jpg", "width": 1920, "height": 1080},
    {"id": 3, "file_name": "img003.jpg", "width": 1920, "height": 1080},
    // ... all images listed here
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [...], "segmentation": [...]},
    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [...], "segmentation": [...]},  // Same image
    {"id": 3, "image_id": 2, "category_id": 0, "bbox": [...], "segmentation": [...]},  // Different image
    // ... all annotations for all images listed here
  ],
  "categories": [
    {"id": 0, "name": "ear of bag"},
    {"id": 1, "name": "defect"},
    {"id": 2, "name": "label"}
  ]
}
```

**Key Points:**
- ✅ **ONE** `train.json` file for all training images
- ✅ **ONE** `val.json` file for all validation images
- ✅ Annotations reference images via `image_id`
- ❌ **NOT** one JSON file per image (that's inefficient and non-standard)

**Why Single File?**
1. ✅ Standard COCO format (used by all tools)
2. ✅ Efficient loading (read once, not thousands of times)
3. ✅ Easy to split train/val (just modify one JSON file)
4. ✅ Simpler dataset management

**Frontend Annotation Tool Responsibility:**
```
User annotates images in UI (multiple images)
  ↓
Export button clicked
  ↓
Frontend generates ONE JSON file:
  {
    "images": [...],      // All annotated images
    "annotations": [...], // All annotations from all images
    "categories": [...]   // Class definitions
  }
  ↓
Save as: annotations.json (or train.json, val.json)
```

**Typical Dataset Structure:**
```
data/raw/
├── images/                 # All image files
│   ├── img001.jpg
│   ├── img002.jpg
│   ├── img003.jpg
│   ├── ...
│   └── img1000.jpg
│
├── train.json             # ONE JSON for 800 training images
└── val.json               # ONE JSON for 200 validation images

# train.json structure:
{
  "images": [              # 800 entries
    {"id": 1, "file_name": "img001.jpg", ...},
    {"id": 2, "file_name": "img002.jpg", ...},
    ...
  ],
  "annotations": [         # ~3000 entries (avg 3-4 objects per image)
    {"id": 1, "image_id": 1, ...},
    {"id": 2, "image_id": 1, ...},
    {"id": 3, "image_id": 2, ...},
    ...
  ],
  "categories": [...]
}

# val.json structure: Same format, just 200 images
```

**📌 IMPORTANT: Who Handles Train/Val/Test Splitting?**

**Answer: Platform (Backend), NOT Frontend!**

**Frontend Responsibility (Simple):**
```javascript
// Frontend annotation tool - ONLY exports all annotations
function exportAnnotations() {
  const allAnnotations = getAllAnnotations();
  
  // Export ONE file with ALL annotations
  downloadJSON(allAnnotations, "annotations.json");
}
// That's it! No splitting logic needed in frontend.
```

**Platform Responsibility (Smart Splitting):**
```bash
# Platform handles splitting with proper ML practices
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.7,val:0.15,test:0.15 \
    --stratify \           # Maintain class distribution
    --random-seed 42 \     # Reproducible splits
    --output-dir data/raw/

# Creates:
# - data/raw/train.json (70% of data, ~700 images)
# - data/raw/val.json   (15% of data, ~150 images)
# - data/raw/test.json  (15% of data, ~150 images)
```

**Why Platform Should Handle Splitting:**

| Aspect | Frontend Splitting | Platform Splitting |
|--------|-------------------|-------------------|
| **Stratification** | ❌ Hard to implement | ✅ Maintains class distribution |
| **Reproducibility** | ❌ No random seed control | ✅ Fixed seed = same split |
| **Consistency** | ❌ Each user splits differently | ✅ Everyone uses same logic |
| **ML Best Practices** | ❌ Frontend devs not ML experts | ✅ Proper validation strategies |
| **Frontend Complexity** | ❌ More complex code | ✅ Simple export only |
| **Re-splitting** | ❌ Must re-annotate | ✅ Just change ratio/seed |

**Platform Splitting Features:**

```python
# In ml_engine/data/preprocessing.py

def split_dataset(
    coco_json_path,
    splits={'train': 0.7, 'val': 0.15, 'test': 0.15},
    stratify=True,
    random_seed=42,
    output_dir='data/raw/'
):
    """
    Split COCO dataset into train/val/test with proper ML practices.
    
    Features:
    - Stratified splitting (maintains class distribution)
    - Reproducible (fixed random seed)
    - Validates split ratios
    - Checks minimum samples per class in each split
    """
    # Load all annotations
    coco_data = load_json(coco_json_path)
    
    # Stratify by category to maintain class distribution
    if stratify:
        splits = stratified_split(
            coco_data, 
            splits=splits, 
            random_seed=random_seed
        )
    else:
        splits = random_split(
            coco_data,
            splits=splits,
            random_seed=random_seed
        )
    
    # Validate each split
    for split_name, split_data in splits.items():
        validate_split(split_data)  # Check minimum samples
        save_json(split_data, f"{output_dir}/{split_name}.json")
    
    return splits
```

**CLI Usage Examples:**

```bash
# Basic split (70/15/15)
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.7,val:0.15,test:0.15

# Simple split (80/20, no test set)
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.8,val:0.2

# Custom split with reproducibility
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.6,val:0.2,test:0.2 \
    --random-seed 12345 \
    --stratify

# Re-split with different ratio (no re-annotation needed!)
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.85,val:0.15 \  # Different ratio
    --random-seed 42
```

**Output Example:**

```bash
$ python cli/validate_dataset.py --data annotations.json --split train:0.7,val:0.15,test:0.15

Loading dataset... ✓ (1000 images, 3450 annotations)
Detecting annotation mode... ✓ DETECTION_AND_SEGMENTATION

Splitting dataset (stratified, seed=42):
  ├─ Train: 700 images (70%), 2415 annotations
  │   └─ Class distribution: ear:450 (63%), defect:350 (49%), label:200 (28%)
  ├─ Val: 150 images (15%), 518 annotations
  │   └─ Class distribution: ear:96 (64%), defect:75 (50%), label:43 (29%)
  └─ Test: 150 images (15%), 517 annotations
      └─ Class distribution: ear:95 (63%), defect:74 (49%), label:42 (28%)

✓ Class distributions maintained across all splits
✓ All splits have minimum 40 samples per class

Saved:
  ├─ data/raw/train.json
  ├─ data/raw/val.json
  └─ data/raw/test.json

Dataset ready for training!
```

**Summary for Frontend Team:**

Your annotation tool should:
1. ✅ Export **ONE** JSON file with **ALL** annotated images
2. ✅ Use standard COCO format: `{images: [...], annotations: [...], categories: [...]}`
3. ✅ Include `segmentation` field (for masks) - our platform auto-generates `bbox` if missing
4. ❌ **DO NOT** implement train/val/test splitting logic
5. ❌ **DO NOT** export separate files per image
6. ❌ **DO NOT** compute bounding boxes from masks (platform does this)

**Simple Frontend Export Function:**
```javascript
function exportAnnotations() {
  const coco = {
    images: getAllImages(),           // All image metadata
    annotations: getAllAnnotations(), // All annotations from all images
    categories: getCategories()       // Class definitions
  };
  
  downloadJSON(coco, "annotations.json");  // Done!
}
```

The platform handles:
- ✅ bbox/area auto-generation from masks
- ✅ Train/val/test splitting (stratified, reproducible)
- ✅ Annotation mode detection
- ✅ Data validation
- ✅ All ML preprocessing

The platform supports three annotation modes:

#### Mode 1: Detection Only (Boxes)
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],  // Required
      "area": 1234,
      "iscrowd": 0
      // NO segmentation field
    }
  ],
  "categories": [
    {"id": 0, "name": "ear of bag"},
    {"id": 1, "name": "defect"},
    {"id": 2, "name": "label"}
  ]
}
```
**Use case:** When you only have bounding box annotations

#### Mode 2: Segmentation Only (Masks)
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[x1,y1,x2,y2,...]],  // Polygon or RLE (from frontend tool)
      "bbox": [x, y, width, height],  // Optional: can be auto-generated by platform
      "area": 1234,  // Optional: can be auto-computed
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[x1,y1,x2,y2,...]],  // Second object (from frontend tool)
      "bbox": [x, y, width, height],  // Optional: can be auto-generated by platform
      "area": 2345,  // Optional: can be auto-computed
      "iscrowd": 0
    }
  ]
}
```
**Use case:** When you only have segmentation masks

**📌 Important Design Decision - Who Generates Boxes from Masks?**

**Answer: The Platform (Backend), NOT the Frontend Annotation Tool**

**Why?**
1. **Separation of Concerns**: Frontend tool's job is to capture user annotations, not ML preprocessing
2. **Flexibility**: Accept COCO data from ANY source (labelme, CVAT, custom tools, etc.)
3. **Consistency**: Ensure boxes are computed using the same algorithm across all datasets
4. **Standard Practice**: Many annotation tools (labelme, Supervisely) only save masks, boxes are optional

**Frontend Annotation Tool Responsibility:**
```json
// Frontend ONLY needs to export:
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[x1,y1,x2,y2,...]],  // ✅ Required: User-drawn mask
      "iscrowd": 0
      // bbox: NOT required from frontend!
      // area: NOT required from frontend!
    }
  ]
}
```

**Platform Preprocessing (Automatic):**
```python
# In ml_engine/data/preprocessing.py or validators.py

def preprocess_coco_dataset(coco_json):
    """
    Automatically generate missing bbox and area fields from segmentation masks.
    This runs during dataset validation (python cli/validate_dataset.py).
    """
    for ann in coco_json['annotations']:
        # If annotation has segmentation but no bbox, generate it
        if 'segmentation' in ann and 'bbox' not in ann:
            ann['bbox'] = compute_bbox_from_mask(ann['segmentation'])
            print(f"✓ Generated bbox for annotation {ann['id']}")
        
        # If no area, compute it
        if 'segmentation' in ann and 'area' not in ann:
            ann['area'] = compute_area_from_mask(ann['segmentation'])
    
    return coco_json

def compute_bbox_from_mask(segmentation):
    """
    Compute tight bounding box from polygon or RLE mask.
    Returns: [x_min, y_min, width, height] in COCO format
    """
    if isinstance(segmentation, list):  # Polygon format
        # Flatten polygon points
        all_x = [segmentation[i] for i in range(0, len(segmentation), 2)]
        all_y = [segmentation[i] for i in range(1, len(segmentation), 2)]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    elif isinstance(segmentation, dict):  # RLE format
        # Decode RLE and compute bbox
        mask = decode_rle(segmentation)
        y_indices, x_indices = np.where(mask > 0)
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
```

**When Platform Auto-Generates Boxes:**
- During `python cli/validate_dataset.py` (first step of pipeline)
- Happens automatically if `bbox` field is missing but `segmentation` exists
- Transparent to user - they just get a validated, complete dataset

**Complete Workflow: Frontend → Platform**

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend Annotation Tool (Your Colleague's Responsibility)  │
│                                                              │
│ User draws segmentation masks in UI                         │
│     ↓                                                        │
│ Export to COCO JSON:                                        │
│   {                                                          │
│     "annotations": [                                         │
│       {                                                      │
│         "segmentation": [[x1,y1,x2,y2,...]],  ✅ Required  │
│         "category_id": 0,                     ✅ Required  │
│         // bbox: ❌ NOT required!                           │
│         // area: ❌ NOT required!                           │
│       }                                                      │
│     ]                                                        │
│   }                                                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Platform Data Pipeline (Your ML Platform's Responsibility)  │
│                                                              │
│ Step 1: python cli/validate_dataset.py                     │
│   ├─ Load COCO JSON                                        │
│   ├─ Auto-generate bbox from segmentation ✅ (automatic!)  │
│   ├─ Auto-compute area ✅ (automatic!)                     │
│   ├─ Detect annotation mode                                │
│   └─ Validate & save preprocessed dataset                  │
│                                                              │
│ Step 2: python cli/train_teacher.py                        │
│   └─ Use complete dataset (with auto-generated boxes)      │
│                                                              │
│ Step 3: python cli/train_student.py                        │
│   └─ Distill to prompt-free model                          │
└─────────────────────────────────────────────────────────────┘
```

**Summary:**
- **Frontend**: Just export segmentation masks (minimal COCO format)
- **Platform**: Automatically handles all preprocessing (bbox, area computation)
- **Result**: Clean separation of concerns, flexible data pipeline

**IMPORTANT - Multiple Objects Handling:**
- Each object instance gets its own annotation entry (separate `id`)
- Each annotation has ONE mask for ONE object
- When auto-generating boxes, each mask generates ONE tight bounding box
- If you have 2 disconnected objects in an image, you should have 2 annotations
- Example: Two "ear of bag" objects → 2 separate annotations with same `category_id` but different `id`

```python
# CORRECT: Two separate annotations for two objects
annotations = [
    {"id": 1, "category_id": 0, "segmentation": mask_obj1},  # Object 1
    {"id": 2, "category_id": 0, "segmentation": mask_obj2},  # Object 2
]
# → Auto-generates: [box_obj1, box_obj2] (two tight boxes)

# WRONG: One annotation with disconnected mask regions
# This is NOT standard COCO format and will cause issues!
```

#### Mode 3: Both (Detection + Segmentation)
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],        // Both provided
      "segmentation": [[x1,y1,x2,y2,...]],  // Both provided
      "area": 1234,
      "iscrowd": 0
    }
  ]
}
```
**Use case:** Full annotations (recommended for best results)

**Data Validation & Mode Detection (with Auto-Preprocessing):**
```python
# CLI: python cli/validate_dataset.py --data train.json

# Step 1: Auto-generate missing fields (bbox, area) from segmentation
def preprocess_and_validate(coco_json):
    """
    Automatically fills in missing bbox/area fields before validation.
    Frontend annotation tool doesn't need to compute these!
    """
    for ann in coco_json['annotations']:
        # Auto-generate bbox from segmentation if missing
        if 'segmentation' in ann and 'bbox' not in ann:
            ann['bbox'] = compute_bbox_from_mask(ann['segmentation'])
            logger.info(f"✓ Auto-generated bbox for annotation {ann['id']}")
        
        # Auto-compute area if missing
        if 'segmentation' in ann and 'area' not in ann:
            ann['area'] = compute_area_from_mask(ann['segmentation'])
    
    return coco_json

# Step 2: Detect annotation mode (after preprocessing)
def detect_annotation_mode(coco_json):
    has_boxes = any(ann.get('bbox') for ann in annotations)
    has_masks = any(ann.get('segmentation') for ann in annotations)
    
    if has_boxes and has_masks:
        return "DETECTION_AND_SEGMENTATION"
    elif has_boxes:
        return "DETECTION_ONLY"
    elif has_masks:
        return "SEGMENTATION_ONLY"
    else:
        raise ValueError("No valid annotations found")

# Validation checks (after preprocessing):
- COCO format compliance
- Image-annotation consistency
- Auto-generate missing bbox/area from segmentation (transparent to user)
- Annotation mode detection
- Class distribution analysis
- Bbox/mask validity (based on mode)
- Minimum samples per class
- Multiple objects per image handled correctly (one annotation per object)
```

**CLI Output Example:**
```bash
$ python cli/validate_dataset.py --data data/raw/train.json

Loading dataset... ✓
Preprocessing annotations...
  ✓ Auto-generated 245 bounding boxes from masks
  ✓ Computed 245 areas from masks
Validating COCO format... ✓
Detecting annotation mode... ✓ SEGMENTATION_ONLY → DETECTION_AND_SEGMENTATION (boxes auto-added)

Dataset validated successfully!
```

**Visual Example: Multiple Objects Handling**

```
Image with 2 disconnected objects:
┌─────────────────────────────────┐
│                                 │
│   🟦 Object 1        🟦 Object 2│
│   (ear)              (ear)      │
│                                 │
└─────────────────────────────────┘

✅ CORRECT COCO Format:
annotations: [
  {"id": 1, "category_id": 0, "segmentation": mask1},  # Object 1
  {"id": 2, "category_id": 0, "segmentation": mask2}   # Object 2
]

Auto-generated boxes:
  box1: tight box around Object 1 only ✓
  box2: tight box around Object 2 only ✓

❌ INCORRECT (Don't do this):
annotations: [
  {"id": 1, "category_id": 0, "segmentation": [mask1_and_mask2_combined]}
]
# This would generate ONE large box encompassing both objects!
```

**Data Augmentation: Characteristic-Based System**

The platform uses an **intelligent, characteristic-based augmentation system** that automatically selects appropriate augmentations based on:
1. **Object characteristics** (what the object is like)
2. **Environment conditions** (where/how it's captured)
3. **Intensity level** (how aggressive the augmentation should be)

**Why Characteristic-Based?**
- ✅ **No hardcoding**: Describe your objects, get appropriate augmentations automatically
- ✅ **Domain-agnostic**: Works for any application (industrial, medical, agricultural, etc.)
- ✅ **User-friendly**: Non-ML users can configure augmentations by answering simple questions
- ✅ **Automatic deduplication**: Merges overlapping augmentations intelligently
- ✅ **Intensity control**: Scale augmentation strength with one parameter

**Available Object Characteristics:**

| Characteristic | What It Means | Example Augmentations Applied |
|----------------|---------------|-------------------------------|
| `changes_shape` | Object can deform/bend | ElasticTransform, PiecewiseAffine |
| `changes_size` | Appears at different scales | RandomScale, RandomSizedBBoxSafeCrop |
| `reflective_surface` | Creates lighting variations | RandomSunFlare, RandomShadow, ColorJitter |
| `low_contrast` | Hard to see against background | CLAHE, RandomBrightnessContrast, Sharpen |
| `moves_or_vibrates` | Creates motion blur | MotionBlur, SafeRotate |
| `semi_transparent` | Varying transparency | RandomFog, GaussNoise, Blur |
| `similar_to_background` | Camouflaged | CLAHE, Sharpen, RandomGamma |
| `multiple_objects` | Multiple instances per image | RandomSizedBBoxSafeCrop |
| `partially_hidden` | Occlusion present | CoarseDropout |

**Available Environment Conditions:**

| Environment Type | Options | Example Augmentations |
|-----------------|---------|----------------------|
| **Lighting** | `stable`, `variable`, `poor` | RandomBrightnessContrast, RandomGamma, CLAHE |
| **Camera** | `fixed`, `moving`, `shaky` | SafeRotate, Affine, MotionBlur, Perspective |
| **Background** | `clean`, `busy`, `changing` | RandomBrightnessContrast, CoarseDropout, HueSaturationValue |
| **Distance** | `fixed`, `variable`, `close` | RandomScale, Perspective, RandomSizedBBoxSafeCrop |

**Implementation:**

```python
# augmentation/__init__.py - Primary API
from augmentation import get_augmentation_registry

# Example 1: Industrial bag inspection
registry = get_augmentation_registry()
pipeline = registry.get_pipeline(
    characteristics=[
        "changes_shape",      # Flexible bags deform
        "reflective_surface", # Plastic surface
        "low_contrast"        # Similar color to conveyor belt
    ],
    environment={
        "lighting": "variable",  # Factory lighting changes
        "camera": "fixed",       # Mounted camera
        "background": "busy"     # Conveyor belt, other items
    },
    intensity="medium"  # Balance between diversity and realism
)

# Apply augmentation
augmented = pipeline(
    image=img,           # numpy array (H, W, C)
    masks=[mask1, mask2], # List of masks
    bboxes=[[x1,y1,x2,y2]], # Optional bboxes
    keypoints=[[x,y]]    # Optional keypoints
)
# Returns: {"image": ..., "masks": [...], "bboxes": [...], "keypoints": [...]}


# Example 2: Medical imaging (different characteristics)
pipeline = registry.get_pipeline(
    characteristics=[
        "low_contrast",      # Medical scans often low contrast
        "semi_transparent",  # Some tissues semi-transparent
    ],
    environment={
        "lighting": "stable",  # Controlled medical environment
        "camera": "fixed",     # Medical scanner
        "background": "clean", # Clean scan background
        "distance": "fixed"    # Fixed scanner distance
    },
    intensity="low"  # Conservative for medical data
)


# Example 3: Preview before creating pipeline
info = registry.get_pipeline_info(
    characteristics=["changes_shape", "reflective_surface"],
    environment={"lighting": "variable"},
    intensity="medium"
)
print(f"Will apply {info['total_augmentations']} augmentations:")
print(f"Types: {info['augmentation_types']}")
# Output:
# Will apply 6 augmentations:
# Types: ['ElasticTransform', 'PiecewiseAffine', 'RandomSunFlare', 
#         'RandomShadow', 'ColorJitter', 'RandomBrightnessContrast']
```

**Intensity Levels:**

| Intensity | Use When | Augmentation Strength | Probability (p) |
|-----------|----------|----------------------|----------------|
| `low` | Small dataset, needs conservative augmentation | Minimal parameter ranges | 0.15-0.25 |
| `medium` | Balanced approach (recommended) | Moderate parameter ranges | 0.3-0.5 |
| `high` | Large dataset, robust model needed | Aggressive parameter ranges | 0.5-0.8 |

**Integration with Training Pipeline:**

```python
# In ml_engine/training/teacher_trainer.py or student_trainer.py
from augmentation import get_augmentation_registry

class TeacherTrainer:
    def __init__(self, config):
        # Load augmentation config from training config
        aug_config = config['augmentation']
        
        # Create characteristic-based pipeline
        registry = get_augmentation_registry()
        self.augmentation_pipeline = registry.get_pipeline(
            characteristics=aug_config['characteristics'],
            environment=aug_config['environment'],
            intensity=aug_config['intensity']
        )
    
    def train_epoch(self):
        for batch in dataloader:
            # Apply augmentations
            augmented = self.augmentation_pipeline(
                image=batch['image'],
                masks=batch['masks'],
                bboxes=batch.get('bboxes'),  # Optional
                keypoints=batch.get('keypoints')  # Optional
            )
            
            # Train with augmented data
            loss = self.model(augmented['image'], augmented['masks'])
            loss.backward()
```

**Configuration in Training YAML:**

```yaml
# configs/training/teacher_grounding_dino_lora.yaml
augmentation:
  # Describe your objects and environment
  characteristics:
    - "changes_shape"
    - "reflective_surface"
    - "low_contrast"
  
  environment:
    lighting: "variable"  # Options: stable, variable, poor
    camera: "fixed"       # Options: fixed, moving, shaky
    background: "busy"    # Options: clean, busy, changing
    distance: "fixed"     # Options: fixed, variable, close
  
  intensity: "medium"  # Options: low, medium, high
  
  # Traditional augmentations still available for distillation
  distillation_specific:
    mosaic: 0.5           # Mosaic augmentation probability
    mixup: 0.1            # MixUp probability
    copy_paste: 0.15      # Copy-paste for rare classes
```

**Available Characteristics Programmatically:**

```python
# For building a GUI/frontend selection interface
registry = get_augmentation_registry()

# Get all available characteristics
characteristics = registry.get_available_characteristics()
# Returns: ['changes_shape', 'changes_size', 'reflective_surface', 
#           'low_contrast', 'moves_or_vibrates', ...]

# Get all environment options
environments = registry.get_available_environments()
# Returns: {
#     'lighting': ['stable', 'variable', 'poor'],
#     'camera': ['fixed', 'moving', 'shaky'],
#     'background': ['clean', 'busy', 'changing'],
#     'distance': ['fixed', 'variable', 'close']
# }

# Validate user input
validation = registry.translator.validate_characteristics(
    ["changes_shape", "unknown_characteristic"]
)
if not validation['valid']:
    print(f"Invalid: {validation['unsupported_characteristics']}")
    print(f"Available: {validation['available_characteristics']}")
```

**Benefits Over Traditional Augmentation:**

| Aspect | Traditional (Hardcoded) | Characteristic-Based |
|--------|------------------------|---------------------|
| **Configuration** | List specific transforms | Describe object + environment |
| **Domain Transfer** | Must re-configure for each domain | Same system, different characteristics |
| **User-Friendliness** | Requires ML knowledge | Domain experts can configure |
| **Redundancy** | Manually avoid duplicates | Automatic deduplication |
| **Intensity Control** | Adjust each transform separately | Single intensity parameter |
| **Extensibility** | Add new transforms manually | Add characteristic, get transforms |

**Architecture:**

```
User Input (Simple)
├─ Characteristics: ["changes_shape", "reflective_surface"]
├─ Environment: {"lighting": "variable"}
└─ Intensity: "medium"
        ↓
CharacteristicTranslator (Smart)
├─ Maps characteristics → augmentation rules
├─ Maps environment → augmentation rules
├─ Applies intensity-specific parameters
├─ Deduplicates overlapping augmentations
└─ Returns unified parameter dict
        ↓
ConfigurableAugmentationPipeline
├─ Converts parameters → albumentations transforms
├─ Builds Compose pipeline
└─ Ready for training
        ↓
Applied to Data
└─ Returns: {"image": ..., "masks": ..., "bboxes": ..., "keypoints": ...}
```

**For Distillation (Additional Augmentations):**

Student models benefit from additional augmentations not suitable for teacher fine-tuning:

```python
# In distillation config
distillation:
  # Characteristic-based augmentation (same as teacher)
  augmentation:
    characteristics: ["changes_shape", "reflective_surface"]
    environment: {"lighting": "variable"}
    intensity: "high"  # More aggressive for student
  
  # Additional augmentations specific to distillation
  student_augmentation:
    mosaic: 0.5        # Combine 4 images into one (YOLO-specific)
    mixup: 0.1         # Blend two images
    copy_paste: 0.15   # Paste rare objects into scenes
```

**Summary:**
- **Teacher fine-tuning**: Use characteristic-based augmentation (domain-specific, realistic)
- **Student distillation**: Add mosaic, mixup, copy-paste (data diversity for generalization)
- **Configuration**: User-friendly, no ML expertise required
- **Implementation**: Built on `albumentations`, fully tested, production-ready

### Stage 2: Teacher Model Fine-tuning

This stage adapts the teacher models to your specific domain. The pipeline automatically selects which models to fine-tune based on available annotations in your dataset.

#### Step 2.1: Fine-tune Grounding DINO (Detection) with LoRA

**When to fine-tune:** If annotations contain boxes
**Skip if:** Only masks available - boxes will be auto-generated from masks

```python
# CLI Command (Simplified):
# python cli/train_teacher.py \
#     --data data/raw/train.json \
#     --val data/raw/val.json \
#     --output experiments/my_experiment \
#     --gpu 0

# Platform automatically loads defaults and auto-fills from dataset
# Saved to: experiments/my_experiment/teacher_config.yaml

# Default config: configs/defaults/teacher_grounding_dino_lora.yaml
# Platform auto-fills dataset-specific values (num_classes, class_names)

model_config = {
    # === MODEL CONFIGURATION (Defaults - usually don't change) ===
    "base_checkpoint": "data/models/pretrained/groundingdino_swint_ogc.pth",
    
    # === DATASET-SPECIFIC (AUTO-FILLED BY PLATFORM) ===
    "num_classes": 3,  # Auto-detected from COCO dataset
    "class_names": ["class_1", "class_2", "class_3"],  # Auto-filled from COCO categories
    
    # === LoRA Configuration (Defaults - can tune if needed) ===
    "use_lora": True,
    "lora_config": {
        "r": 16,                    # LoRA rank (16 is good default, try 8/32 to tune)
        "lora_alpha": 32,           # LoRA scaling factor (typically 2*r)
        "target_modules": [          # Which layers to add LoRA adapters
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj"
        ],
        "lora_dropout": 0.1,        # Dropout for LoRA layers (0.05-0.1 works well)
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION"
    },
    
    # === TRAINING HYPERPARAMETERS (Defaults - adjust based on dataset size) ===
    "learning_rate": 1e-4,          # Higher LR for LoRA (adapters only)
    "batch_size": 8,                # 2x larger due to LoRA memory savings
    "epochs": 50,                    # Increase for small datasets (<500 images)
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "warmup_steps": 500,            # Typically 5-10% of total steps
    "gradient_accumulation": 2,     # Effective batch_size = batch_size * this
    "mixed_precision": "fp16",      # Save memory
}

# === USAGE ===
# Platform auto-generates config from defaults + dataset:
python cli/train_teacher.py --data train.json --val val.json --output experiments/exp1
# Automatically:
# - Inspects COCO dataset for num_classes, class_names
# - Loads defaults from configs/defaults/teacher_grounding_dino_lora.yaml
# - Merges dataset info with defaults
# - Saves to experiments/exp1/teacher_config.yaml
# - Starts training

# LoRA benefits:
# - Memory: 14.4GB GPU (vs 47GB full fine-tuning)
# - Trainable params: ~2.5M (vs 176M full model)  
# - Training time: 2-3x faster
# - Adapter size: Only 19MB! (vs 11GB full checkpoint)
#
# 🔑 IMPORTANT: What LoRA Training Saves:
# Training output: adapter_model.bin (~19MB) + adapter_config.json
# This is NOT a full model! It's just the "delta" (difference) from base model.
#
# At inference, you need BOTH:
#   1. Base pretrained model (11GB) - download once, reuse for all tasks
#   2. LoRA adapter (19MB) - one per task/dataset
#
# Optional: Merge adapter INTO base to create single fine-tuned model:
#   merged_model = model.merge_and_unload()  # Creates 11GB fine-tuned model

# Expected results after LoRA fine-tuning:
# - mAP50: 0.85-0.95 on your domain (comparable to full fine-tuning!)
# - Better prompt-to-detection alignment for your classes
```

#### 🔑 CRITICAL: Our Approach - Partial Freeze + LoRA

**The Strategy: Freeze Backbone, Apply LoRA to Task-Specific Parts**

We use the **Partial Freeze + LoRA** approach, which provides optimal balance:
- ✅ **Freeze large, general-purpose parts** (image encoders/backbones)
- ✅ **Apply LoRA ONLY to task-specific parts** (decoders/heads)
- ✅ **Minimal memory usage** (only adapter gradients)
- ✅ **Small checkpoint size** (only adapters saved)
- ✅ **Prevents catastrophic forgetting** (frozen backbone retains pretrained features)

**The Core Principle: Freeze What's Expensive, Train What Matters**

```python
# This is THE fundamental concept that makes LoRA work!

# Step 1: Load pretrained base model
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

base_model = AutoModel.from_pretrained("groundingdino_swint_ogc")
print(f"Total parameters: {sum(p.numel() for p in base_model.parameters()):,}")
# Output: 176,000,000 parameters (176M)

# Step 2: Apply LoRA configuration (this AUTOMATICALLY freezes base model)
lora_config = LoraConfig(
    r=16,                          # Rank of LoRA matrices
    lora_alpha=32,                 # Scaling factor
    target_modules=[               # Which layers get LoRA adapters
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.out_proj"
    ],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, lora_config)

# Step 3: Check what's frozen and what's trainable
print(model.print_trainable_parameters())
# Output: trainable params: 2,457,600 || all params: 178,457,600 || trainable%: 1.38%

# 🔑 KEY INSIGHT:
# - Base model (176M params): FROZEN ❄️ - gradients NOT computed
# - LoRA adapters (2.5M params): TRAINABLE 🔥 - only these are updated
# - Total trainable: 1.38% of full model!
# - What gets saved: ONLY LoRA adapters (~19MB)
# - What you need for distillation: Base model + LoRA adapters
```

**What Happens During Forward Pass:**

```python
# Mathematical explanation of LoRA:

# Original layer (frozen):
# h = W₀ · x
# where W₀ is the pretrained weight matrix (e.g., 768 × 768)

# LoRA layer (trainable):
# h = W₀ · x + ΔW · x
# where ΔW = B · A (low-rank decomposition)
#   - A is (r × 768) where r=16 (much smaller!)
#   - B is (768 × r) where r=16
#   - ΔW = B · A gives (768 × 768) but only uses 2 × r × 768 = 24,576 params!

# During training:
#   - W₀ is frozen (no gradients computed) ❄️
#   - Only A and B are updated (gradients computed) 🔥
#   - Memory savings: Don't need to store gradients for W₀!

# Code visualization:
for name, param in model.named_parameters():
    if "lora" in name:
        print(f"✅ {name}: TRAINABLE, shape={param.shape}")
        param.requires_grad = True   # Already set by PEFT
    else:
        print(f"❌ {name}: FROZEN, shape={param.shape}")
        param.requires_grad = False  # Already set by PEFT
```

**Memory Savings Breakdown:**

```python
# Why LoRA uses less memory:

# Full Fine-tuning Memory:
# ├─ Model weights: 11GB
# ├─ Gradients: 11GB      ← Store gradients for ALL parameters
# ├─ Optimizer states: 22GB ← Adam stores 2× gradients (momentum, variance)
# └─ Activations: ~3GB
# Total: ~47GB GPU memory

# LoRA Fine-tuning Memory:
# ├─ Base model weights: 11GB (loaded once, frozen)
# ├─ LoRA adapter weights: 0.019GB (tiny!)
# ├─ Gradients: 0.019GB    ← Only for LoRA adapters! 🎉
# ├─ Optimizer states: 0.038GB ← Only for LoRA adapters! 🎉
# └─ Activations: ~3GB
# Total: ~14.4GB GPU memory (3.3x less!)
```

**Freezing Configuration in Code:**

```python
# In ml_engine/training/teacher_trainer.py

from peft import get_peft_model, LoraConfig

class GroundingDINOLoRATrainer:
    def __init__(self, base_model_path, lora_config):
        # Load base model
        self.base_model = load_grounding_dino(base_model_path)
        
        # Apply LoRA (automatically freezes base model)
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Verify freezing
        self.verify_freezing()
    
    def verify_freezing(self):
        """Ensure base model is frozen and only LoRA adapters are trainable."""
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                assert "lora" in name.lower(), f"Non-LoRA param is trainable: {name}"
            else:
                frozen_params += param.numel()
        
        print(f"✅ Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.1f}M)")
        print(f"✅ Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"✅ Trainable ratio: {100*trainable_params/(frozen_params+trainable_params):.2f}%")
        
        assert trainable_params < frozen_params * 0.05, "Too many trainable params for LoRA!"
```

**Visual Architecture:**

```
Grounding DINO Architecture with LoRA:

┌─────────────────────────────────────────────────────────────┐
│ Input Image                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ Swin Transformer Backbone (FROZEN ❄️)                       │
│ ├─ Patch Embedding: 176M params (frozen)                    │
│ ├─ Transformer Blocks:                                      │
│ │   ├─ Self-Attention:                                      │
│ │   │   ├─ W_q (frozen ❄️) + LoRA_q (trainable 🔥)        │
│ │   │   ├─ W_k (frozen ❄️) + LoRA_k (trainable 🔥)        │
│ │   │   ├─ W_v (frozen ❄️) + LoRA_v (trainable 🔥)        │
│ │   │   └─ W_out (frozen ❄️) + LoRA_out (trainable 🔥)    │
│ │   └─ MLP (frozen ❄️)                                      │
│ └─ ... (more blocks)                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ Detection Head (FROZEN ❄️)                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                    Predictions

Legend:
❄️ FROZEN: No gradients, no updates, saves memory
🔥 TRAINABLE: Gradients computed, weights updated (only 2.5M params!)
```

**Why This Freezing Strategy Works:**

1. ✅ **Preserves pretrained knowledge**: Base model keeps its general features
2. ✅ **Efficient adaptation**: LoRA adapters learn task-specific adjustments
3. ✅ **Memory efficient**: No gradients for 176M params, only for 2.5M
4. ✅ **Fast training**: Fewer params to update = faster backprop
5. ✅ **Prevents overfitting**: Limited capacity prevents memorization

**Common Mistake (DON'T DO THIS):**

```python
# ❌ WRONG: Unfreezing parts of base model defeats LoRA purpose
model = get_peft_model(base_model, lora_config)

# Don't do this!
for name, param in model.named_parameters():
    if "layer.23" in name:  # Unfreezing last layer
        param.requires_grad = True  # ❌ BAD! Increases memory usage!

# This defeats the purpose of LoRA:
# - Memory usage increases
# - Training slows down
# - Loses LoRA's efficiency benefits
```

---

#### 🔑 Implementation: Partial Freeze + LoRA for SAM

**How to Apply This to SAM Fine-Tuning**
```python
# Example: SAM fine-tuning
# Freeze image encoder (large, general), add LoRA ONLY to decoder

model = SAM(checkpoint="pretrained/sam_vit_h.pth")

# Freeze image encoder (expensive, doesn't need task-specific tuning)
for param in model.image_encoder.parameters():
    param.requires_grad = False  # Freeze ViT: 308M params ❄️

# Keep prompt encoder frozen too (small, works well pretrained)
for param in model.prompt_encoder.parameters():
    param.requires_grad = False  # Freeze: 3.8M params ❄️

# Apply LoRA ONLY to mask decoder (task-specific part)
# 🔑 IMPORTANT: This ALSO freezes the decoder's base weights!
# LoRA adds trainable adapters ON TOP of frozen decoder weights
from peft import inject_adapter_in_model, LoraConfig

lora_config = LoraConfig(r=8, target_modules=[".*mask_decoder.*self_attn.*proj"])
model = inject_adapter_in_model(lora_config, model)

# 🔑 CRITICAL CLARIFICATION:
# What's frozen: 
#   - Image encoder: 308M params ❄️
#   - Prompt encoder: 3.8M params ❄️
#   - Mask decoder BASE weights: 4.1M params ❄️  ← Also frozen!
#   Total frozen: 315.9M params
#
# What's trainable:
#   - Mask decoder LoRA adapters ONLY: 0.4M params 🔥  ← New, added on top
#   Total trainable: 0.4M params (0.13% of model!)
#
# What's modified:
#   - NOTHING in the pretrained model is modified!
#   - Only NEW LoRA adapter weights are created and trained
#
# What gets saved:
#   - ONLY LoRA adapters (~1.5MB)
#   - Pretrained model unchanged, reusable for other tasks
#
# Memory: 9GB
# For distillation: Load base SAM (unchanged) + LoRA adapters
```

**🔑 CRITICAL: What Gets Saved and How to Load for Distillation**

```python
# === What Happened During Training ===
# - Pretrained model: UNCHANGED (all base weights frozen) ❄️
# - New LoRA adapters: CREATED and trained 🔥
#
# === Saved After Training ===
# - data/models/pretrained/sam_vit_h.pth  (unchanged, still 2.4GB)
# - data/models/teachers/sam_lora/
#   ├── adapter_config.json       (LoRA config)
#   └── adapter_model.bin          (~1.5MB - ONLY adapters!)

# === Loading for Distillation ===
from peft import PeftModel

# Load base model (unchanged)
base_sam = load_sam("pretrained/sam_vit_h.pth")

# Apply LoRA adapters
teacher_sam = PeftModel.from_pretrained(base_sam, "teachers/sam_lora/")

# How it works: output = W₀x + BAx
#   - W₀: frozen pretrained weights
#   - BA: trained LoRA adapters
#   - Base weights W₀ unchanged, adapters BA applied on top

# Similarly for Grounding DINO:
base_dino = load_grounding_dino("data/models/pretrained/groundingdino_swint_ogc.pth")
teacher_dino = PeftModel.from_pretrained(base_dino, "experiments/exp1/teachers/dino_lora/")
```

**🔑 KEY INSIGHT: Why This Approach Works Best**

The Partial Freeze + LoRA strategy provides optimal balance:

1. ✅ **Image encoders are general**: ViT/Swin backbones learn general visual features, don't need task-specific tuning
2. ✅ **Task-specific parts benefit most**: Decoders/heads adapt to your specific classes via LoRA
3. ✅ **Minimal memory**: Freeze the large backbone (300M+ params), train tiny adapters (0.4M)
4. ✅ **Small checkpoints**: Only adapters saved (1.5MB vs gigabytes)
5. ✅ **Prevents overfitting**: Frozen backbone prevents forgetting general features
6. ✅ **Pretrained model unchanged**: Can reuse base models for multiple experiments

**📝 Key Clarification: What "Partial Freeze + LoRA" Means**

- ✅ **"Partial Freeze"** = Freeze encoder, apply LoRA to decoder
- ✅ **ALL base weights remain frozen** - including the decoder base weights
- ✅ **Only NEW LoRA adapters are trained** - these are added on top of frozen weights
- ✅ **Pretrained model is NEVER modified** - completely unchanged
- ✅ **Only adapters are saved** - 1.5MB, not full model

**Visual Architecture:**
```
Partial Freeze + LoRA:
┌──────────────┐
│ Image Encoder│ ❄️ Frozen (base weights unchanged)
└──────────────┘
┌──────────────┐
│ Mask Decoder │ ❄️ Frozen (base weights unchanged)
│   + LoRA     │ 🔥 NEW adapters trained on top
└──────────────┘
Save: ONLY 1.5MB LoRA adapters
Load: Base model + Apply adapters
```

**Implementation for This Project:**

```python
# For Grounding DINO:
# - Freeze: Swin Transformer backbone (158M params) ❄️
# - LoRA: Attention layers in transformer decoder (2.5M → trainable) 🔥
# - Saves: ~19MB LoRA adapters

# For SAM:
# - Freeze: ViT image encoder (308M params) ❄️
# - Freeze: Prompt encoder (3.8M params) ❄️
# - LoRA: Mask decoder only (0.4M trainable) 🔥
# - Saves: ~1.5MB LoRA adapters

# For distillation, use merged models for speed:
teacher = GroundedSAM(
    grounding_dino_base="data/models/pretrained/groundingdino.pth",
    grounding_dino_lora="experiments/exp1/teachers/dino_lora/",  # LoRA adapters
    sam_base="data/models/pretrained/sam_vit_h.pth",
    sam_lora="experiments/exp1/teachers/sam_lora/",  # LoRA adapters
    use_merged=True  # Merge adapters into base model for faster distillation
)
```

---

### 🔧 Training Dynamics & Stability Configuration

**Why This Matters**: Proper training dynamics prevent common failures like gradient explosion, overfitting, and unstable convergence.

#### Complete Training Configuration Template

```yaml
# configs/defaults/training_dynamics.yaml
training_dynamics:
  # Gradient management
  gradient_clipping:
    enabled: true
    max_norm: 0.1        # For LoRA (small gradients). Use 1.0 for full fine-tuning
    norm_type: 2.0       # L2 norm
    error_if_nonfinite: true  # Stop training if NaN/Inf gradients
    
  # Mixed precision training (FP16)
  mixed_precision:
    enabled: true
    backend: "amp"       # PyTorch Automatic Mixed Precision
    init_scale: 65536    # 2^16, initial loss scaling
    growth_factor: 2.0   # Scale multiplier when no overflow
    backoff_factor: 0.5  # Scale divisor when overflow detected
    growth_interval: 2000  # Steps between scale increases
    
  # Batch normalization strategy
  normalization:
    # For LoRA fine-tuning (teacher models)
    freeze_bn_teacher: true  # Freeze BatchNorm statistics
    track_running_stats_teacher: false  # Don't update running mean/var
    
    # For distillation (student training)
    student_bn_mode: "train"   # Student BN in train mode
    teacher_bn_mode: "eval"    # Teacher BN in eval mode (frozen)
    sync_bn: false  # Don't sync BN across GPUs (not needed for single GPU)
    
  # Learning rate warmup
  lr_warmup:
    enabled: true
    method: "linear"     # "linear", "cosine", or "constant"
    warmup_epochs: 3     # Typically 3-5 epochs
    warmup_ratio: 0.1    # Start from 10% of base LR
    warmup_by_epoch: true  # vs by iteration
```

#### Implementation: Training Manager

```python
# ml_engine/training/training_manager.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
import yaml

class TrainingManager:
    """
    Manages all training dynamics: gradient clipping, mixed precision, etc.
    
    Why centralized?
    - Consistent training behavior across teacher/student
    - Easy to modify hyperparameters
    - Prevents common training failures
    - Config-driven (no hardcoding!)
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 config_path: str):
        """
        Args:
            model: The model being trained
            optimizer: The optimizer
            config_path: Path to training dynamics config
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config['training_dynamics']
        self.model = model
        self.optimizer = optimizer
        
        # Setup mixed precision
        self.use_amp = self.config['mixed_precision']['enabled']
        if self.use_amp:
            amp_config = self.config['mixed_precision']
            self.scaler = GradScaler(
                init_scale=amp_config['init_scale'],
                growth_factor=amp_config['growth_factor'],
                backoff_factor=amp_config['backoff_factor'],
                growth_interval=amp_config['growth_interval'],
            )
        else:
            self.scaler = None
        
        # Setup gradient clipping
        self.clip_cfg = self.config['gradient_clipping']
        
        # Setup BN behavior
        self._configure_batch_norm()
    
    def _configure_batch_norm(self):
        """Configure BatchNorm layers based on config."""
        norm_cfg = self.config['normalization']
        
        # For teacher fine-tuning with LoRA
        if norm_cfg.get('freeze_bn_teacher', False):
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()  # Keep in eval mode
                    module.track_running_stats = False
                    # Freeze parameters
                    for param in module.parameters():
                        param.requires_grad = False
    
    def training_step(self, batch: Dict, compute_loss_fn) -> Dict:
        """
        Execute one training step with proper gradient handling.
        
        Args:
            batch: Input batch
            compute_loss_fn: Function that computes loss given batch
            
        Returns:
            Dict with loss and metrics
        """
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with automatic mixed precision
        if self.use_amp:
            with autocast():
                loss_dict = compute_loss_fn(batch)
                loss = loss_dict['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (if enabled)
            if self.clip_cfg['enabled']:
                self.scaler.unscale_(self.optimizer)  # Unscale before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.clip_cfg['max_norm'],
                    norm_type=self.clip_cfg['norm_type'],
                    error_if_nonfinite=self.clip_cfg.get('error_if_nonfinite', True)
                )
                loss_dict['grad_norm'] = grad_norm.item()
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        else:
            # Standard training without AMP
            loss_dict = compute_loss_fn(batch)
            loss = loss_dict['loss']
            loss.backward()
            
            # Gradient clipping (if enabled)
            if self.clip_cfg['enabled']:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.clip_cfg['max_norm'],
                    norm_type=self.clip_cfg['norm_type']
                )
                loss_dict['grad_norm'] = grad_norm.item()
            
            self.optimizer.step()
        
        return loss_dict
    
    def get_grad_statistics(self) -> Dict:
        """Get gradient statistics for monitoring."""
        stats = {
            'grad_norm': [],
            'grad_max': [],
            'grad_mean': [],
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                stats['grad_norm'].append(grad.norm().item())
                stats['grad_max'].append(grad.abs().max().item())
                stats['grad_mean'].append(grad.abs().mean().item())
        
        if stats['grad_norm']:
            return {
                'total_grad_norm': sum(stats['grad_norm']),
                'max_grad_value': max(stats['grad_max']),
                'mean_grad_value': sum(stats['grad_mean']) / len(stats['grad_mean']),
            }
        return {}

# Usage
config_path = 'configs/defaults/training_dynamics.yaml'
training_manager = TrainingManager(model, optimizer, config_path)

for batch in dataloader:
    loss_dict = training_manager.training_step(batch, compute_loss_fn=my_loss_fn)
    # loss_dict contains: {'loss': ..., 'grad_norm': ..., ...}
```

#### Why Gradient Clipping is Essential for LoRA

**Problem**: Even though LoRA trains fewer parameters, gradients can still explode with small learning rates.

```python
# Without gradient clipping:
# Iteration 1: grad_norm = 0.05  ✓
# Iteration 2: grad_norm = 0.08  ✓
# Iteration 3: grad_norm = 15.3  ✗ EXPLOSION!
# Iteration 4: NaN gradients → Training crashes

# With gradient clipping (max_norm=0.1):
# Iteration 1: grad_norm = 0.05  ✓
# Iteration 2: grad_norm = 0.08  ✓
# Iteration 3: grad_norm = 0.1 (clipped from 15.3)  ✓
# Iteration 4: Training continues normally  ✓
```

**Recommended values**:
- LoRA training: `max_norm=0.1` (small because gradients only on adapters)
- Full fine-tuning: `max_norm=1.0` (larger because all parameters)
- Student distillation: `max_norm=10.0` (largest because training from scratch)

---

### 📊 Checkpoint Management & Model Selection

**Why This Matters**: Proper checkpointing enables:
- Recovery from crashes
- Model selection based on validation metrics
- Experiment reproducibility
- Storage efficiency

#### Checkpoint Configuration

```yaml
# configs/defaults/checkpoint_config.yaml
checkpointing:
  # Saving strategy
  save_interval: 5        # Save every N epochs
  save_interval_steps: null  # Or save every N steps (optional)
  save_last: true         # Always save last checkpoint
  save_best: true         # Save best model based on metric
  max_keep_checkpoints: 5  # Keep only last N checkpoints (saves disk space)
  
  # Best model selection
  monitor_metric: "mAP50"  # Metric to monitor for best model
  # Options: "mAP50", "mAP50-95", "mask_IoU", "val_loss"
  mode: "max"             # "max" for accuracy metrics, "min" for loss
  min_delta: 0.001        # Minimum improvement to consider
  
  # Checkpoint content
  save_optimizer: true    # Save optimizer state (needed for resume)
  save_scheduler: true    # Save LR scheduler state
  save_scaler: true       # Save AMP scaler state (if using mixed precision)
  save_rng_state: true    # Save random states for reproducibility
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 15          # Stop if no improvement for N epochs
    min_delta: 0.001      # Minimum improvement threshold
    restore_best_weights: true  # Restore best weights after stopping
  
  # Checkpoint naming
  checkpoint_format: "epoch_{epoch:04d}_map{mAP50:.4f}.pth"
  best_checkpoint_name: "best.pth"
  last_checkpoint_name: "last.pth"
```

#### Implementation: Checkpoint Manager

```python
# ml_engine/training/checkpoint_manager.py

import torch
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import yaml
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages model checkpoints: saving, loading, best model selection, early stopping.
    
    Key features:
    - Config-driven (no hardcoded paths/metrics)
    - Automatic cleanup of old checkpoints
    - Best model tracking across experiments
    - Early stopping support
    - Full reproducibility (saves RNG states)
    """
    
    def __init__(self, output_dir: str, config_path: str):
        """
        Args:
            output_dir: Directory to save checkpoints
            config_path: Path to checkpoint config
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config['checkpointing']
        
        # Best model tracking
        self.monitor_metric = self.config['monitor_metric']
        self.mode = self.config['mode']
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.best_epoch = -1
        
        # Early stopping
        self.early_stop_cfg = self.config['early_stopping']
        self.patience_counter = 0
        self.should_stop = False
        
        # Checkpoint history
        self.checkpoint_history: List[Path] = []
    
    def save_checkpoint(self, 
                       epoch: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       scaler: Optional[torch.cuda.amp.GradScaler],
                       metrics: Dict[str, float],
                       extra_info: Optional[Dict] = None) -> Path:
        """
        Save a checkpoint with all necessary information.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            scaler: AMP scaler state
            metrics: Dictionary of metrics (must include monitor_metric)
            extra_info: Additional info to save (e.g., config, args)
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
        }
        
        # Add optional components based on config
        if self.config['save_optimizer'] and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.config['save_scheduler'] and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if self.config['save_scaler'] and scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if self.config['save_rng_state']:
            checkpoint['rng_state'] = {
                'python': torch.get_rng_state(),
                'numpy': None,  # Add numpy.random.get_state() if using numpy
                'cuda': torch.cuda.get_rng_state_all(),
            }
        
        if extra_info:
            checkpoint['extra_info'] = extra_info
        
        # Determine checkpoint path
        is_best = self._is_best(metrics)
        save_paths = []
        
        # Save regular checkpoint (every N epochs)
        if epoch % self.config['save_interval'] == 0:
            # Format checkpoint name
            checkpoint_name = self.config['checkpoint_format'].format(
                epoch=epoch,
                **metrics
            )
            checkpoint_path = self.output_dir / checkpoint_name
            torch.save(checkpoint, checkpoint_path)
            save_paths.append(checkpoint_path)
            self.checkpoint_history.append(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best and self.config['save_best']:
            best_path = self.output_dir / self.config['best_checkpoint_name']
            torch.save(checkpoint, best_path)
            save_paths.append(best_path)
            logger.info(f"✨ New best model! {self.monitor_metric}={metrics[self.monitor_metric]:.4f}")
            logger.info(f"Saved best checkpoint: {best_path}")
        
        # Save last checkpoint (overwrite)
        if self.config['save_last']:
            last_path = self.output_dir / self.config['last_checkpoint_name']
            torch.save(checkpoint, last_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Check early stopping
        self._check_early_stopping(metrics)
        
        return save_paths[0] if save_paths else None
    
    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far."""
        if self.monitor_metric not in metrics:
            logger.warning(f"Monitor metric '{self.monitor_metric}' not in metrics: {metrics.keys()}")
            return False
        
        current_metric = metrics[self.monitor_metric]
        min_delta = self.config['min_delta']
        
        if self.mode == 'max':
            is_better = current_metric > (self.best_metric + min_delta)
        else:
            is_better = current_metric < (self.best_metric - min_delta)
        
        if is_better:
            self.best_metric = current_metric
            self.best_epoch = metrics.get('epoch', -1)
            self.patience_counter = 0  # Reset early stopping counter
            return True
        
        return False
    
    def _check_early_stopping(self, metrics: Dict[str, float]):
        """Check if training should stop early."""
        if not self.early_stop_cfg['enabled']:
            return
        
        if self.monitor_metric not in metrics:
            return
        
        current_metric = metrics[self.monitor_metric]
        min_delta = self.early_stop_cfg['min_delta']
        
        # Check if improved
        if self.mode == 'max':
            improved = current_metric > (self.best_metric + min_delta)
        else:
            improved = current_metric < (self.best_metric - min_delta)
        
        if not improved:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter}/{self.early_stop_cfg['patience']} epochs")
            
            if self.patience_counter >= self.early_stop_cfg['patience']:
                self.should_stop = True
                logger.info(f"⚠️ Early stopping triggered! Best {self.monitor_metric}={self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        max_keep = self.config.get('max_keep_checkpoints', 5)
        
        if len(self.checkpoint_history) > max_keep:
            # Keep only the most recent checkpoints
            to_remove = self.checkpoint_history[:-max_keep]
            self.checkpoint_history = self.checkpoint_history[-max_keep:]
            
            for checkpoint_path in to_remove:
                if checkpoint_path.exists() and checkpoint_path.name not in [
                    self.config['best_checkpoint_name'],
                    self.config['last_checkpoint_name']
                ]:
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       load_optimizer: bool = True,
                       strict: bool = True) -> Dict:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            scaler: Scaler to load state into
            load_optimizer: Whether to load optimizer state
            strict: Strict mode for model loading
            
        Returns:
            Checkpoint dictionary with metadata
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        logger.info(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        
        # Load optimizer
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state loaded")
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ Scheduler state loaded")
        
        # Load scaler
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("✓ AMP scaler state loaded")
        
        # Load RNG state for reproducibility
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['python'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
            logger.info("✓ RNG states restored")
        
        # Restore best metric tracking
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', -1)
        
        logger.info(f"✓ Checkpoint loaded successfully")
        logger.info(f"  Best metric so far: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        return checkpoint

# Usage
checkpoint_manager = CheckpointManager(
    output_dir='experiments/exp1/teachers/grounding_dino_lora',
    config_path='configs/defaults/checkpoint_config.yaml'
)

# During training
for epoch in range(start_epoch, num_epochs):
    train_metrics = train_one_epoch(...)
    val_metrics = validate(...)
    
    # Combine metrics
    metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metrics=metrics,
        extra_info={'config': training_config}
    )
    
    # Check if should stop early
    if checkpoint_manager.should_stop:
        logger.info("Training stopped early!")
        break
```

#### Key Benefits

1. **Config-Driven**: All checkpoint behavior controlled by YAML config
2. **Automatic Cleanup**: Prevents disk space issues
3. **Best Model Selection**: Tracks best model across all epochs
4. **Early Stopping**: Prevents overfitting, saves compute time
5. **Reproducibility**: Saves RNG states for exact reproduction
6. **Resume Training**: Full state restoration for crashed runs

#### Understanding LoRA: What Gets Saved & How to Use It

**🔑 CRITICAL CONCEPT: LoRA Saves Adapters, Not Full Models**

```
Training Process:
┌──────────────────────────────────────────────────────────┐
│ 1. Load Base Model (11GB for DINO, 2.4GB for SAM)       │
│    ↓                                                     │
│ 2. Add LoRA Layers (small adapter matrices)             │
│    ↓                                                     │
│ 3. Train ONLY LoRA Layers (freeze base model)           │
│    ↓                                                     │
│ 4. Save ONLY LoRA Weights (~19MB for DINO, ~1.5MB SAM)  │
│    ├─ adapter_model.bin       (LoRA weight deltas)      │
│    ├─ adapter_config.json     (LoRA configuration)      │
│    └─ (Base model NOT saved - already have it!)         │
└──────────────────────────────────────────────────────────┘

Storage After Training:
├─ Base models (download once, reuse forever):
│   ├─ groundingdino_swint_ogc.pth  (11GB) ✓
│   └─ sam_vit_h_4b8939.pth         (2.4GB) ✓
│
└─ LoRA adapters (one per task/dataset):
    ├─ Task 1 (industrial bags):
    │   ├─ grounding_dino_lora/  (19MB)
    │   └─ sam_lora/             (1.5MB)
    │
    ├─ Task 2 (medical images):
    │   ├─ grounding_dino_lora/  (19MB)
    │   └─ sam_lora/             (1.5MB)
    │
    └─ Task 3 (satellite images):
        ├─ grounding_dino_lora/  (19MB)
        └─ sam_lora/             (1.5MB)

Total storage for 3 tasks:
  - Full fine-tuning: 3 × (11GB + 2.4GB) = 40.2GB 😱
  - LoRA fine-tuning: 13.4GB base + 3 × 20.5MB = 13.46GB 🎉
```

**Two Ways to Use LoRA Models:**

```python
# Option 1: Load base + adapter separately (recommended for multiple tasks)
from peft import PeftModel

base_model = load_grounding_dino("data/models/pretrained/groundingdino_swint_ogc.pth")  # 11GB
lora_model = PeftModel.from_pretrained(base_model, "experiments/exp1/teachers/dino_lora/")   # +19MB

# Pros: 
#  - One base model, many adapters
#  - Easy to switch tasks
#  - Save disk space
# Cons:
#  - Slight inference overhead from adapter

# Option 2: Merge adapter into base (recommended for deployment)
merged_model = lora_model.merge_and_unload()  # Creates 11GB fine-tuned model
torch.save(merged_model.state_dict(), "experiments/exp1/teachers/grounding_dino_merged.pth")

# Pros:
#  - Faster inference (no adapter overhead)
#  - Single file deployment
# Cons:
#  - Full model size (11GB)
#  - One merged file per task
```

**Recommendation for This Project:**

```python
# During distillation (our use case):
# Use merged models for faster inference during distillation training
teacher = GroundedSAM(
    grounding_dino_base="pretrained/groundingdino_swint_ogc.pth",
    grounding_dino_lora="teachers/grounding_dino_lora/",
    sam_base="pretrained/sam_vit_h_4b8939.pth",
    sam_lora="teachers/sam_lora/",
    use_merged=True  # ← Merge for 10-15% faster distillation inference
)
# Internally creates merged models in memory, discards after distillation
```

#### Step 2.2: Fine-tune SAM (Segmentation) with LoRA

**When to fine-tune:** If annotations contain masks
**Skip if:** Only boxes available - can use pretrained SAM or skip segmentation

```python
# CLI Command (Simplified):
# Same command trains both teachers (if both annotations present):
# python cli/train_teacher.py \
#     --data data/raw/train.json \
#     --val data/raw/val.json \
#     --output experiments/my_experiment \
#     --gpu 0

# Default config: configs/defaults/teacher_sam_lora.yaml
from peft import LoraConfig, get_peft_model

sam_config = {
    "base_checkpoint": "data/models/pretrained/sam_vit_h_4b8939.pth",
    "model_type": "vit_h",
    
    # LoRA Configuration (PEFT) - Efficient Mask Decoder Fine-tuning
    "use_lora": True,
    "lora_config": {
        "r": 8,                     # Lower rank for decoder (smaller)
        "lora_alpha": 16,
        "target_modules": [          # Apply LoRA to mask decoder only
            "mask_decoder.transformer.layers.*.self_attn.q_proj",
            "mask_decoder.transformer.layers.*.self_attn.k_proj",
            "mask_decoder.transformer.layers.*.self_attn.v_proj",
            "mask_decoder.output_upscaling.*.weight"
        ],
        "lora_dropout": 0.05,
        "bias": "none",
    },
    
    # Freezing strategy (Partial Freeze + LoRA)
    "freeze_image_encoder": True,   # Keep ViT backbone frozen (308M params) ❄️
    "freeze_prompt_encoder": True,  # Keep prompt encoder frozen (3.8M params) ❄️
    "train_mask_decoder": True,     # Fine-tune decoder with LoRA (4.1M params → 0.4M trainable) 🔥
    
    # Training hyperparameters
    "learning_rate": 5e-4,   # Higher LR for LoRA
    "batch_size": 16,        # 2x larger due to LoRA memory savings
    "epochs": 100,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "mixed_precision": "fp16",
    
    # Prompt strategy during fine-tuning
    "prompt_type": "boxes",  # Use GT boxes as prompts (auto-gen if masks-only)
    "multimask_output": True,
}

# LoRA benefits for SAM:
# - Memory: 8GB GPU (vs 20GB+ full decoder fine-tuning)
# - Trainable params: ~380K (vs 5.6M decoder parameters)
# - Training time: 2x faster
# - Adapter size: Only 1.5MB!
#
# 🔑 IMPORTANT: What LoRA Training Saves:
# Training output: adapter_model.bin (~1.5MB) + adapter_config.json
# This is NOT a full model! It's just the "delta" from base SAM model.
#
# At inference, you need BOTH:
#   1. Base SAM model (2.4GB) - download once, reuse for all tasks
#   2. LoRA adapter (1.5MB) - one per task/dataset

# Expected results after LoRA fine-tuning:
# - Mask IoU: 0.90-0.96 on your domain (comparable to full fine-tuning!)
# - Better mask quality for your specific object types
```

#### Step 2.3: Data-Driven Teacher Selection

**Teacher Model Selection Based on Data:**

| Data Available | Grounding DINO | SAM | Reasoning |
|----------------|----------------|-----|-----------|
| **Boxes only** | ✅ Fine-tune | ❌ **NOT NEEDED** | Student outputs boxes only, no masks |
| **Masks only** | 🔧 Optional | ✅ Fine-tune | SAM needs box prompts (auto-generated from masks) |
| **Both boxes + masks** | ✅ Fine-tune | ✅ Fine-tune | Student outputs boxes + masks (recommended) |

```python
# Data-driven teacher selection
# CLI: python cli/train_teacher.py --data train.json --val val.json

# Inspect dataset
dataset_info = inspect_dataset(load_json("train.json"))

# Load teachers based on data presence
teachers = {}

if dataset_info['has_boxes']:
    # Has bounding box annotations → train Grounding DINO
    teachers['grounding_dino'] = train_grounding_dino(dataset)
    # Why? Student will output boxes, needs DINO teacher
    
if dataset_info['has_masks']:
    # Has mask annotations → train SAM
    # Auto-generate boxes from masks if needed for SAM prompts
    teachers['sam'] = train_sam(dataset)
    # Why? Student will output masks, needs SAM teacher

# Result: Only train what's needed based on available annotations
# - Boxes only: {'grounding_dino': model}
# - Masks only: {'sam': model}  
# - Both: {'grounding_dino': model, 'sam': model}
```

#### Step 2.4: Validate Fine-tuned Teacher

```python
# CLI Command (Simple):
python cli/evaluate.py \
    --teacher-dir experiments/my_experiment \
    --data data/raw/val.json

# Platform automatically:
# 1. Finds all trained teachers in experiments/my_experiment/teachers/
# 2. Loads base models + LoRA adapters for each
# 3. Evaluates each teacher on validation set
# 4. Reports metrics for loaded models

# Output example (if both teachers trained):
# ✓ Grounding DINO: mAP50 = 0.90
# ✓ SAM: Mask IoU = 0.93

# Python API (data-driven):
from ml_engine.models.teacher import load_teachers
from peft import PeftModel

# Load all available teachers from experiment directory
teachers = load_teachers("experiments/my_experiment/teachers/")
# Returns dict with only actually trained models:
# - Boxes only: {'grounding_dino': model}
# - Masks only: {'sam': model}
# - Both: {'grounding_dino': model, 'sam': model}

# Load config (auto-generated during training)
config = load_config('experiments/my_experiment/teacher_config.yaml')

# Evaluate
for model_name, model in teachers.items():
    metrics = model.evaluate(
    dataset="data/raw/val.json",
        class_prompts=config['class_mapping']
    )
    print(f"{model_name}: {metrics}")

# Target metrics before distillation:
# - Grounding DINO: mAP50 > 0.85
# - SAM: Mask IoU > 0.88
```

### Stage 3: Student Model Selection

Select student model based on available annotations and deployment requirements:

**Data-Driven Selection:**

| Available Annotations | Recommended Student | Size | Speed | Outputs |
|----------------------|---------------------|------|-------|---------|
| **Boxes only** | YOLOv8s | 11MB | 50+ FPS | boxes + class_ids |
| **Boxes only** (edge) | YOLOv8n | 3MB | 80+ FPS | boxes + class_ids |
| **Masks only** | FastSAM-s | 23MB | 25+ FPS | masks + class_ids |
| **Masks only** (edge) | MobileSAM | 10MB | 30+ FPS | masks + class_ids |
| **Both boxes + masks** | **YOLOv8s-seg** | 11.8MB | 40+ FPS | boxes + masks + class_ids |
| **Both** (edge) | YOLOv8n-seg | 3.4MB | 60+ FPS | boxes + masks + class_ids |

**Detailed Model Comparison:**

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| **YOLOv8n** | 3MB | 80+ FPS | Detection only, ultra-fast |
| **YOLOv8s** | 11MB | 50+ FPS | Detection only, balanced ✅ |
| **YOLOv8n-seg** | 3.4MB | 60+ FPS | Detection + Segmentation, fast |
| **YOLOv8s-seg** | 11.8MB | 40+ FPS | Detection + Segmentation, accurate ✅ |
| **FastSAM-s** | 23MB | 25+ FPS | Segmentation-only, high quality |
| **MobileSAM** | 10MB | 30+ FPS | Segmentation-only, mobile-optimized |

**CLI Command to Select Student:**
```bash
# Automatic selection based on dataset (recommended)
python cli/train_student.py \
    --data data/raw/train.json \
    --teacher-dir experiments/my_experiment

# Platform inspects data:
# - has_boxes + has_masks → auto-selects yolov8s-seg
# - has_boxes only → auto-selects yolov8s
# - has_masks only → auto-selects fastsam
    
# Manual override if needed
python cli/train_student.py \
    --data data/raw/train.json \
    --teacher-dir experiments/my_experiment \
    --student yolov8n-seg  # Force specific model
```

**Recommendation for Industrial Inspection:**
- **If both boxes + masks**: Use **YOLOv8s-seg** (11.8MB)
- **If only boxes**: Use **YOLOv8s** (11MB)
- **If only masks**: Use **FastSAM-s** or **MobileSAM**
- **For edge-constrained devices**: Use "n" variants (YOLOv8n, YOLOv8n-seg)

### Stage 4: Knowledge Distillation (Prompt-Free Training)

**🔑 PREREQUISITE: Use LoRA-Adapted Fine-Tuned Teachers!**

Before starting distillation, ensure you have:
- ✅ **Base pretrained models**: `data/models/pretrained/groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`
- ✅ **Fine-tuned LoRA adapters**: From Stage 2 teacher fine-tuning
  - `experiments/{experiment_name}/teachers/grounding_dino_lora/` (adapter_config.json, adapter_model.bin)
  - `experiments/{experiment_name}/teachers/sam_lora/` (adapter_config.json, adapter_model.bin)

**Why This Matters:**
- ❌ Using pretrained base models only → Poor student performance (0.65-0.75 mAP)
- ✅ Using LoRA-adapted fine-tuned teachers → Good student performance (0.85-0.92 mAP)

This is where the student learns to predict **without prompts** by learning from the **fine-tuned teacher**.

#### Step 4.1: Distillation Training Loop

```python
# Distillation trainer: ml_engine/training/distillation.py
class DistillationTrainer:
    def __init__(self, teachers, student, config):
        """
        Args:
            teachers: Dict of loaded teacher models, e.g., {'grounding_dino': model, 'sam': model}
            student: Student model
            config: Distillation config (auto-generated from data + defaults)
        """
        self.teachers = teachers  # Only contains actually loaded teachers
        self.student = student
        self.config = config
        
        # CRITICAL: Class mapping auto-filled from dataset
        self.class_mapping = self.config['class_mapping']
        # Example: {0: "ear of bag", 1: "defect", 2: "label"}
    
    def training_step(self, batch):
        images = batch['images']
        gt_annotations = batch['annotations']
        
        # === Teacher Forward Pass (RUN LOADED TEACHERS) ===
        teacher_outputs = {}
        with torch.no_grad():
            for class_id, class_name in self.class_mapping.items():
                teacher_out = {}
                
                # Stage 1: Grounding DINO (if loaded)
                if 'grounding_dino' in self.teachers:
                    dino_result = self.teachers['grounding_dino'].predict(
                        images, text_prompt=class_name
                    )
                    teacher_out['boxes'] = dino_result.boxes
                    teacher_out['dino_features'] = dino_result.features
                    teacher_out['logits'] = dino_result.logits
                
                # Stage 2: SAM (if loaded)
                if 'sam' in self.teachers:
                    # Get box prompts from DINO output, GT, or generate from masks
                    box_prompts = (teacher_out.get('boxes') or 
                                   gt_annotations.get('boxes') or 
                                   self._boxes_from_masks(gt_annotations.get('masks')))
                    
                    sam_result = self.teachers['sam'].predict(
                        images, box_prompts=box_prompts
                    )
                    teacher_out['masks'] = sam_result.masks
                    teacher_out['sam_features'] = sam_result.features
                
                teacher_outputs[class_id] = teacher_out
        
        # === Student Forward Pass (SINGLE-STAGE, NO PROMPTS!) ===
        student_predictions = self.student(images)
        # Output structure adapts to student model type
        
        # === Loss Calculation (DATA-DRIVEN) ===
        loss = self.compute_distillation_loss(
            student_predictions, teacher_outputs, gt_annotations
        )
        return loss
    
    def compute_distillation_loss(self, student_pred, teacher_out, gt):
        """
        Compute loss using only available components from data and models.
        No mode checking - just check what's actually present!
        """
        total_loss = 0.0
        weights = self.config['loss_weights']
        
        # Compute loss for whatever is available
        if 'boxes' in student_pred and 'boxes' in teacher_out:
            total_loss += weights['detection'] * self._detection_loss(
                student_pred['boxes'], teacher_out['boxes'], gt['class_ids']
            )
                
        if 'masks' in student_pred and 'masks' in teacher_out:
            total_loss += weights['segmentation'] * self._segmentation_loss(
                student_pred['masks'], teacher_out['masks']
            )
                
        if 'dino_features' in teacher_out or 'sam_features' in teacher_out:
            total_loss += weights['feature'] * self._feature_loss(
                student_pred['features'], teacher_out
            )
                
        if 'logits' in teacher_out:
            total_loss += weights['logit'] * self._kl_loss(
                student_pred['logits'], teacher_out['logits']
            )
        
        return total_loss
    
    def _boxes_from_masks(self, masks):
        """
        Helper: Generate tight bounding boxes from segmentation masks.
        
        CRITICAL: This processes each mask annotation individually.
        - Input: List of masks, where each mask is one annotation/object instance
        - Output: List of boxes, one per mask
        
        Example:
            If COCO annotation has 2 separate object instances with masks:
            masks = [mask_obj1, mask_obj2]  # 2 separate masks
            → returns [box_obj1, box_obj2]   # 2 separate tight boxes
            
            NOT: One large box encompassing both objects!
        """
        if masks is None:
            return None
        
        boxes = []
        # Each mask is processed individually - one mask → one tight box
        for mask in masks:
            # Find all pixels belonging to this specific mask
            y_indices, x_indices = torch.where(mask > 0.5)
            
            if len(x_indices) > 0:
                # Compute tight bounding box for THIS mask only
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        
        return torch.tensor(boxes) if boxes else None
```

**Key Insight for Prompt-Free Learning:**

```python
# CRITICAL ARCHITECTURAL DIFFERENCE:

# Load class mapping from config (NOT hardcoded!)
class_mapping = config['class_mapping']  # e.g., {0: "ear of bag", 1: "defect", 2: "label"}

# Teacher (Two-Stage Sequential):
# Input: Image + class_mapping[0] → DINO → boxes → SAM → masks
# Takes 150ms, needs text prompt + box prompts

# Student (Single-Stage End-to-End):
# Input: Image → Student → boxes + masks + class_ids
# Takes 8ms, NO prompts needed!

# During training, this mapping is established:
for class_id, class_name in class_mapping.items():
    # Student Class ID → Learn patterns from Teacher(class_name)
    # e.g., Student Class 0 → Learn from Teacher("ear of bag")

# After training, student's weights contain this knowledge:
# Input: Image → Student predicts class 0, 1, or 2 directly in one pass!
# No text prompt needed - the class meanings are "baked in"
# Single forward pass - no sequential pipeline needed!
```

#### Step 4.2: Unified Distillation Configuration

The platform uses **ONE config with conditional components** that automatically adapt:

```yaml
# configs/defaults/distillation.yaml
# Platform auto-fills class_mapping from COCO categories - NO manual editing!

distillation:
  # Class mapping (AUTO-FILLED from your COCO dataset's categories)
  # Platform reads this during: python cli/train_student.py --data train.json
  class_mapping:  # {} - filled automatically from COCO
  
  # Loss components - ALL included, weights auto-adjusted based on available data
  # If boxes not in data → detection_weight becomes 0 automatically
  # If masks not in data → segmentation_weight becomes 0 automatically
  loss_weights:
    detection: 0.3        # Auto-disabled if no boxes in dataset
    segmentation: 0.3     # Auto-disabled if no masks in dataset  
    logit: 0.2            # Auto-disabled if no DINO teacher
    feature: 0.2          # From whichever teachers are loaded
  
  temperature: 4.0
  
  # Training hyperparameters (same for all annotation types)
  training:
    epochs: 300
    batch_size: 32
    learning_rate: 1e-3
    optimizer: "SGD"
    momentum: 0.937
    weight_decay: 0.0005
    grad_clip: 10.0
    
  # Learning rate schedule
  scheduler:
    type: "cosine"
    warmup_epochs: 3
    warmup_lr: 1e-5
    min_lr: 1e-5
    
  # Data augmentation
  augmentation:
    mosaic: 0.5
    mixup: 0.1
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 0.0
    translate: 0.1
    scale: 0.5
    shear: 0.0
    flipud: 0.0
    fliplr: 0.5
    
  # Evaluation
  evaluation:
    interval: 10
    save_best: true
    metric: "mAP50"
```

**How Conditional Loss Works:**

```python
# ml_engine/training/distillation.py

class DistillationTrainer:
    def compute_loss(self, student_pred, teacher_out, config):
        """Compute loss using only available components."""
        total_loss = 0.0
        weights = config['loss_weights']
        
        # Automatically include/exclude based on data presence
        if 'boxes' in student_pred and 'boxes' in teacher_out:
            total_loss += weights['detection'] * detection_loss(...)
        
        if 'masks' in student_pred and 'masks' in teacher_out:
            total_loss += weights['segmentation'] * segmentation_loss(...)
        
        if 'logits' in teacher_out:
            total_loss += weights['logit'] * kl_divergence(...)
        
        if 'dino_features' in teacher_out or 'sam_features' in teacher_out:
            total_loss += weights['feature'] * feature_loss(...)
        
        return total_loss
        # No mode checking - just check what's actually present in the data!
```

#### Step 4.3: Run Distillation Training

```python
# Script: scripts/train_distillation.py
from ml_engine.training.distillation import DistillationTrainer
from ml_engine.models.teacher import GroundedSAM
from ml_engine.models.student import YOLOv8Seg
from peft import PeftModel

# 🔑 CRITICAL: Load FINE-TUNED teacher (NOT pretrained base model!)
# The teacher must be fine-tuned on your domain for effective distillation

# The GroundedSAM class loads base models + LoRA adapters:
teacher = GroundedSAM(
    # === Grounding DINO (Partial Freeze + LoRA) ===
    grounding_dino_base="data/models/pretrained/groundingdino_swint_ogc.pth",  # Base (11GB)
    grounding_dino_lora="experiments/exp1/teachers/grounding_dino_lora/",  # LoRA adapters (19MB)
    # Backbone frozen, LoRA applied to decoder attention
    
    # === SAM (Partial Freeze + LoRA) ===
    sam_base="data/models/pretrained/sam_vit_h_4b8939.pth",  # Base (2.4GB)
    sam_lora="experiments/exp1/teachers/sam_lora/",  # LoRA adapters (1.5MB)
    # Image encoder frozen, LoRA on mask decoder
    
    # Merge for faster inference during distillation
    use_merged=True  # Merge adapters into base model in memory (10-15% faster)
)

# The GroundedSAM class internally:
# 1. Loads pretrained base models (frozen parts)
# 2. Applies PEFT LoRA adapters using PeftModel.from_pretrained()
# 3. If use_merged=True: Merges adapters into base model for speed
# 4. Returns fully assembled fine-tuned teacher model

# Initialize student
student = YOLOv8Seg(
    model_size="s",  # or "n" for smaller, "m" for larger
    num_classes=3,   # Based on your dataset
    input_size=640
)

# Create distillation trainer
# Config auto-generated and saved during CLI call
config = load_config("experiments/exp1/distillation_config.yaml")

trainer = DistillationTrainer(
    teachers=teachers,  # Dict of loaded teachers
    student=student,
    config=config
)

# Train student model (PROMPT-FREE)
trainer.train(
    train_dataset="data/raw/train.json",
    val_dataset="data/raw/val.json",
    output_dir="experiments/exp1/student",
    device="cuda",
    num_workers=8
)

# Expected training time:
# - With RTX 4090: ~6-12 hours for 300 epochs on 1000 images
# - With A100: ~4-8 hours

# Expected results after distillation:
# - mAP50: 0.85-0.92 (vs fine-tuned teacher 0.90-0.95)
# - Mask IoU: 0.86-0.90 (vs fine-tuned teacher 0.92-0.96)
# - Inference speed: 8ms vs 150ms (19x faster!)
# - Model size: 11.8MB vs 2.9GB (245x smaller!)
```

**🔑 CRITICAL NOTE: Why LoRA-Adapted Teachers Are Essential**

```python
# ❌ WRONG: Using pretrained base models (NOT fine-tuned on your domain!)
teacher = GroundedSAM(
    grounding_dino="pretrained/groundingdino_swint_ogc.pth",  # Base model only
    sam="pretrained/sam_vit_h_4b8939.pth"                     # Base model only
)
# This will give POOR results because:
# - Teacher not adapted to your domain (e.g., industrial bags)
# - Student learns generic features, not domain-specific patterns
# - Final accuracy much lower than expected

# ✅ CORRECT: Using LoRA-adapted fine-tuned teachers
teacher = GroundedSAM(
    grounding_dino_base="pretrained/groundingdino_swint_ogc.pth",
    grounding_dino_lora="teachers/grounding_dino_lora/",  # ← Fine-tuned adapters!
    sam_base="pretrained/sam_vit_h_4b8939.pth",
    sam_lora="teachers/sam_lora/",                        # ← Fine-tuned adapters!
    use_merged=True
)
# This gives GOOD results because:
# - Teacher adapted to your domain (fine-tuned on your dataset)
# - Student learns domain-specific patterns (ears, defects, labels)
# - Final accuracy close to fine-tuned teacher performance
```

**How PEFT LoRA Loading Works:**

```python
# Inside GroundedSAM class (ml_engine/models/teacher/grounded_sam.py):

from peft import PeftModel
import torch

class GroundedSAM:
    def __init__(self, grounding_dino_base, grounding_dino_lora, 
                 sam_base, sam_lora, use_merged=True):
        # Step 1: Load base pretrained models
        self.base_dino = load_grounding_dino(grounding_dino_base)
        self.base_sam = load_sam(sam_base)
        
        # Step 2: Apply LoRA adapters (fine-tuned weights)
        self.grounding_dino = PeftModel.from_pretrained(
            self.base_dino, 
            grounding_dino_lora  # LoRA adapter directory from fine-tuning
        )
        self.sam = PeftModel.from_pretrained(
            self.base_sam,
            sam_lora  # LoRA adapter directory from fine-tuning
        )
        
        # Step 3 (Optional): Merge adapters into base model for faster inference
        if use_merged:
            self.grounding_dino = self.grounding_dino.merge_and_unload()
            self.sam = self.sam.merge_and_unload()
            # After merge: Base model + LoRA = single fine-tuned model
            # Faster inference, no adapter overhead

# Directory structure of LoRA adapters:
# experiments/{experiment_name}/teachers/grounding_dino_lora/
#   ├── adapter_config.json         # LoRA configuration
#   ├── adapter_model.bin           # LoRA weights (only ~19MB!)
#   └── ...
#
# experiments/{experiment_name}/teachers/sam_lora/
#   ├── adapter_config.json
#   ├── adapter_model.bin           # LoRA weights (only ~1.5MB!)
#   └── ...
```

**Why This Matters for Distillation:**

| Teacher Type | Domain Adaptation | Distillation Quality | Student Performance |
|--------------|-------------------|---------------------|---------------------|
| **Pretrained only** | ❌ Generic | Poor | 0.65-0.75 mAP |
| **Fine-tuned (LoRA)** | ✅ Your domain | Good | 0.85-0.92 mAP |
| **Fully fine-tuned** | ✅ Your domain | Good | 0.85-0.92 mAP |

**Key Insight**: LoRA-adapted teachers provide **comparable distillation quality** to fully fine-tuned teachers, but with **much lower memory** and **faster training**!

```

### Stage 5: Model Optimization

#### 5.1 Quantization

```python
# Post-Training Quantization (PTQ)
# INT8 quantization: 4x smaller, ~2-3x faster

# Dynamic Quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# Static Quantization (better accuracy)
# 1. Fuse layers (Conv+BN+ReLU)
fused_model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

# 2. Calibrate with representative data
calibration_loader = DataLoader(calibration_set, batch_size=32)
quantized_model = quantize_static(fused_model, calibration_loader)

# Expected results:
# - Size: 11.8MB → 3MB (4x reduction)
# - Speed: 1.5-2x faster on CPU
# - Accuracy: -1% to -3% mAP
```

#### 5.2 Pruning (Optional)

```python
# Structured pruning: Remove entire channels/filters
import torch.nn.utils.prune as prune

# L1 unstructured pruning
parameters_to_prune = [
    (module, 'weight') for module in model.modules()
    if isinstance(module, torch.nn.Conv2d)
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,  # Prune 30% of weights
)

# Expected results:
# - Size: 11.8MB → 8-9MB
# - Speed: 1.2-1.5x faster
# - Accuracy: -2% to -5% mAP
```

#### 5.3 ONNX Export

```python
# Export to ONNX for cross-platform deployment
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "student_model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['boxes', 'scores', 'classes', 'masks'],
    dynamic_axes={
        'images': {0: 'batch_size', 2: 'height', 3: 'width'},
        'boxes': {0: 'batch_size'},
        'masks': {0: 'batch_size'},
    }
)

# Optimize ONNX model
import onnxruntime as ort
session = ort.InferenceSession(
    "student_model.onnx",
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

#### 5.4 TensorRT Export (NVIDIA Devices)

```python
# For Jetson Nano, Xavier, Orin
import torch2trt

# Convert to TensorRT
model_trt = torch2trt(
    model,
    [dummy_input],
    fp16_mode=True,  # FP16 for speed
    max_batch_size=8,
    max_workspace_size=1 << 30  # 1GB
)

# Save TensorRT engine
torch.save(model_trt.state_dict(), 'student_model_trt.pth')

# Expected results on Jetson Orin:
# - YOLOv8s-seg: ~60 FPS at 640x640
# - YOLOv8n-seg: ~100+ FPS at 640x640
```

#### 5.5 TFLite Export (Mobile/ARM Devices)

```python
# For mobile phones, Raspberry Pi
# First convert to TensorFlow
import tf2onnx
import tensorflow as tf

# ONNX → TensorFlow
tf_model = tf2onnx.convert.from_onnx("student_model.onnx")

# TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16

tflite_model = converter.convert()

with open('student_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Stage 6: Evaluation and Benchmarking

**Metrics to Track:**

```python
# Accuracy Metrics
- mAP50: Mean Average Precision at IoU=0.5
- mAP50-95: Mean AP across IoU thresholds
- Precision, Recall, F1-Score
- Mask IoU (for segmentation)

# Efficiency Metrics
- Model Size (MB)
- Inference Time (ms)
- FPS (Frames Per Second)
- Memory Usage (MB)
- Power Consumption (W)

# Comparison Matrix
|              | Teacher (Grounded SAM) | Student (YOLOv8s-seg) | Ratio |
|--------------|------------------------|----------------------|-------|
| Size         | 2.9 GB                 | 11.8 MB              | 245x  |
| Speed (CPU)  | 3000 ms                | 100 ms               | 30x   |
| Speed (GPU)  | 150 ms                 | 8 ms                 | 19x   |
| mAP50        | 0.95 (baseline)        | 0.89 (-6%)           | 94%   |
| Power        | 250W                   | 15W                  | 17x   |
```

## Hardware Requirements

### Development/Training Environment
- GPU: NVIDIA RTX 3090/4090 or A100 (24GB+ VRAM)
- CPU: 16+ cores
- RAM: 64GB+
- Storage: 500GB+ SSD

### Edge Deployment Targets
- **Jetson Orin**: Best choice (32 TOPS AI, 8-64GB RAM)
- **Jetson Xavier NX**: Good balance (21 TOPS AI, 8GB RAM)
- **Jetson Nano**: Budget option (472 GFLOPS, 4GB RAM) - use smallest models
- **Raspberry Pi 4/5**: CPU-only, use INT8 quantized models
- **Mobile Phones**: Use TFLite, FP16 quantization

## Expected Results (Based on Similar Projects)

For industrial inspection with ~500-2000 images:

| Metric | Grounded SAM (Teacher) | YOLOv8s-seg (Student) |
|--------|------------------------|----------------------|
| mAP50 | 0.92-0.96 | 0.85-0.91 |
| Size | 2.9GB | 11.8MB → 3MB (INT8) |
| Jetson Orin FPS | 5-8 | 60-80 (FP16) |
| Jetson Nano FPS | 1-2 | 15-25 (INT8) |

## Risk Mitigation

### Risk 1: Small Dataset → Poor Generalization
**Solution:**
- Use strong data augmentation
- Generate pseudo-labels on unlabeled data
- Use few-shot learning techniques
- Transfer learning from similar domains

### Risk 2: Teacher-Student Gap Too Large
**Solution:**
- Use intermediate-sized student (YOLOv8m instead of YOLOv8n)
- Multi-stage distillation (Teacher → Medium → Small)
- Feature alignment losses

### Risk 3: Edge Device Performance
**Solution:**
- Profile on target device early
- Use hardware-specific optimizations (TensorRT, OpenVINO)
- Consider model ensembles for accuracy vs speed trade-off

## Implementation Timeline Estimate

1. **Week 1-2**: Platform setup, API development
2. **Week 3**: Data pipeline implementation
3. **Week 4**: Teacher model integration
4. **Week 5-6**: Student training pipeline
5. **Week 7-8**: Distillation implementation
6. **Week 9**: Optimization pipeline (ONNX, TensorRT)
7. **Week 10**: Testing and benchmarking
8. **Week 11-12**: Edge deployment and validation

## Recommended Starting Point

1. **Fine-tune** Grounding DINO and SAM on your labeled COCO dataset
2. Use **YOLOv8s-seg** as student model (best balance)
3. **Distill** with fixed class mapping (prompt-free training)
4. **Optimize** with INT8 quantization + TensorRT
5. Target **Jetson Orin** as primary edge device
6. Build FastAPI backend for dataset management and training orchestration

## Summary: Ensuring Prompt-Free Deployment

| Stage | Teacher (Grounded SAM) | Student (YOLOv8-seg) |
|-------|------------------------|----------------------|
| **Architecture** | Two-stage sequential (DINO→SAM) | Single-stage end-to-end |
| **Training** | Uses prompts (from config['class_mapping']) | Learns from teacher WITHOUT prompts |
| **Class Mapping** | Text → Visual features | Class ID → Visual features |
| **Knowledge** | Open-vocabulary (any text) | Fixed vocabulary (0, 1, 2, ...) |
| **Deployment** | ❌ Cannot deploy (needs prompts) | ✅ Fully prompt-free |
| **Inference** | `predict(image, class_name)` then SAM | `predict(image)` → returns all classes |
| **Speed** | 150ms (sequential, 2 models) | 8ms (single pass, 1 model) |
| **Edge Compatible** | ❌ Too large (2.9GB) | ✅ Optimized (3MB) |

**The key is**: During distillation, the teacher uses prompts to generate targets, but the student learns to predict the same outputs WITHOUT any prompts by encoding the class information directly into its weights.

## 🔒 Prompt-Free Validation & Testing

### Validation Checklist

Before deployment, verify the student model is truly prompt-free:

#### ✅ Check 1: Model Interface
```python
# File: ml_engine/models/student/yolov8.py

class YOLOv8Seg(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 640):
        """
        Args:
            num_classes: Fixed number of classes (e.g., 3)
            input_size: Input image size
        
        Note: NO prompt-related parameters!
        """
        super().__init__()
        self.num_classes = num_classes
        # ... model architecture ...
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Dict with:
                - 'boxes': [B, N, 4] bounding boxes
                - 'masks': [B, N, H, W] segmentation masks
                - 'class_ids': [B, N] class predictions (0 to num_classes-1)
                - 'scores': [B, N] confidence scores
        
        ✅ CRITICAL: NO prompt inputs in forward pass!
        """
        features = self.backbone(x)
        boxes, masks, class_logits = self.head(features)
        class_ids = torch.argmax(class_logits, dim=-1)
        
        return {
            'boxes': boxes,
            'masks': masks,
            'class_ids': class_ids,
            'scores': torch.sigmoid(class_logits).max(dim=-1)[0]
        }
```

#### ✅ Check 2: ONNX Export Verification
```python
import onnx

# Export student model
torch.onnx.export(
    student_model,
    dummy_input,
    "student_model.onnx",
    input_names=['images'],  # Only image input!
    output_names=['boxes', 'masks', 'class_ids', 'scores']
)

# Verify ONNX model has no prompt inputs
model = onnx.load("student_model.onnx")
assert len(model.graph.input) == 1, "Model should have only 1 input (images)"
assert model.graph.input[0].name == 'images'

print("✅ Model is prompt-free!")
```

#### ✅ Check 3: Automated Test Suite
```python
# Test script: tests/test_prompt_free.py

def test_student_is_prompt_free():
    """Ensure student model doesn't accept or need prompts"""
    
    # Load student model
    student = load_student("models/students/yolov8s_seg/best.onnx")
    
    # Test 1: Check input signature
    inputs = student.get_inputs()
    assert len(inputs) == 1, "Should have only 1 input"
    assert inputs[0].name == "images", "Input should be named 'images'"
    
    # Test 2: Inference without prompts
    image = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = student.run(None, {'images': image})
    
    assert len(outputs) >= 3, "Should output boxes, masks, class_ids"
    boxes, masks, class_ids = outputs[:3]
    
    # Test 3: Verify class_ids are integers
    assert class_ids.dtype in [np.int32, np.int64], "Class IDs should be integers"
    assert np.all(class_ids >= 0), "Class IDs should be non-negative"
    assert np.all(class_ids < num_classes), f"Class IDs should be < {num_classes}"
    
    print("✅ All tests passed - model is prompt-free!")

# Run test
test_student_is_prompt_free()
```

#### ✅ Check 4: Edge Deployment Verification
```python
# TensorRT deployment on Jetson
import tensorrt as trt

# Build TensorRT engine from ONNX
with trt.Builder(logger) as builder:
    engine = builder.build_cuda_engine(
        onnx_model,
        input_shape=(1, 3, 640, 640)  # Only image input
    )

# Inference
context = engine.create_execution_context()
outputs = context.execute_v2([image_buffer])

# Result: Direct predictions without any prompts
# class_ids will be integers: 0, 1, 2, ...
# Load class names from config (deployed alongside model)
class_names = load_class_mapping("configs/deployment/class_mapping.yaml")
# e.g., {0: "ear of bag", 1: "defect", 2: "label"}
predicted_name = class_names[class_ids[0]]
```

### Why Student is Guaranteed Prompt-Free

**The student model is GUARANTEED to be prompt-free because:**

1. ✅ **Architecture**: Student's `forward()` method only accepts image tensor
2. ✅ **Training**: Prompts used ONLY by teacher, student never sees them
3. ✅ **Export**: ONNX/TensorRT models have single image input
4. ✅ **Deployment**: Edge inference interface has no prompt parameters
5. ✅ **Validation**: Automated tests verify no prompt parameters exist

**The class information is embedded in the student's weights during distillation, eliminating the need for runtime prompts.**

### Information Theory Perspective

#### Teacher's Knowledge Representation
```
Teacher: Text Embedding Space → Visual Feature Space → Detection
         class_name → [0.2, 0.8, ...] → boxes, masks
         (e.g., "ear of bag" from config['class_mapping'])

Open vocabulary: Can process ANY text prompt
Disadvantage: Needs prompt at inference time
```

#### Student's Knowledge Representation
```
Student: Visual Feature Space → Class Embedding Space → Detection
         Image features → [P(class 0), P(class 1), ...] → boxes, masks

Fixed vocabulary: Only knows trained classes
Advantage: NO prompt needed - class info is in weights!
```

#### Knowledge Transfer During Distillation
```
# Load class mapping from config (NOT hardcoded!)
class_mapping = config['class_mapping']  # e.g., {0: "ear of bag", 1: "defect", 2: "label"}

For each class_id, class_name in class_mapping.items():
    Teacher: class_name text → visual patterns → predictions
             (e.g., "ear of bag" from config)
    Student: visual patterns → class_id → predictions
             (learns to predict class_id directly)
    
The text prompt is used to GENERATE training targets
But the student never sees the text - only the visual patterns!

Result: Student directly maps visual patterns → class IDs
        No text intermediate step needed!
```

## Next Steps

Based on your requirements:

1. **Dataset Preparation**:
   - Ensure COCO format is correct (boxes, masks, or both)
   - Verify annotation quality
   - Define class names clearly
   
2. **Infrastructure Setup**:
   - GPU server for training (RTX 3090/4090 with 24GB for LoRA)
   - Storage for datasets and models
   - Consider Docker for reproducibility

3. **Implementation Priorities**:
   - Week 1: Data pipeline + data inspection utilities
   - Week 2-3: Fine-tune teacher models with LoRA + auto-config generation
   - Week 4-5: Implement distillation trainer (data-driven)
   - Week 6: Optimize and export models
   - Week 7: Inference & evaluation
   - Week 8: Testing & documentation

4. **Questions for You**:
   - Dataset size (number of images)?
   - Annotation type (boxes/masks/both)?
   - Number of classes?
   - Target edge device (Jetson model)?
   - Accuracy vs speed requirements?
   - Available compute resources?

## Architecture Simplifications Summary

**Eliminated Unnecessary Complexity:**

1. **❌ Removed: AnnotationMode Enum**
   - **Before**: `mode = AnnotationMode.DETECTION_ONLY`, then lookup in `PIPELINE_CONFIG[mode]`
   - **After**: `if dataset_info['has_boxes']: load_grounding_dino()`
   - **Why**: Data structure already encodes this information

2. **❌ Removed: .mode_config.json State File**
   - **Before**: Save mode to file, read in next step
   - **After**: Inspect data fresh at each step
   - **Why**: Stateless is simpler, no sync issues

3. **❌ Removed: Separate Configs per Mode**
   - **Before**: `kd_detection_only.yaml`, `kd_segmentation_only.yaml`, `kd_both.yaml`
   - **After**: One `distillation.yaml` with conditional loss components
   - **Why**: Single source of truth, no duplication

4. **❌ Removed: Manual Config Copying/Editing**
   - **Before**: `cp template.yaml config.yaml`, then `vim config.yaml`
   - **After**: `python cli/train.py --data train.json` (auto-generates config)
   - **Why**: Error-prone, users will mess up class_names

**Kept Essential Complexity:**

1. **✅ Kept: LoRA Fine-tuning Strategy**
   - Freeze backbone + LoRA on decoder = memory efficient
   - This is core value, not unnecessary abstraction

2. **✅ Kept: Two-Stage Teacher → Single-Stage Student**
   - Teacher with prompts → Student prompt-free
   - This is the fundamental innovation

3. **✅ Kept: Multi-Model Preprocessing**
   - Different models need different preprocessing (DINO 800×1333, SAM 1024×1024, YOLO 640×640)
   - This is a real requirement, not overengineering

**Result:**
- **User workflow**: 3-step manual process → 1 command
- **Code complexity**: Mode enums + lookups → Direct data inspection
- **Configuration**: 9+ template files → 4 default files
- **Maintainability**: Better (less abstraction layers)
- **Extensibility**: Better (add keypoints = add one if statement)

---

## 🎓 Design Lessons: What We Learned

### Lesson 1: Data Structures > Code

**Bad programmers worry about the code. Good programmers worry about data structures.**

Our COCO annotations already contain all the information needed:
- `'bbox'` field present → need detection
- `'segmentation'` field present → need segmentation
- `categories` list → gives us class_mapping

We don't need to "detect a mode" and save it to a file. **The data structure IS the mode.**

### Lesson 2: Eliminate Special Cases

**Good code has no special cases.**

Instead of:
```python
if mode == "detection_only":
    loss = detection_loss(...)
elif mode == "segmentation_only":
    loss = segmentation_loss(...)
else:  # mode == "both"
    loss = detection_loss(...) + segmentation_loss(...)
```

We do:
```python
loss = 0.0
if 'boxes' in outputs:
    loss += detection_loss(...)
if 'masks' in outputs:
    loss += segmentation_loss(...)
# No special cases - just compute what's available
```

### Lesson 3: YAGNI (You Ain't Gonna Need It)

**Don't build for imaginary future requirements.**

We removed:
- ❌ Modular config architecture (Phase 2) - not needed yet
- ❌ Config composition system - over-engineered
- ❌ Hardware-specific config modules - premature

We'll add these IF and WHEN we actually need them (probably never).

### Lesson 4: Automation > Configuration

**The best configuration is no configuration.**

Auto-generate from data whenever possible:
- ✅ `num_classes` from COCO categories
- ✅ `class_names` from COCO categories
- ✅ `class_mapping` from COCO categories
- ✅ Which models to load from annotation fields

Users only configure what they actually need to: batch_size, epochs, learning_rate.

### Lesson 5: Stateless > Stateful

**State is the root of all bugs.**

We eliminated `.mode_config.json` because:
- If it gets out of sync with data → bugs
- If user modifies data → must regenerate
- If file is deleted → pipeline breaks

Stateless is simpler: just read the data each time (it's fast).

## References

- **PEFT Library**: https://github.com/huggingface/peft
- **LoRA Paper**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Grounding DINO**: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- **SAM**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **Knowledge Distillation**: [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

