# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/run_tests.py --type unit
```

## Data Preparation

Your COCO dataset should look like:

```
data/raw/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── annotations.json    # Single JSON with ALL annotations
```

**COCO JSON Format:**

```json
{
  "images": [
    {"id": 1, "file_name": "img001.jpg", "width": 1920, "height": 1080},
    {"id": 2, "file_name": "img002.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],           # Optional
      "segmentation": [[x1,y1,x2,y2,...]],     # Optional
      "area": 1234                              # Optional
    }
  ],
  "categories": [
    {"id": 0, "name": "class_0"},
    {"id": 1, "name": "class_1"}
  ]
}
```

**Note**: You only need `segmentation` OR `bbox` OR both. The platform auto-generates missing fields.

## Complete Pipeline (4 Commands)

### Step 1: Validate and Split Dataset

```bash
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.7,val:0.15,test:0.15 \
    --stratify \
    --seed 42
```

**Output:**
- `data/raw/train.json` (70% of data)
- `data/raw/val.json` (15% of data)
- `data/raw/test.json` (15% of data)

**What it does:**
- ✅ Validates COCO format
- ✅ Auto-generates bbox from masks (if missing)
- ✅ Auto-computes area (if missing)
- ✅ Splits dataset with stratification
- ✅ Reports dataset statistics

### Step 2: Fine-tune Teacher Models

```bash
python cli/train_teacher.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --output experiments/exp1 \
    --batch-size 8 \
    --epochs 50
```

**What it does:**
- ✅ Inspects dataset (detects annotation types)
- ✅ Loads appropriate models (DINO if has_boxes, SAM if has_masks)
- ✅ Auto-generates config from defaults + dataset
- ✅ Applies LoRA (memory-efficient)
- ✅ Trains with gradient clipping, mixed precision
- ✅ Saves LoRA adapters (19MB + 1.5MB)

**Time:** 1-2 days on RTX 3090  
**Output:** LoRA adapters in `experiments/exp1/teachers/`

### Step 3: Train Student Model (TODO - Not Implemented Yet)

```bash
python cli/train_student.py \
    --data data/raw/train.json \
    --teacher-dir experiments/exp1 \
    --output experiments/exp1/student
```

**What it will do:**
- Load LoRA-adapted teachers
- Auto-select student model from data
- Distill knowledge (prompt-free training)
- Save student model

### Step 4: Optimize for Edge (TODO - Not Implemented Yet)

```bash
python cli/optimize_model.py \
    --model experiments/exp1/student/best.pt \
    --format onnx tensorrt \
    --quantize int8
```

**What it will do:**
- Export to ONNX
- Quantize to INT8 (4x smaller)
- Convert to TensorRT (for Jetson)

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir experiments/exp1/logs
```

Open browser to `http://localhost:6006`

### Check Checkpoints

```bash
# List checkpoints
ls experiments/exp1/teachers/grounding_dino_lora/

# Output:
# - adapter_config.json
# - adapter_model.bin
# - best.pth
# - last.pth
```

## Data-Driven Behavior

The platform automatically adapts to your data:

| Your Dataset | Models Loaded | Student Model |
|--------------|--------------|---------------|
| Only boxes | Grounding DINO | YOLOv8 |
| Only masks | SAM | YOLOv8-seg or FastSAM |
| Both | DINO + SAM | YOLOv8-seg |

**No configuration needed - just point to your data!**

## Example Datasets

### Detection Only (Boxes)

```json
{
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 50, 50]}
  ]
}
```

→ Platform trains: **Grounding DINO only**

### Segmentation Only (Masks)

```json
{
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 0, "segmentation": [[x1,y1,x2,y2,...]]}
  ]
}
```

→ Platform trains: **SAM only** (auto-generates bbox for prompts)

### Both (Recommended)

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 100, 50, 50],
      "segmentation": [[x1,y1,x2,y2,...]]
    }
  ]
}
```

→ Platform trains: **Both DINO and SAM**

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | A100 (40GB) |
| RAM | 32GB | 64GB |
| Storage | 100GB | 500GB SSD |

## Customization

### Override Hyperparameters

```bash
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --batch-size 16 \      # Override
    --epochs 100 \         # Override
    --lr 2e-4              # Override
```

### Custom Augmentation

```bash
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --aug-characteristics changes_shape reflective_surface low_contrast \
    --aug-intensity high
```

### LoRA Configuration

```bash
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --lora-r 32 \          # Larger rank = more capacity
    --lora-alpha 64 \      # Typically 2× rank
    --lora-dropout 0.05    # Lower dropout for small datasets
```

## Troubleshooting

### GPU Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch-size 4
```

**Solution 2**: Reduce LoRA rank
```bash
--lora-r 8
```

**Solution 3**: Disable mixed precision
Edit `configs/defaults/training_dynamics.yaml`:
```yaml
mixed_precision:
  enabled: false
```

### Slow Training

**Solution 1**: Increase workers
```bash
--num-workers 8
```

**Solution 2**: Use larger batch size (if GPU allows)
```bash
--batch-size 16
```

### Poor Convergence

**Solution 1**: Increase epochs
```bash
--epochs 100
```

**Solution 2**: Adjust learning rate
```bash
--lr 5e-4  # Higher for faster convergence
--lr 1e-5  # Lower for more stability
```

**Solution 3**: Stronger augmentation
```bash
--aug-intensity high
```

## Next Steps

1. **Validate your dataset**: `python cli/validate_dataset.py ...`
2. **Train teachers**: `python cli/train_teacher.py ...`
3. **Monitor training**: `tensorboard --logdir experiments/exp1/logs`
4. **Check outputs**: `ls experiments/exp1/teachers/`
5. **Proceed to distillation** (once implemented)

## FAQ

**Q: Do I need to manually edit config files?**  
A: No! All configs are auto-generated from your dataset.

**Q: What if I only have segmentation masks?**  
A: The platform auto-generates bounding boxes from masks.

**Q: Can I use my own augmentation?**  
A: Yes, either use the characteristic-based system or provide a custom pipeline.

**Q: How do I resume training?**  
A: Use `--resume experiments/exp1/teachers/dino_lora/last.pth`

**Q: Where are the full fine-tuned models?**  
A: LoRA only saves adapters (19MB). You need base model + adapters for inference.

**Q: Can I train both models on one GPU?**  
A: Yes! LoRA uses 22GB for both (fits on RTX 3090 24GB).


