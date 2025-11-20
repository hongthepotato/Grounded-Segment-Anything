# CLI Usage Guide

## Overview

The platform provides command-line tools for the complete fine-tuning and distillation pipeline.

## Quick Start: 4 Commands to Production

```bash
# 1. Validate dataset (auto-generates missing bbox from masks, splits data)
python cli/validate_dataset.py \
    --data data/raw/annotations.json \
    --split train:0.7,val:0.15,test:0.15 \
    --stratify --seed 42

# 2. Fine-tune teachers (auto-detects which models needed from data)
python cli/train_teacher.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --output experiments/exp1

# 3. Distill student (auto-selects student model, auto-fills classes)
python cli/train_student.py \
    --data data/raw/train.json \
    --teacher-dir experiments/exp1 \
    --output experiments/exp1/student

# 4. Optimize for edge (quantize + export)
python cli/optimize_model.py \
    --model experiments/exp1/student/best.pt \
    --format onnx tensorrt \
    --quantize int8

# Done! Deploy to edge device.
```

## Command Reference

### 1. Dataset Validation

**Purpose**: Validate COCO format, auto-generate missing fields, split dataset.

```bash
python cli/validate_dataset.py --data <COCO_JSON> [OPTIONS]
```

**Options:**
- `--data`: Path to COCO JSON file (required)
- `--images`: Image directory (auto-detected if not provided)
- `--output-dir`: Output directory for split files
- `--split`: Split ratios (e.g., `train:0.7,val:0.15,test:0.15`)
- `--stratify`: Use stratified splitting (maintains class distribution)
- `--seed`: Random seed for reproducibility
- `--check-format`: Validate COCO format compliance
- `--check-images`: Verify image files exist
- `--fix-missing`: Auto-generate missing bbox/area from masks

**Examples:**

```bash
# Basic validation
python cli/validate_dataset.py --data annotations.json

# Validate and split
python cli/validate_dataset.py \
    --data annotations.json \
    --split train:0.7,val:0.15,test:0.15 \
    --stratify --seed 42

# Full validation with image checks
python cli/validate_dataset.py \
    --data annotations.json \
    --images data/raw/images/ \
    --check-format \
    --check-images \
    --fix-missing
```

### 2. Teacher Fine-tuning

**Purpose**: Fine-tune teacher models (Grounding DINO and/or SAM) with LoRA.

```bash
python cli/train_teacher.py --data <TRAIN_JSON> --val <VAL_JSON> --output <EXP_DIR> [OPTIONS]
```

**Options:**
- `--data`: Training COCO JSON (required)
- `--val`: Validation COCO JSON (required)
- `--images`: Image directory (auto-detected)
- `--output`: Experiment output directory (required)
- `--grounding-dino-ckpt`: Path to pretrained Grounding DINO
- `--sam-ckpt`: Path to pretrained SAM
- `--sam-type`: SAM model type (vit_h, vit_l, vit_b)
- `--batch-size`: Batch size
- `--epochs`: Number of epochs
- `--lr`: Learning rate
- `--num-workers`: DataLoader workers
- `--lora-r`: LoRA rank
- `--lora-alpha`: LoRA alpha
- `--lora-dropout`: LoRA dropout
- `--aug-characteristics`: Object characteristics for augmentation
- `--aug-intensity`: Augmentation intensity (low, medium, high)
- `--gpu`: GPU ID
- `--resume`: Resume from checkpoint
- `--experiment-name`: Custom experiment name

**Examples:**

```bash
# Basic usage (auto-detects everything)
python cli/train_teacher.py \
    --data data/raw/train.json \
    --val data/raw/val.json \
    --output experiments/exp1

# Override hyperparameters
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --batch-size 16 \
    --epochs 100 \
    --lr 2e-4

# Override LoRA configuration
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05

# Custom augmentation
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --aug-characteristics changes_shape reflective_surface \
    --aug-intensity high

# Resume training
python cli/train_teacher.py \
    --data train.json \
    --val val.json \
    --output exp1 \
    --resume experiments/exp1/teachers/grounding_dino_lora/last.pth
```

## Data-Driven Behavior

The platform automatically detects annotation types and loads appropriate models:

| Dataset Annotations | Teacher Models Loaded | Training Behavior |
|---------------------|----------------------|-------------------|
| Boxes only | Grounding DINO only | Detection fine-tuning |
| Masks only | SAM only | Segmentation fine-tuning |
| Both boxes + masks | DINO + SAM | Both fine-tuned |

**No manual configuration needed!** The platform inspects your COCO data and loads the right models.

## Auto-Configuration

All configs are auto-generated from:
1. Default config templates (`configs/defaults/`)
2. Dataset inspection (num_classes, class_names)
3. CLI overrides (optional)

**No manual config editing required!**

## Output Structure

After training, your experiment directory contains:

```
experiments/exp1/
├── teachers/
│   ├── grounding_dino_lora/     # LoRA adapters (19MB)
│   │   ├── adapter_config.json
│   │   └── adapter_model.bin
│   └── sam_lora/                # LoRA adapters (1.5MB)
│       ├── adapter_config.json
│       └── adapter_model.bin
├── teacher_config.yaml          # Auto-generated config
├── metadata.json                # Experiment metadata
└── logs/                        # TensorBoard logs
    ├── grounding_dino/
    └── sam/
```

## LoRA Fine-tuning Benefits

| Aspect | Full Fine-tuning | LoRA Fine-tuning |
|--------|-----------------|------------------|
| GPU Memory | 47GB (A100 needed) | 14GB (RTX 3090 OK) |
| Training Time | 24-36 hours | 8-12 hours |
| Checkpoint Size | 13.4GB | 20.5MB |
| Accuracy | 100% baseline | 98-99% of baseline |

## Tips

1. **Start with default configs**: The auto-generated configs work well for most cases
2. **Use stratified splitting**: Ensures balanced class distribution across splits
3. **Monitor TensorBoard**: Check `experiments/exp1/logs/` for training curves
4. **Save GPU memory**: Use smaller batch sizes if OOM errors occur
5. **LoRA rank tuning**: Try r=8 for faster training, r=32 for better accuracy

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python cli/train_teacher.py --data train.json --val val.json --batch-size 4

# Reduce LoRA rank
python cli/train_teacher.py --data train.json --val val.json --lora-r 8
```

### Slow Training

```bash
# Increase batch size (if GPU allows)
python cli/train_teacher.py --data train.json --val val.json --batch-size 16

# Use more workers
python cli/train_teacher.py --data train.json --val val.json --num-workers 8
```

### Poor Convergence

```bash
# Increase epochs
python cli/train_teacher.py --data train.json --val val.json --epochs 100

# Adjust learning rate
python cli/train_teacher.py --data train.json --val val.json --lr 5e-4

# Stronger augmentation
python cli/train_teacher.py --data train.json --val val.json --aug-intensity high
```

## Next Steps

After teacher training completes:

1. Check TensorBoard logs for training curves
2. Verify LoRA adapters were saved (`experiments/exp1/teachers/`)
3. Proceed to student distillation (`cli/train_student.py`)

## Support

For issues or questions, check:
- `TECHNICAL_APPROACH.md` for technical details
- `PLATFORM_ARCHITECTURE.md` for architecture overview
- GitHub issues for bug reports


