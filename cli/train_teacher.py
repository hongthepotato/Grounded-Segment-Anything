"""
CLI for teacher model fine-tuning with LoRA.

This script provides a simple command-line interface for fine-tuning
teacher models (Grounding DINO and/or SAM) with automatic configuration.

Usage:
    python cli/train_teacher.py --data annotations.json --images images/ --output experiments/exp1
    
Design philosophy:
- User provides ONE dataset file (all annotations)
- Platform handles EVERYTHING:
  1. Validates and auto-fixes data (bbox from masks if needed)
  2. Splits train/val/test (70%/20%/10%)
  3. Inspects dataset for annotation types
  4. Loads appropriate models (DINO if has_boxes, SAM if has_masks)
  5. Auto-generates config from defaults + dataset info
  6. Trains with LoRA (memory-efficient)
  7. Saves LoRA adapters (not full models)

User responsibility: Provide one JSON file with annotations. That's it.
Platform responsibility: Everything else.
"""

import argparse
import sys
from pathlib import Path
import os
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.training.teacher_trainer import TeacherTrainer
from ml_engine.data.manager import DataManager
from core.config import (
    save_config, create_experiment_dir, load_config, merge_configs,
    save_experiment_metadata
)
from core.logger import setup_logger
from core.constants import DEFAULT_CONFIGS_DIR


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune teacher models (Grounding DINO + SAM) with LoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (platform auto-splits dataset and auto-detects models)
  python cli/train_teacher.py --data data/raw/annotations.json --images data/raw/images --output experiments/exp1
  
  # Override hyperparameters
  python cli/train_teacher.py --data annotations.json --images images/ --output exp1 --batch-size 16 --epochs 100
  
  # Override LoRA rank
  python cli/train_teacher.py --data annotations.json --images images/ --output exp1 --lora-r 32
  
  # Resume from checkpoint
  python cli/train_teacher.py --data annotations.json --images images/ --output exp1 --resume experiments/exp1/teachers/grounding_dino_lora/last.pth

Note: User provides ONE dataset file. Platform automatically:
  - Validates and auto-fixes data (generates bbox from masks if needed)
  - Splits into train (70%), val (20%), test (10%)
  - Detects which models to train based on annotations
  - Auto-generates config from dataset
        """
    )

    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to COCO JSON file (platform will split train/val/test)')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for experiment (e.g., experiments/exp1)')

    # Model configuration
    parser.add_argument('--grounding-dino-ckpt', type=str, default=None,
                        help='Path to pretrained Grounding DINO checkpoint')
    parser.add_argument('--sam-ckpt', type=str, default=None,
                        help='Path to pretrained SAM checkpoint')
    parser.add_argument('--sam-type', type=str, default=None,
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')

    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=None,
                        help='LoRA rank (default: from config)')
    parser.add_argument('--lora-alpha', type=int, default=None,
                        help='LoRA alpha (default: from config)')
    parser.add_argument('--lora-dropout', type=float, default=None,
                        help='LoRA dropout (default: from config)')

    # Augmentation
    parser.add_argument('--aug-characteristics', type=str, nargs='+',
                        default=None,
                        help='Object characteristics for augmentation')
    parser.add_argument('--aug-intensity', type=str,
                        choices=['low', 'medium', 'high'],
                        default='medium',
                        help='Augmentation intensity')

    # Other options
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')

    return parser.parse_args()

def build_cli_overrides(args) -> Dict[str, Any]:
    """Extract CLI overrides from parsed arguments."""
    overrides = {}

    # Training hyperparameters
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.lr is not None:
        overrides['learning_rate'] = args.lr
    if args.num_workers is not None:
        overrides['num_workers'] = args.num_workers

    # LoRA configuration
    lora_overrides = {}
    if args.lora_r is not None:
        lora_overrides['r'] = args.lora_r
    if args.lora_alpha is not None:
        lora_overrides['lora_alpha'] = args.lora_alpha
    if args.lora_dropout is not None:
        lora_overrides['lora_dropout'] = args.lora_dropout
    if lora_overrides:
        overrides['lora'] = lora_overrides

    # Augmentation
    if args.aug_characteristics:
        overrides['augmentation'] = {
            'enabled': True,
            'characteristics': args.aug_characteristics,
            'intensity': args.aug_intensity
        }
        # TODO(sh/2025-11-12): Add environment overrides

    # Model checkpoints
    model_overrides = {}
    if args.grounding_dino_ckpt is not None:
        model_overrides['grounding_dino'] = {
            'base_checkpoint': args.grounding_dino_ckpt
        }
    if args.sam_ckpt is not None:
        model_overrides['sam'] = {
            'base_checkpoint': args.sam_ckpt
        }
    if args.sam_type is not None:
        if 'sam' not in model_overrides:
            model_overrides['sam'] = {}
        model_overrides['sam']['model_type'] = args.sam_type

    if model_overrides:
        overrides['model'] = model_overrides

    return overrides


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logger('teacher_training', level=20)  # INFO level

    logger.info("=" * 60)
    logger.info("Teacher Model Fine-Tuning with LoRA")
    logger.info("=" * 60)

    # Verify image directory exists
    image_dir = Path(args.images)
    if not image_dir.exists():
        logger.error("Image directory not found: %s", args.images)
        sys.exit(1)

    # Get all image paths from directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [
        str(p) for p in image_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        logger.error("No images found in directory: %s", args.images)
        sys.exit(1)
    
    logger.info("Found %d images in %s", len(image_paths), args.images)

    # Step 1: Create DataManager
    # Note: Normalization (bbox from masks, etc.) is always applied during loading
    logger.info("\n Step 1: Initializing DataManager...")

    data_manager = DataManager(
        data_path=args.data,
        image_paths=image_paths,
        split_config={'train': 0.7, 'val': 0.15, 'test': 0.15}
    )

    # Get inspection results (cached in manager)
    dataset_info = data_manager.get_dataset_info()

    cli_overrides = build_cli_overrides(args)

    # Step 2: Generate configuration
    logger.info("\n Step 2: Generating configuration...")

    # Load shared teacher training config
    logger.info("  Loading shared training config...")
    shared_config_path = DEFAULT_CONFIGS_DIR / 'teacher_training.yaml'
    try:
        shared_config = load_config(str(shared_config_path))
        logger.info("  Loaded shared training config")
    except FileNotFoundError:
        logger.error("Shared teacher training config not found: %s", shared_config_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading shared training config: %s", e)
        sys.exit(1)

    logger.info("  Loading model-specific configs...")
    required_models = data_manager.get_required_models()
    logger.info("  Required teacher models based on dataset:")
    for model_name in required_models:
        logger.info("    - %s", model_name)

    model_configs = {}

    if 'grounding_dino' in required_models:
        dino_config_path = DEFAULT_CONFIGS_DIR / 'teacher_grounding_dino_lora.yaml'
        try:
            model_configs['grounding_dino'] = load_config(str(dino_config_path))
            logger.info("  Loaded Grounding DINO config")
        except FileNotFoundError:
            logger.error("Grounding DINO config not found: %s", dino_config_path)
            sys.exit(1)
        except Exception as e:
            logger.error("Error loading Grounding DINO config: %s", e)
            sys.exit(1)

    if 'sam' in required_models:
        sam_config_path = DEFAULT_CONFIGS_DIR / 'teacher_sam_lora.yaml'
        try:
            model_configs['sam'] = load_config(str(sam_config_path))
            logger.info("  Loaded SAM config")
        except FileNotFoundError:
            logger.error("SAM config not found: %s", sam_config_path)
            sys.exit(1)
        except Exception as e:
            logger.error("Error loading SAM config: %s", e)
            sys.exit(1)

    if not model_configs:
        logger.error("No models to train! Dataset has no valid annotations.")
        sys.exit(1)

    logger.info("  Merging configs...")

    config = {
        **shared_config['training'],
        'num_classes': dataset_info['num_classes'],
        'class_names': list(dataset_info['class_mapping'].values()),
        'class_mapping': dataset_info['class_mapping'],
        'augmentation': shared_config.get('augmentation'),
        'evaluation': shared_config.get('evaluation'),
        'checkpointing': shared_config.get('checkpointing'),
        'models': model_configs
    }

    if cli_overrides:
        config = merge_configs(config, cli_overrides)
        logger.info("  Applied CLI overrides")

    logger.info(" Configuration generated successfully")

    # Step 3: Create experiment directory
    logger.info("\n Step 3: Creating experiment directory...")
    exp_dir = create_experiment_dir(
        base_dir=args.output,
        experiment_name=args.experiment_name
    )
    logger.info("Experiment directory: %s", exp_dir)

    # Save config
    config_path = exp_dir / 'teacher_config.yaml'
    save_config(config, str(config_path))
    logger.info("✓ Saved config to: %s", config_path)

    # Save metadata
    save_experiment_metadata(
        exp_dir=exp_dir,
        dataset_info=dataset_info,
        cli_args=vars(args)
    )

    # Step 4: Initialize trainer
    logger.info("\n Step 4: Initializing trainer...")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trainer = TeacherTrainer(
        data_manager=data_manager,
        output_dir=str(exp_dir),
        config=config,
        resume_from=args.resume
    )

    # Step 5: Train
    logger.info("\n Step 5: Starting training...")
    logger.info("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n  Training interrupted by user")
        logger.info("Saving current state...")
        # Save current state before exiting
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("\n Training completed successfully!")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"LoRA adapters saved to: {exp_dir / 'teachers'}")


if __name__ == '__main__':
    main()
