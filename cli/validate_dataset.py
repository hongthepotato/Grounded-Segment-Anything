"""
CLI for COCO dataset validation and preprocessing.

This script validates COCO format datasets using the same DataManager
pipeline as the API, ensuring consistency between CLI and API workflows.

Features:
- Format validation (automatic)
- Auto-generation of missing bbox/area from masks (automatic)
- Dataset splitting (train/val/test)
- Quality checks
- Annotation mode detection
- Image path validation

Usage:
    # Basic validation
    python cli/validate_dataset.py --data annotations.json --images data/raw/images/
    
    # Validate and split
    python cli/validate_dataset.py --data annotations.json --split train:0.7,val:0.15,test:0.15
    
    # With stratification and random seed
    python cli/validate_dataset.py --data annotations.json --split train:0.8,val:0.2 --stratify --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.data.inspection import print_dataset_report
from ml_engine.data.manager import DataManager
from core.config import save_json
from core.logging_config import configure_logging, get_logger

# Configure logging using centralized configuration
configure_logging()
logger = get_logger('dataset_validation')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate and preprocess COCO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to COCO JSON file')
    parser.add_argument('--images', type=str, default=None,
                        help='Directory containing images (auto-detected if not provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for processed files')
    
    # Splitting options
    parser.add_argument('--split', type=str, default=None,
                        help='Split ratios (e.g., train:0.7,val:0.15,test:0.15)')
    parser.add_argument('--stratify', action='store_true',
                        help='Use stratified splitting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    return parser.parse_args()


def parse_split_ratios(split_str: str) -> dict:
    """
    Parse split ratio string.
    
    Args:
        split_str: String like "train:0.7,val:0.15,test:0.15"
    
    Returns:
        Dictionary with split ratios
    """
    splits = {}
    for part in split_str.split(','):
        name, ratio = part.split(':')
        splits[name.strip()] = float(ratio.strip())
    
    # Validate
    total = sum(splits.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    return splits


def main():
    """Main validation entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("COCO Dataset Validation")
    logger.info("=" * 60)
    
    # Auto-detect image directory if not provided
    if args.images is None:
        data_path = Path(args.data)
        args.images = str(data_path.parent / 'images')
        logger.info("Auto-detected image directory: %s", args.images)
    
    # Auto-detect output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.data).parent)
    
    # Collect image paths from directory
    logger.info("\n Scanning images from: %s", args.images)
    image_dir = Path(args.images)
    if not image_dir.exists():
        logger.error("❌ Image directory not found: %s", image_dir)
        sys.exit(1)
    
    # Collect all image paths (relative to match COCO file_name format)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in image_dir.rglob(ext):
            # Get relative path from parent of image directory
            # This matches the typical COCO file_name format
            try:
                rel_path = img_path.relative_to(image_dir.parent)
                image_paths.append(str(rel_path))
            except ValueError:
                # Fallback: use just the filename
                image_paths.append(img_path.name)
    
    logger.info("Found %d image files", len(image_paths))
    
    # Parse split config if provided
    split_config = None
    if args.split:
        split_config = parse_split_ratios(args.split)
        logger.info("Split configuration: %s", split_config)
        logger.info("Stratify: %s", args.stratify)
        logger.info("Random seed: %s", args.seed)
    
    # Load dataset using DataManager (same as API pipeline)
    logger.info("\n📂 Loading dataset via DataManager: %s", args.data)
    try:
        manager = DataManager.from_file(
            data_path=args.data,
            image_paths=image_paths,
            split_config=split_config,
            stratify=args.stratify,
            random_seed=args.seed
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("❌ Failed to load dataset: %s", e)
        sys.exit(1)
    
    # Dataset inspection report
    logger.info("\n Dataset Inspection Report:")
    dataset_info = manager.get_dataset_info()
    print_dataset_report(dataset_info)
    
    # Quality checks
    logger.info("\n🔍 Quality Report:")
    quality = manager.get_quality_report()
    
    if quality['warnings']:
        logger.warning("⚠️  Quality warnings:")
        for warning in quality['warnings']:
            logger.warning("  - %s", warning)
    else:
        logger.info("✓ No quality issues detected")
    
    # Required models
    required_models = manager.get_required_models()
    logger.info("\n🤖 Required Models: %s", required_models)
    logger.info("   Original annotation mode: %s", manager.original_annotation_mode)
    
    # Save splits if configured
    if split_config:
        logger.info("\n✂️  Saving dataset splits...")
        for split_name in split_config.keys():
            split_data = manager.get_split(split_name)
            output_path = Path(args.output_dir) / f'{split_name}.json'
            save_json(split_data, str(output_path))
            logger.info("✓ Saved %s: %s (%d images)", split_name, output_path, len(split_data['images']))
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset validation completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
