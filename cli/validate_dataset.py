"""
CLI for COCO dataset validation and preprocessing.

This script validates COCO format datasets and performs:
- Format validation
- Auto-generation of missing bbox/area from masks
- Dataset splitting (train/val/test)
- Quality checks
- Annotation mode detection

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

from ml_engine.data.inspection import load_and_inspect_dataset, print_dataset_report
from ml_engine.data.validators import (
    validate_coco_format,
    preprocess_coco_dataset,
    check_data_quality,
    split_dataset
)
from core.config import load_json, save_json
from core.logger import setup_logger

logger = setup_logger('dataset_validation')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate and preprocess COCO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to COCO JSON file')
    parser.add_argument('--images', type=str, default=None,
                        help='Directory containing images (for validation)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for processed files')
    
    # Splitting options
    parser.add_argument('--split', type=str, default=None,
                        help='Split ratios (e.g., train:0.7,val:0.15,test:0.15)')
    parser.add_argument('--stratify', action='store_true',
                        help='Use stratified splitting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    # Validation options
    parser.add_argument('--check-format', action='store_true',
                        help='Check COCO format compliance')
    parser.add_argument('--check-images', action='store_true',
                        help='Check if image files exist')
    parser.add_argument('--fix-missing', action='store_true',
                        help='Auto-generate missing bbox/area from masks')
    
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
        logger.info(f"Auto-detected image directory: {args.images}")
    
    # Auto-detect output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.data).parent)
    
    # Load dataset
    logger.info(f"\n📂 Loading dataset: {args.data}")
    coco_data = load_json(args.data)
    
    # Step 1: Format validation
    if args.check_format:
        logger.info("\n✓ Step 1: Validating COCO format...")
        is_valid, errors = validate_coco_format(coco_data)
        
        if not is_valid:
            logger.error("❌ COCO format validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        else:
            logger.info("✓ COCO format is valid")
    
    # Step 2: Auto-fix missing fields
    if args.fix_missing or not args.check_format:
        logger.info("\n🔧 Step 2: Auto-generating missing bbox/area...")
        coco_data = preprocess_coco_dataset(coco_data, in_place=True)
    
    # Step 3: Dataset inspection
    logger.info("\n📊 Step 3: Inspecting dataset...")
    dataset_info = load_and_inspect_dataset(args.data)
    print_dataset_report(dataset_info)
    
    # Step 4: Quality checks
    logger.info("\n🔍 Step 4: Quality checks...")
    quality = check_data_quality(coco_data)
    
    if quality['warnings']:
        logger.warning("⚠️  Quality warnings:")
        for warning in quality['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.info("✓ No quality issues detected")
    
    # Step 5: Check images exist
    if args.check_images:
        logger.info("\n🖼️  Step 5: Checking image files...")
        image_dir = Path(args.images)
        missing_images = []
        
        for img in coco_data['images'][:10]:  # Check first 10
            img_path = image_dir / img['file_name']
            if not img_path.exists():
                missing_images.append(img['file_name'])
        
        if missing_images:
            logger.warning(f"⚠️  {len(missing_images)} images not found (showing first 10)")
            for fname in missing_images[:10]:
                logger.warning(f"  - {fname}")
        else:
            logger.info("✓ All checked images exist")
    
    # Step 6: Split dataset
    if args.split:
        logger.info(f"\n✂️  Step 6: Splitting dataset...")
        split_ratios = parse_split_ratios(args.split)
        
        logger.info(f"Split ratios: {split_ratios}")
        logger.info(f"Stratify: {args.stratify}")
        logger.info(f"Random seed: {args.seed}")
        
        splits = split_dataset(
            coco_data,
            splits=split_ratios,
            stratify=args.stratify,
            random_seed=args.seed
        )
        
        # Save splits
        for split_name, split_data in splits.items():
            output_path = Path(args.output_dir) / f'{split_name}.json'
            save_json(split_data, str(output_path))
            logger.info(f"✓ Saved {split_name}: {output_path} ({len(split_data['images'])} images)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset validation completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


