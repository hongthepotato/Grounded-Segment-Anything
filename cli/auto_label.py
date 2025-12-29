#!/usr/bin/env python3
"""
Auto-labeling CLI tool.

Generate COCO-format annotations for images using Grounding DINO + MobileSAM.

Usage:
    python cli/auto_label.py \\
        --images data/raw/images/ \\
        --classes "ear of bag,defect,label" \\
        --output annotations.json

    # With custom thresholds
    python cli/auto_label.py \\
        --images data/raw/images/ \\
        --classes "ear of bag,defect" \\
        --output annotations.json \\
        --box-threshold 0.3 \\
        --nms-threshold 0.5

    # Specify GPU
    python cli/auto_label.py \\
        --images data/raw/images/ \\
        --classes "ear of bag" \\
        --output annotations.json \\
        --gpu 0
    
    # With visualization (saves annotated images)
    python cli/auto_label.py \\
        --images data/raw/images/ \\
        --classes "ear of bag,defect" \\
        --output annotations.json \\
        --visualize \\
        --viz-dir output/visualizations/
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from glob import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.utils import (
    validate_file_exists,
    setup_cuda_device,
    print_header,
    print_section,
    format_time
)

from ml_engine.inference.auto_labeler import (
    AutoLabeler,
    AutoLabelerConfig,
    visualize_detections
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label images using Grounding DINO + MobileSAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python cli/auto_label.py --images ./images/ --classes "dog,cat" --output annotations.json
    
    # With custom model paths
    python cli/auto_label.py \\
        --images ./images/ \\
        --classes "ear of bag,defect" \\
        --output annotations.json \\
        --dino-checkpoint ./groundingdino_swint_ogc.pth \\
        --sam-checkpoint ./EfficientSAM/mobile_sam.pt
        """
    )

    # Required arguments
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to images directory or single image file"
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Comma-separated class names to detect (e.g., 'ear of bag,defect,label')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for COCO JSON file"
    )

    # Model paths
    parser.add_argument(
        "--dino-config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to Grounding DINO config file"
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=str,
        default="data/models/pretrained/groundingdino_swint_ogc.pth",
        help="Path to Grounding DINO checkpoint"
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="data/models/pretrained/mobile_sam.pt",
        help="Path to MobileSAM checkpoint"
    )

    # Detection thresholds
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.5,
        help="Box confidence threshold for DINO (default: 0.25)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.5,
        help="Text threshold for DINO (default: 0.25)"
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.5)"
    )

    # Output mode
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["boxes", "masks", "both"],
        default="boxes",
        help="Output mode: 'boxes' (fast, no SAM), 'masks' (segmentation only), 'both' (default)"
    )

    # Device
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use, -1 for CPU (default: 0)"
    )

    # Image filtering
    parser.add_argument(
        "--extensions",
        type=str,
        default="jpg,jpeg,png,bmp",
        help="Comma-separated image extensions to process (default: jpg,jpeg,png,bmp)"
    )
    
    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualized images with annotations overlaid"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: output_dir/visualizations/)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide class labels in visualization"
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Hide confidence scores in visualization"
    )
    
    # Profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed timing profiler for performance analysis"
    )
    
    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "onnx", "custom_onnx"],
        default="pytorch",
        help="Inference backend: 'pytorch' (default), 'onnx' (HuggingFace), or 'custom_onnx' (our export)"
    )
    parser.add_argument(
        "--onnx-model-dir",
        type=str,
        default="grounding-dino-tiny-ONNX",
        help="Path to HuggingFace ONNX model directory (for --backend onnx)"
    )
    parser.add_argument(
        "--onnx-variant",
        type=str,
        choices=["fp32", "fp16", "int8", "q4", "q4f16", "uint8", "quantized"],
        default="fp16",
        help="HuggingFace ONNX model variant (for --backend onnx)"
    )
    parser.add_argument(
        "--custom-onnx-path",
        type=str,
        default="data/models/groundingdino_swint.onnx",
        help="Path to custom ONNX model (for --backend custom_onnx)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[800, 800],
        metavar=('H', 'W'),
        help="Input size (H W) for custom ONNX model (default: 800 800)"
    )

    return parser.parse_args()


def collect_image_paths(images_path: str, extensions: list) -> list:
    """
    Collect image paths from directory or single file.
    
    Args:
        images_path: Path to directory or single file
        extensions: List of valid extensions
        
    Returns:
        List of image file paths
    """
    path = Path(images_path)

    if path.is_file():
        # Single file
        if path.suffix.lower().lstrip('.') in extensions:
            return [str(path)]
        logger.warning("File %s has unsupported extension", path)
        return []

    if path.is_dir():
        # Directory - collect all matching files
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(str(path / f"*.{ext}")))
            image_paths.extend(glob(str(path / f"*.{ext.upper()}")))

        # Sort for consistent ordering
        image_paths = sorted(set(image_paths))
        return image_paths

    logger.error("Path not found: %s", images_path)
    return []


def main():
    """Main entry point."""
    args = parse_args()

    print_header("Auto-Labeling Tool")
    print("Grounding DINO + MobileSAM")
    print()

    # Setup device
    if args.gpu >= 0:
        setup_cuda_device(args.gpu)
        device = "cuda"
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Parse class prompts
    class_prompts = [c.strip() for c in args.classes.split(",")]
    logger.info("Classes to detect: %s", class_prompts)

    # Collect image paths
    extensions = [e.strip().lower() for e in args.extensions.split(",")]
    image_paths = collect_image_paths(args.images, extensions)

    if not image_paths:
        logger.error("No images found in: %s", args.images)
        sys.exit(1)

    logger.info("Found %d images to process", len(image_paths))

    # Validate model checkpoints based on backend
    if args.backend == "pytorch":
        validate_file_exists(args.dino_config, "DINO config")
        validate_file_exists(args.dino_checkpoint, "DINO checkpoint")
    elif args.backend == "onnx":
        # HuggingFace ONNX backend - validate ONNX model directory
        onnx_model_path = Path(args.onnx_model_dir)
        if not onnx_model_path.exists():
            logger.error("ONNX model directory not found: %s", args.onnx_model_dir)
            sys.exit(1)
        variant_file = onnx_model_path / "onnx" / f"model_{args.onnx_variant}.onnx"
        if args.onnx_variant == "fp32":
            variant_file = onnx_model_path / "onnx" / "model.onnx"
        if not variant_file.exists():
            logger.error("ONNX model variant not found: %s", variant_file)
            sys.exit(1)
        logger.info("Using HuggingFace ONNX model: %s", variant_file)
    elif args.backend == "custom_onnx":
        # Custom ONNX backend - validate custom ONNX model
        if not Path(args.custom_onnx_path).exists():
            logger.error("Custom ONNX model not found: %s", args.custom_onnx_path)
            sys.exit(1)
        logger.info("Using custom ONNX model: %s", args.custom_onnx_path)

    # Only validate SAM checkpoint if we need masks
    if args.output_mode in ("masks", "both"):
        validate_file_exists(args.sam_checkpoint, "MobileSAM checkpoint")
    else:
        logger.info("Skipping SAM checkpoint validation (output_mode='boxes')")

    config = AutoLabelerConfig(
        backend=args.backend,
        grounding_dino_config=args.dino_config,
        grounding_dino_checkpoint=args.dino_checkpoint,
        mobile_sam_checkpoint=args.sam_checkpoint,
        onnx_model_dir=args.onnx_model_dir,
        onnx_model_variant=args.onnx_variant,
        custom_onnx_path=args.custom_onnx_path,
        custom_onnx_input_size=tuple(args.input_size),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        output_mode=args.output_mode,
        device=device,
        enable_profiling=args.profile
    )

    print_section("Configuration")
    print(f"  Backend:         {args.backend}")
    if args.backend == "pytorch":
        print(f"  DINO checkpoint: {args.dino_checkpoint}")
    elif args.backend == "onnx":
        print(f"  ONNX model:      {args.onnx_model_dir}/onnx/model_{args.onnx_variant}.onnx")
    elif args.backend == "custom_onnx":
        print(f"  Custom ONNX:     {args.custom_onnx_path}")
        print(f"  Input size:      {args.input_size[0]}x{args.input_size[1]}")
    print(f"  SAM checkpoint:  {args.sam_checkpoint}")
    print(f"  Box threshold:   {args.box_threshold}")
    print(f"  NMS threshold:   {args.nms_threshold}")
    print(f"  Output mode:     {args.output_mode}")
    print(f"  Device:          {device}")
    print(f"  Profiling:       {'enabled' if args.profile else 'disabled'}")
    print()

    # Create labeler
    labeler = AutoLabeler(config)

    # Process images
    print_section("Processing Images")
    start_time = time.time()

    # If visualization is enabled, process images individually to store results
    if args.visualize:
        results = []
        processed_paths = []
        
        for image_path in image_paths:
            logger.info(f"Processing: {image_path}")
            try:
                result = labeler.label_single_image(image_path, class_prompts)
                results.append(result)
                processed_paths.append(image_path)
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
        
        # Build COCO output from results
        from ml_engine.inference.auto_labeler import export_to_coco
        coco_output = export_to_coco(results, class_prompts, args.output_mode)
        
        # Save COCO JSON
        from core.config import save_json
        save_json(coco_output, args.output)
        logger.info(f"Saved COCO annotations to: {args.output}")
        
        # Save visualizations
        viz_dir = args.viz_dir or str(Path(args.output).parent / "visualizations")
        print_section("Saving Visualizations")
        
        show_boxes = args.output_mode in ("boxes", "both")
        show_masks = args.output_mode in ("masks", "both")
        
        viz_count = 0
        for image_path, result in zip(processed_paths, results):
            filename = Path(image_path).stem + "_viz.jpg"
            viz_path = str(Path(viz_dir) / filename)
            
            # Profile visualization if enabled
            with labeler.profiler.measure("visualization"):
                visualize_detections(
                    image_path=image_path,
                    result=result,
                    class_prompts=class_prompts,
                    output_path=viz_path,
                    show_boxes=show_boxes,
                    show_masks=show_masks,
                    show_labels=not args.no_labels,
                    show_scores=not args.no_scores
                )
            viz_count += 1
        
        logger.info(f"Saved {viz_count} visualizations to: {viz_dir}")
    else:
        # Standard processing without visualization
        coco_output = labeler.label_images(
            image_paths=image_paths,
            class_prompts=class_prompts,
            output_path=args.output
        )

    elapsed_time = time.time() - start_time

    # Print summary
    print_section("Summary")
    print(f"  Images processed:  {len(coco_output['images'])}")
    print(f"  Annotations:       {len(coco_output['annotations'])}")
    print(f"  Categories:        {len(coco_output['categories'])}")
    print(f"  Time elapsed:      {format_time(elapsed_time)}")
    print(f"  Avg time/image:    {elapsed_time / max(len(image_paths), 1):.2f}s")
    print()
    print(f"Output saved to: {args.output}")
    
    if args.visualize:
        viz_dir = args.viz_dir or str(Path(args.output).parent / "visualizations")
        print(f"Visualizations saved to: {viz_dir}")

    # Print category breakdown
    if coco_output['annotations']:
        print("\nAnnotations per category:")
        category_counts = {}
        for ann in coco_output['annotations']:
            cat_id = ann['category_id']
            cat_name = class_prompts[cat_id] if cat_id < len(class_prompts) else f"Unknown({cat_id})"
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        for cat_name, count in sorted(category_counts.items()):
            print(f"  - {cat_name}: {count}")
    
    # Print profiler summary if enabled
    if args.profile:
        labeler.profiler.print_summary()


if __name__ == "__main__":
    main()
