#!/usr/bin/env python3
"""
Grounding DINO Inference Script

This script runs object detection using your fine-tuned Grounding DINO model.

Usage:
    # Basic usage (uses default classes from training)
    python inference.py --image photo.jpg
    
    # Custom text prompt
    python inference.py --image photo.jpg --text "dog . cat . person"
    
    # Adjust confidence threshold
    python inference.py --image photo.jpg --threshold 0.5
    
    # Process multiple images
    python inference.py --image_dir ./images --output_dir ./results

Requirements:
    pip install -r requirements.txt

For more information, see README.md
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Add GroundingDINO to path if needed
SCRIPT_DIR = Path(__file__).parent
GROUNDINGDINO_PATH = SCRIPT_DIR / "GroundingDINO"
if GROUNDINGDINO_PATH.exists():
    sys.path.insert(0, str(GROUNDINGDINO_PATH))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the fine-tuned Grounding DINO model."""
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get class names from checkpoint
    class_names = checkpoint.get('class_names', [])

    # Build model from config
    config_path = SCRIPT_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please ensure GroundingDINO repository is in the same directory."
        )

    args = SLConfig.fromfile(str(config_path))
    args.aux_loss = False  # Not needed for inference

    # Check for local BERT model
    bert_path = SCRIPT_DIR / "bert-base-uncased"
    if bert_path.exists():
        args.bert_base_uncased_path = str(bert_path)

    model = build_model(args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Trained classes: {', '.join(class_names)}")

    return model, class_names


def load_image(image_path: str):
    """Load and preprocess image for Grounding DINO."""
    from groundingdino.util.inference import load_image as gdino_load_image

    image_source, image_tensor = gdino_load_image(image_path)
    return image_source, image_tensor


def predict(
    model,
    image_tensor: torch.Tensor,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = "cuda"
):
    """Run inference and get predictions."""
    from groundingdino.util.inference import predict as gdino_predict

    boxes, logits, phrases = gdino_predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    return boxes, logits, phrases


def visualize_results(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: list,
    output_path: str = None
):
    """Visualize detection results on image."""
    from groundingdino.util.inference import annotate

    annotated = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    # Convert BGR to RGB
    annotated_rgb = annotated[:, :, ::-1]
    result_image = Image.fromarray(annotated_rgb)

    if output_path:
        result_image.save(output_path)
        print(f"Result saved to: {output_path}")

    return result_image


def main():
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO inference on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to input image"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results (default: ./outputs)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt (e.g., 'dog . cat . person'). If not provided, uses trained classes."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="merged_model.pth",
        help="Path to model checkpoint (default: merged_model.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)"
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip visualization, only print results"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be provided")

    # Load model
    model_path = SCRIPT_DIR / args.model
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    print(f"Using device: {args.device}")

    model, class_names = load_model(str(model_path), args.device)

    # Build text prompt
    if args.text:
        text_prompt = args.text
    else:
        # Use trained classes
        text_prompt = " . ".join(class_names)
        if not text_prompt.endswith("."):
            text_prompt += "."

    print(f"Text prompt: {text_prompt}")

    # Get list of images to process
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.image_dir:
        image_dir = Path(args.image_dir)
        images.extend(image_dir.glob("*.jpg"))
        images.extend(image_dir.glob("*.jpeg"))
        images.extend(image_dir.glob("*.png"))

    if not images:
        print("No images found to process")
        sys.exit(1)

    print(f"Processing {len(images)} image(s)...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in images:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")

        try:
            # Load image
            image_source, image_tensor = load_image(str(image_path))

            # Run inference
            boxes, logits, phrases = predict(
                model=model,
                image_tensor=image_tensor,
                text_prompt=text_prompt,
                box_threshold=args.threshold,
                device=args.device
            )

            # Print results
            print(f"Found {len(boxes)} detections:")
            for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
                print(f"  [{i+1}] {phrase}: {score:.3f} at {box.tolist()}")

            # Visualize
            if not args.no_visualize and len(boxes) > 0:
                output_path = output_dir / f"{image_path.stem}_result.jpg"
                visualize_results(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases,
                    output_path=str(output_path)
                )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
