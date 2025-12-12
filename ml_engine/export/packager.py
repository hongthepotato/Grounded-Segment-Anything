"""
Export package creator.

Creates a downloadable ZIP package containing:
- Merged model weights
- Inference script
- README with usage instructions
- Requirements file
"""

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from torch import nn

from .merger import merge_lora_weights, save_merged_model

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_export_package(
    model: nn.Module,
    output_dir: Path,
    class_names: List[str],
    model_name: str = "grounding_dino",
    training_info: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create a downloadable ZIP package with model and inference scripts.
    
    The package includes:
    - merged_model.pth: Model weights (merged LoRA)
    - inference.py: Ready-to-run inference script
    - README.md: Usage instructions
    - requirements.txt: Python dependencies
    - class_names.txt: Classes the model was trained on
    
    Args:
        model: GroundingDINOLoRA model to export
        output_dir: Directory to save the package
        class_names: List of class names used in training
        model_name: Name prefix for the model (default: grounding_dino)
        training_info: Optional training metadata (epochs, mAP, etc.)
        
    Returns:
        Path to the created ZIP file
        
    Example:
        >>> package_path = create_export_package(
        ...     model=trained_model,
        ...     output_dir=Path("exports"),
        ...     class_names=["dog", "cat", "car"],
        ...     training_info={"epochs": 50, "mAP50": 0.85}
        ... )
    """
    output_dir = Path(output_dir)
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for package contents
    package_dir = exports_dir / f"{model_name}_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    logger.info("Creating export package in: %s", package_dir)

    # 1. Merge LoRA weights and save model
    logger.info("Step 1/5: Merging LoRA weights...")
    merged_model = merge_lora_weights(model)

    model_path = package_dir / "merged_model.pth"
    save_merged_model(
        model=merged_model,
        output_path=model_path,
        class_names=class_names,
        extra_metadata=training_info
    )

    # 2. Copy inference script
    logger.info("Step 2/5: Adding inference script...")
    inference_template = TEMPLATES_DIR / "inference_template.py"
    if inference_template.exists():
        shutil.copy(inference_template, package_dir / "inference.py")
    else:
        logger.warning("Inference template not found: %s", inference_template)
        _create_minimal_inference_script(package_dir / "inference.py")

    # 3. Create README with filled-in values
    logger.info("Step 3/5: Generating README...")
    _create_readme(
        output_path=package_dir / "README.md",
        class_names=class_names,
        training_info=training_info
    )

    # 4. Copy requirements
    logger.info("Step 4/5: Adding requirements...")
    requirements_template = TEMPLATES_DIR / "requirements.txt"
    if requirements_template.exists():
        shutil.copy(requirements_template, package_dir / "requirements.txt")
    else:
        _create_requirements(package_dir / "requirements.txt")

    # 5. Create class_names.txt
    logger.info("Step 5/5: Saving class names...")
    with open(package_dir / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    # Create ZIP archive
    zip_path = exports_dir / "model_package.zip"
    logger.info("Creating ZIP archive: %s", zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)

    # Get ZIP size
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("Export package created: %s (%.1f MB)", zip_path, zip_size_mb)

    # Clean up temporary directory (keep ZIP only)
    shutil.rmtree(package_dir)

    return zip_path


def _create_readme(
    output_path: Path,
    class_names: List[str],
    training_info: Optional[Dict[str, Any]] = None
) -> None:
    """Create README with filled-in template values."""
    readme_template = TEMPLATES_DIR / "README_template.md"

    if readme_template.exists():
        content = readme_template.read_text()
    else:
        content = _get_minimal_readme()

    # Fill in template values
    training_info = training_info or {}

    replacements = {
        "{class_names}": ", ".join(class_names),
        "{num_classes}": str(len(class_names)),
        "{training_date}": training_info.get("training_date", "N/A"),
        "{epochs}": str(training_info.get("epochs", "N/A")),
        "{map50}": f"{training_info.get('mAP50', 0):.1%}" if training_info.get('mAP50') else "N/A",
        "{generation_date}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for key, value in replacements.items():
        content = content.replace(key, value)

    output_path.write_text(content)


def _create_requirements(output_path: Path) -> None:
    """Create minimal requirements file."""
    content = """# Requirements for Grounding DINO Inference
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=9.0.0
transformers>=4.25.0
opencv-python>=4.7.0
"""
    output_path.write_text(content)


def _create_minimal_inference_script(output_path: Path) -> None:
    """Create minimal inference script if template not found."""
    content = '''#!/usr/bin/env python3
"""
Grounding DINO Inference Script
Usage: python inference.py --image photo.jpg --text "dog . cat"
"""
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="object")
    parser.add_argument("--model", default="merged_model.pth")
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.model, map_location="cpu")
    print(f"Model loaded. Classes: {checkpoint.get('class_names', [])}")
    print("For full inference, install GroundingDINO and update this script.")

if __name__ == "__main__":
    main()
'''
    output_path.write_text(content)


def _get_minimal_readme() -> str:
    """Return minimal README content."""
    return """# Fine-tuned Grounding DINO Model

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone GroundingDINO:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO && pip install -e .
   ```

3. Run inference:
   ```bash
   python inference.py --image your_image.jpg
   ```

## Model Info
- Trained classes: {class_names}
- Training date: {training_date}

Generated on {generation_date}
"""
