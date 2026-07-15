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
from typing import Any, Dict, List, Optional

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
    training_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create a downloadable ZIP package with merged model weights and docs.

    Args:
        model: Model with LoRA adapters (GroundingDINOLoRA or SAMHQLoRA)
        output_dir: Experiment directory
        class_names: Class names used in training
        model_name: Model identifier ("grounding_dino" or "sam")
        training_info: Optional training metadata (epochs, metrics, etc.)

    Returns:
        Path to the created ZIP file
    """
    # Validate class_names contain no embedded newlines/carriage returns.
    # class_names.txt is newline-delimited, so a class name with '\n' would
    # silently split into two entries on read-back — silent data corruption.
    # Reject at the boundary instead of corrupting downstream artifacts.
    for i, name in enumerate(class_names):
        if "\n" in name or "\r" in name:
            raise ValueError(
                f"class_names[{i}]={name!r} contains a newline or carriage "
                f"return character; class_names.txt is newline-delimited so "
                f"embedded newlines would corrupt the round-trip. Strip or "
                f"replace newlines in class names before calling."
            )

    output_dir = Path(output_dir)
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    package_dir = exports_dir / f"{model_name}_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    try:
        logger.info("Creating %s export package in: %s", model_name, package_dir)

        logger.info("Step 1/4: Merging LoRA weights...")
        merged_model = merge_lora_weights(model)

        model_path = package_dir / "merged_model.pth"
        save_merged_model(
            model=merged_model,
            output_path=model_path,
            class_names=class_names,
            training_info=training_info,
            model_name=model_name,
        )

        logger.info("Step 2/4: Adding inference script...")
        template_name = f"{model_name}_inference_template.py"
        inference_template = TEMPLATES_DIR / template_name
        if not inference_template.exists():
            inference_template = TEMPLATES_DIR / "inference_template.py"
        if inference_template.exists():
            shutil.copy(inference_template, package_dir / "inference.py")
        else:
            _create_minimal_inference_script(package_dir / "inference.py", model_name)

        logger.info("Step 3/4: Generating README...")
        _create_readme(
            output_path=package_dir / "README.md",
            class_names=class_names,
            training_info=training_info,
            model_name=model_name,
        )

        logger.info("Step 4/4: Saving class names...")
        with open(package_dir / "class_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))

        zip_path = exports_dir / f"{model_name}_package.zip"
        logger.info("Creating ZIP archive: %s", zip_path)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info("Export package created: %s (%.1f MB)", zip_path, zip_size_mb)

        return zip_path
    finally:
        try:
            shutil.rmtree(package_dir)
        except Exception:
            logger.warning("Failed to clean up temp package dir: %s", package_dir)


def _create_readme(
    output_path: Path,
    class_names: List[str],
    training_info: Optional[Dict[str, Any]] = None,
    model_name: str = "grounding_dino",
) -> None:
    """Create README with filled-in template values."""
    readme_template = TEMPLATES_DIR / f"{model_name}_README_template.md"
    if not readme_template.exists():
        readme_template = TEMPLATES_DIR / "README_template.md"

    if readme_template.exists():
        content = readme_template.read_text()
    else:
        content = _get_minimal_readme(model_name)

    training_info = training_info or {}

    replacements = {
        "{model_name}": model_name,
        "{class_names}": ", ".join(class_names),
        "{num_classes}": str(len(class_names)),
        "{training_date}": training_info.get("training_date", "N/A"),
        "{epochs}": str(training_info.get("epochs", "N/A")),
        # Use `is not None` instead of truthiness so a genuinely-zero metric
        # (catastrophic training failure: mAP50=0.0) renders as "0.0%" instead
        # of being silently misrepresented as "N/A". Same fix for mIoU.
        "{map50}": (f"{training_info['mAP50']:.1%}" if training_info.get("mAP50") is not None else "N/A"),
        "{miou}": (f"{training_info['mIoU']:.1%}" if training_info.get("mIoU") is not None else "N/A"),
        "{generation_date}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for key, value in replacements.items():
        content = content.replace(key, value)

    output_path.write_text(content)


def _create_minimal_inference_script(output_path: Path, model_name: str = "grounding_dino") -> None:
    """Create minimal inference script if template not found."""
    content = f'''#!/usr/bin/env python3
"""
{model_name} Inference Script
"""
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="merged_model.pth")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    print(f"Model loaded ({model_name}). Classes: {{checkpoint.get('class_names', [])}}")

if __name__ == "__main__":
    main()
'''
    output_path.write_text(content)


def _get_minimal_readme(model_name: str = "grounding_dino") -> str:
    """Return minimal README content."""
    return f"""# Fine-tuned {model_name} Model

## Quick Start

```bash
pip install torch torchvision
python inference.py --image your_image.jpg
```

## Model Info
- Model: {{model_name}}
- Trained classes: {{class_names}}
- Epochs: {{epochs}}

Generated on {{generation_date}}
"""
