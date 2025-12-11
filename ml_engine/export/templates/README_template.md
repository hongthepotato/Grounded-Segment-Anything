# Your Fine-tuned Grounding DINO Model

This package contains your custom-trained Grounding DINO object detection model.

## Package Contents

| File | Description |
|------|-------------|
| `merged_model.pth` | Your fine-tuned model weights |
| `inference.py` | Ready-to-run inference script |
| `requirements.txt` | Python dependencies |
| `class_names.txt` | Classes your model was trained on |
| `README.md` | This file |

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Clone GroundingDINO Repository

The model requires the GroundingDINO codebase:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Download BERT Model (Optional - for offline use)

```bash
# If you need offline inference, download BERT model:
mkdir bert-base-uncased
# Download from: https://huggingface.co/bert-base-uncased
# Place config.json, vocab.txt, and model files in bert-base-uncased/
```

### 4. Run Inference

```bash
# Basic usage (uses trained classes: {class_names})
python inference.py --image your_photo.jpg

# With custom text prompt
python inference.py --image photo.jpg --text "dog . cat . person"

# Adjust confidence threshold
python inference.py --image photo.jpg --threshold 0.5

# Process entire directory
python inference.py --image_dir ./my_images --output_dir ./results
```

## Model Information

| Property | Value |
|----------|-------|
| **Model Type** | Grounding DINO (SwinT backbone) |
| **Training Date** | {training_date} |
| **Trained Classes** | {class_names} |
| **Number of Classes** | {num_classes} |
| **Training Epochs** | {epochs} |
| **mAP@50** | {map50} |

## Usage Examples

### Python API

```python
import torch
from PIL import Image

# Load model
checkpoint = torch.load("merged_model.pth", map_location="cpu")
class_names = checkpoint['class_names']
print(f"Trained on: {class_names}")

# For full inference, see inference.py
```

### Command Line

```bash
# Single image
python inference.py --image dog.jpg

# Multiple images
python inference.py --image_dir ./photos --output_dir ./detections

# Custom classes (not limited to trained classes)
python inference.py --image scene.jpg --text "chair . table . lamp"
```

## Tips for Best Results

1. **Use trained classes**: The model works best on the classes it was trained on: `{class_names}`

2. **Adjust threshold**: If you get too many false positives, increase `--threshold` (default: 0.3)

3. **Text prompt format**: Separate classes with ` . ` (space-dot-space):
   - Good: `"dog . cat . bird"`
   - Bad: `"dog, cat, bird"` or `"dog cat bird"`

4. **GPU recommended**: The model runs much faster on GPU. Use `--device cpu` for CPU-only inference.

## Troubleshooting

### "ModuleNotFoundError: No module named 'groundingdino'"
- Make sure you cloned and installed GroundingDINO (see step 2)

### "CUDA out of memory"
- Try reducing image size or use `--device cpu`

### Low detection accuracy
- Check if you're using the right text prompt format
- Try lowering `--threshold` to 0.2

## Support

This model was trained using the Grounded-Segment-Anything platform.
For issues with inference, check:
- https://github.com/IDEA-Research/GroundingDINO

---
*Generated on {generation_date}*


