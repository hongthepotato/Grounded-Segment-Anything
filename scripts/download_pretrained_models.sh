#!/bin/bash
# Download pretrained models for teacher fine-tuning

set -e

echo "======================================================================"
echo "Downloading Pretrained Models"
echo "======================================================================"

# Create directory
mkdir -p data/models/pretrained
cd data/models/pretrained

# Grounding DINO
echo ""
echo "📥 Downloading Grounding DINO (11GB)..."
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    echo "✓ Grounding DINO downloaded"
else
    echo "✓ Grounding DINO already exists"
fi

# SAM ViT-H
echo ""
echo "📥 Downloading SAM ViT-H (2.4GB)..."
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    echo "✓ SAM ViT-H downloaded"
else
    echo "✓ SAM ViT-H already exists"
fi

# SAM ViT-L (optional)
echo ""
echo "📥 Downloading SAM ViT-L (1.2GB) [Optional]..."
if [ ! -f "sam_vit_l_0b3195.pth" ]; then
    read -p "Download SAM ViT-L? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        echo "✓ SAM ViT-L downloaded"
    fi
else
    echo "✓ SAM ViT-L already exists"
fi

echo ""
echo "======================================================================"
echo "✅ Download complete!"
echo "======================================================================"
echo ""
echo "Downloaded models:"
ls -lh *.pth
echo ""
echo "Next steps:"
echo "1. Prepare your COCO dataset in data/raw/"
echo "2. Run: python cli/validate_dataset.py --data data/raw/annotations.json"
echo "3. Run: python cli/train_teacher.py --data train.json --val val.json --output exp1"


