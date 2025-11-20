#!/usr/bin/env python3
"""
Test loading Grounding DINO with local BERT (offline mode).

Prerequisites:
1. Download BERT model first (on a machine with internet):
   python download_bert_model.py --output_dir data/models/pretrained/bert-base-uncased

2. Transfer the directory to your offline dev machine

3. Run this script to verify it works offline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ml_engine.models.teacher.grounding_dino_lora import load_grounding_dino_with_lora
import torch


def test_offline_bert():
    """Test loading Grounding DINO with local BERT."""
    
    print("="*60)
    print("Testing Offline BERT Setup")
    print("="*60)
    
    # Paths
    dino_checkpoint = 'data/models/pretrained/groundingdino_swint_ogc.pth'
    bert_path = 'data/models/pretrained/bert-base-uncased'
    
    # Check if BERT exists
    bert_dir = Path(bert_path)
    if not bert_dir.exists():
        print(f"\n❌ BERT model not found at: {bert_path}")
        print("\nPlease download BERT first:")
        print("  python download_bert_model.py --output_dir", bert_path)
        return False
    
    # Check BERT files
    required_files = ['config.json', 'vocab.txt', 'tokenizer.json', 'model.safetensors']
    missing_files = [f for f in required_files if not (bert_dir / f).exists()]
    
    if missing_files:
        print(f"\n❌ Missing BERT files: {missing_files}")
        print("Re-download the complete BERT model")
        return False
    
    print(f"\n✓ Found BERT model at: {bert_path}")
    for file in required_files:
        size_mb = (bert_dir / file).stat().st_size / (1024 * 1024)
        print(f"  {file:<25} {size_mb:>8.2f} MB")
    
    # Load model with local BERT
    print(f"\nLoading Grounding DINO with local BERT...")
    try:
        model = load_grounding_dino_with_lora(
            base_checkpoint=dino_checkpoint,
            lora_config={'r': 16, 'lora_alpha': 32},
            freeze_backbone=True,
            bert_model_path=bert_path  # ← This makes it offline!
        )
        print("✓ Model loaded successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_images = torch.randn(1, 3, 800, 800)
        class_names = ['dog', 'cat', 'car']
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images, class_names=class_names)
        
        print(f"✓ Forward pass successful!")
        print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
        print(f"  pred_boxes shape: {outputs['pred_boxes'].shape}")
        
        # Verify it's token-level (not class-level)
        num_queries = outputs['pred_logits'].shape[1]
        num_tokens = outputs['pred_logits'].shape[2]
        num_classes = len(class_names)
        
        print(f"\n✓ Verification:")
        print(f"  Number of classes: {num_classes}")
        print(f"  Number of tokens: {num_tokens}")
        print(f"  Shape correct: {num_tokens != num_classes} (should be True)")
        
        if num_tokens != num_classes:
            print("\n✓✓✓ SUCCESS! Grounding DINO works offline!")
            print("    Output is token-level as expected.")
        else:
            print("\n⚠ Warning: Output shape might be wrong")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_offline_bert()
    sys.exit(0 if success else 1)

