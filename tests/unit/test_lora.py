"""
Unit tests for LoRA (Parameter-Efficient Fine-Tuning) utilities.

Tests:
- LoRA application
- Freezing verification
- Parameter counting
- Adapter saving/loading
"""

import unittest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for testing LoRA."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.self_attn = nn.ModuleDict({
            'q_proj': nn.Linear(64, 64),
            'k_proj': nn.Linear(64, 64),
            'v_proj': nn.Linear(64, 64),
            'out_proj': nn.Linear(64, 64)
        })
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)


class TestLoRAUtilities(unittest.TestCase):
    """Test LoRA utilities."""
    
    def setUp(self):
        """Create a simple model for testing."""
        self.model = SimpleModel()
        self.lora_config = {
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj']
        }
    
    def test_apply_lora(self):
        """Test applying LoRA to model."""
        from ml_engine.training.peft_utils import apply_lora
        
        original_params = sum(p.numel() for p in self.model.parameters())
        
        # Apply LoRA
        model_with_lora = apply_lora(self.model, self.lora_config)
        
        # Check that model has more parameters now (LoRA adapters added)
        lora_params = sum(p.numel() for p in model_with_lora.parameters())
        self.assertGreater(lora_params, original_params)
    
    def test_verify_freezing(self):
        """Test freezing verification."""
        from ml_engine.training.peft_utils import apply_lora, verify_freezing
        
        # Apply LoRA
        model_with_lora = apply_lora(self.model, self.lora_config)
        
        # Verify freezing
        stats = verify_freezing(model_with_lora, strict=False)
        
        # Check that most parameters are frozen
        self.assertGreater(stats['frozen_params'], 0)
        self.assertGreater(stats['trainable_params'], 0)
        self.assertLess(stats['trainable_ratio'], 50.0)  # Should be much less than 50%
    
    def test_trainable_parameters_are_lora(self):
        """Test that only LoRA parameters are trainable."""
        from ml_engine.training.peft_utils import apply_lora
        
        model_with_lora = apply_lora(self.model, self.lora_config)
        
        # Check each trainable parameter
        for name, param in model_with_lora.named_parameters():
            if param.requires_grad:
                # All trainable params should have 'lora' in name
                self.assertIn('lora', name.lower(),
                            f"Trainable parameter without 'lora' in name: {name}")
    
    def test_freeze_module(self):
        """Test module freezing."""
        from ml_engine.training.peft_utils import freeze_module
        
        # Freeze conv layer
        freeze_module(self.model.conv1)
        
        # Check all params are frozen
        for param in self.model.conv1.parameters():
            self.assertFalse(param.requires_grad)
    
    def test_count_lora_parameters(self):
        """Test LoRA parameter counting."""
        from ml_engine.training.peft_utils import apply_lora, count_lora_parameters
        
        model_with_lora = apply_lora(self.model, self.lora_config)
        lora_counts = count_lora_parameters(model_with_lora)
        
        # Should have LoRA parameters
        self.assertGreater(lora_counts['total_lora_params'], 0)
    
    def test_get_lora_rank(self):
        """Test getting LoRA rank."""
        from ml_engine.training.peft_utils import apply_lora, get_lora_rank
        
        model_with_lora = apply_lora(self.model, self.lora_config)
        rank = get_lora_rank(model_with_lora)
        
        if rank is not None:
            self.assertEqual(rank, self.lora_config['r'])


class TestPartialFreeze(unittest.TestCase):
    """Test partial freezing strategies."""
    
    def setUp(self):
        """Create model."""
        self.model = SimpleModel()
    
    def test_partial_freeze_for_lora(self):
        """Test partial freeze + LoRA strategy."""
        from ml_engine.training.peft_utils import partial_freeze_for_lora
        
        lora_config = {
            'r': 4,
            'lora_alpha': 8,
            'target_modules': ['q_proj', 'v_proj']
        }
        
        model = partial_freeze_for_lora(
            self.model,
            freeze_modules=['conv1', 'bn1'],
            lora_config=lora_config,
            lora_modules=['self_attn']
        )
        
        # Check that specified modules are frozen
        for param in self.model.conv1.parameters():
            self.assertFalse(param.requires_grad)
        
        # Should have trainable LoRA params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(trainable, 0)


if __name__ == '__main__':
    unittest.main()


