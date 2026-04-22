"""
Unit tests for LoRA (Parameter-Efficient Fine-Tuning) utilities.

Tests:
- LoRA application
- Freezing verification (both strict and non-strict paths)
- Adapter saving and error handling
- Module freeze/unfreeze
"""

import os
import unittest
import tempfile
import torch
import torch.nn as nn


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
        """Test freezing verification in non-strict mode."""
        from ml_engine.training.peft_utils import apply_lora, verify_freezing

        # Apply LoRA
        model_with_lora = apply_lora(self.model, self.lora_config)

        # Verify freezing
        stats = verify_freezing(model_with_lora, strict=False)

        # Check that most parameters are frozen
        self.assertGreater(stats['frozen_params'], 0)
        self.assertGreater(stats['trainable_params'], 0)
        self.assertLess(stats['trainable_ratio'], 50.0)  # Should be much less than 50%

    def test_verify_freezing_strict_raises_on_non_lora_trainable(self):
        """Test that strict mode raises when non-LoRA params are trainable."""
        from ml_engine.training.peft_utils import apply_lora, verify_freezing

        model_with_lora = apply_lora(self.model, self.lora_config)

        # Manually make a non-LoRA parameter trainable to trigger strict error
        for name, param in model_with_lora.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = True
                break

        with self.assertRaises(AssertionError) as ctx:
            verify_freezing(model_with_lora, strict=True)

        self.assertIn("Non-LoRA param is trainable", str(ctx.exception))

    def test_trainable_parameters_are_lora(self):
        """Test that only LoRA parameters are trainable after apply_lora."""
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

    def test_unfreeze_module(self):
        """Test module unfreezing."""
        from ml_engine.training.peft_utils import freeze_module, unfreeze_module

        # First freeze
        freeze_module(self.model.conv1)
        for param in self.model.conv1.parameters():
            self.assertFalse(param.requires_grad)

        # Then unfreeze
        unfreeze_module(self.model.conv1)
        for param in self.model.conv1.parameters():
            self.assertTrue(param.requires_grad)

    def test_save_lora_adapters_happy_path(self):
        """Test that LoRA adapters are written to disk."""
        from ml_engine.training.peft_utils import apply_lora, save_lora_adapters

        model_with_lora = apply_lora(self.model, self.lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_lora_adapters(model_with_lora, tmpdir)
            saved_files = os.listdir(tmpdir)
            self.assertTrue(
                len(saved_files) > 0,
                "save_lora_adapters wrote no files to output_dir"
            )
            # PEFT always writes adapter_config.json
            self.assertIn('adapter_config.json', saved_files)

    def test_save_lora_adapters_non_peft_model_raises(self):
        """Test that saving a non-PEFT model raises ValueError."""
        from ml_engine.training.peft_utils import save_lora_adapters

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                save_lora_adapters(self.model, tmpdir)

            self.assertIn("save_pretrained", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
