"""
Integration tests for SAM LoRA fine-tuning pipeline.

Tests:
1. SAMLoRA model instantiation with LoRA adapters
2. Forward pass with proper 3-stage pipeline
3. Gradient flow through LoRA adapters
4. SegmentationLoss compatibility
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import List, Tuple


class MockImageEncoder(nn.Module):
    """Mock image encoder that returns features and intermediate embeddings."""
    
    def __init__(self, embed_dim: int = 256, img_size: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        # Just a simple conv to make it differentiable (though frozen)
        self.dummy = nn.Conv2d(3, embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        # Return features [B, 256, 64, 64] and list of intermediate embeddings
        features = torch.randn(B, self.embed_dim, 64, 64, device=x.device)
        interm_embeddings = [
            torch.randn(B, 64, 64, self.embed_dim, device=x.device)
            for _ in range(4)  # 4 global attention blocks
        ]
        return features, interm_embeddings


class MockPromptEncoder(nn.Module):
    """Mock prompt encoder."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = (64, 64)
        # Point embeddings
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)
        ])
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    
    def get_dense_pe(self) -> torch.Tensor:
        return torch.randn(1, self.embed_dim, 64, 64)
    
    def forward(
        self,
        points=None,
        boxes=None,
        masks=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine batch size from inputs
        if boxes is not None:
            B = boxes.shape[0]
        elif points is not None:
            B = points[0].shape[0]
        else:
            B = 1
        
        device = boxes.device if boxes is not None else (
            points[0].device if points is not None else 'cpu'
        )
        
        # Sparse embeddings: [B, num_tokens, embed_dim]
        # For boxes, 2 tokens per box (corners)
        sparse = torch.randn(B, 2, self.embed_dim, device=device)
        
        # Dense embeddings: [B, embed_dim, 64, 64]
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B, -1, 64, 64
        ).to(device)
        
        return sparse, dense


class MockMaskDecoder(nn.Module):
    """Mock mask decoder with LoRA-compatible structure."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer layers (targets for LoRA)
        self.transformer = MockTwoWayTransformer(embed_dim)
        
        # Output heads (trainable during fine-tuning)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
        )
        
        self.iou_prediction_head = nn.Linear(embed_dim, 4)  # 4 masks
        
        # Tokens
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(4, embed_dim)
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool = False,
        interm_embeddings: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = image_embeddings.shape[0]
        device = image_embeddings.device
        
        # Simple mock: return random masks and IoU predictions
        num_masks = 3 if multimask_output else 1
        
        # Low-res masks: [B, num_masks, 256, 256]
        masks = torch.randn(B, num_masks, 256, 256, device=device, requires_grad=True)
        
        # IoU predictions: [B, num_masks]
        iou_pred = torch.sigmoid(torch.randn(B, num_masks, device=device, requires_grad=True))
        
        return masks, iou_pred


class MockTwoWayTransformer(nn.Module):
    """Mock transformer with attention layers for LoRA."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Create layers with q_proj, k_proj, v_proj, out_proj (LoRA targets)
        self.layers = nn.ModuleList([
            MockAttentionBlock(embed_dim) for _ in range(2)
        ])
    
    def forward(self, x, pos, tokens):
        for layer in self.layers:
            x, tokens = layer(x, tokens)
        return tokens, x


class MockAttentionBlock(nn.Module):
    """Mock attention block with projection layers."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Self attention
        self.self_attn = MockAttention(embed_dim)
        # Cross attention
        self.cross_attn_token_to_image = MockAttention(embed_dim)
        self.cross_attn_image_to_token = MockAttention(embed_dim)
    
    def forward(self, x, tokens):
        return x, tokens


class MockAttention(nn.Module):
    """Mock attention with projection layers (LoRA targets)."""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k=None, v=None):
        if k is None:
            k = q
        if v is None:
            v = q
        return self.out_proj(self.v_proj(v))


class MockSAM(nn.Module):
    """Mock SAM model with proper structure."""
    
    def __init__(self):
        super().__init__()
        self.image_encoder = MockImageEncoder()
        self.prompt_encoder = MockPromptEncoder()
        self.mask_decoder = MockMaskDecoder()
        
        # Standard SAM attributes
        self.mask_threshold = 0.0
        self.image_format = "RGB"
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1))


@pytest.fixture
def mock_sam():
    """Create a mock SAM model."""
    return MockSAM()


class TestSAMLoRAConfig:
    """Test LoRA configuration for SAM."""
    
    def test_lora_target_modules_format(self):
        """Test that LoRA target_modules use correct PEFT format."""
        from core.constants import DEFAULT_SAM_LORA_CONFIG
        
        lora_config = DEFAULT_SAM_LORA_CONFIG['lora']
        target_modules = lora_config['target_modules']
        
        # Should NOT contain wildcards (PEFT doesn't support them)
        for module in target_modules:
            assert '*' not in module, f"Wildcard found in target_module: {module}"
        
        # Should contain attention projection layers
        assert 'q_proj' in target_modules
        assert 'k_proj' in target_modules
        assert 'v_proj' in target_modules
        assert 'out_proj' in target_modules


class TestSAMLoRAForwardPass:
    """Test SAMLoRA forward pass implementation."""
    
    def test_forward_returns_expected_keys(self, mock_sam):
        """Test that forward returns expected output keys."""
        from ml_engine.models.teacher.sam_lora import SAMLoRA
        
        # Patch the model loading to use our mock
        with patch.object(SAMLoRA, '_load_base_model', return_value=mock_sam):
            with patch.object(SAMLoRA, '_apply_lora'):
                with patch.object(SAMLoRA, '_unfreeze_mask_decoder_heads'):
                    with patch('ml_engine.training.peft_utils.verify_freezing'):
                        model = SAMLoRA(
                            base_checkpoint='dummy.pth',
                            model_type='vit_h',
                            lora_config={'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj', 'k_proj']}
                        )
        
        # Create mock inputs
        batch_size = 2
        num_objects = 3
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024  # xyxy format
        
        # Run forward
        outputs = model.forward(images, box_prompts=box_prompts)
        
        # Check output keys
        assert 'pred_masks' in outputs
        assert 'iou_predictions' in outputs
        
        # Check shapes (single mask output by default)
        assert outputs['pred_masks'].shape == (batch_size, num_objects, 1024, 1024)
        assert outputs['iou_predictions'].shape == (batch_size, num_objects)
    
    def test_forward_with_multimask_output(self, mock_sam):
        """Test forward with multimask_output=True."""
        from ml_engine.models.teacher.sam_lora import SAMLoRA
        
        with patch.object(SAMLoRA, '_load_base_model', return_value=mock_sam):
            with patch.object(SAMLoRA, '_apply_lora'):
                with patch.object(SAMLoRA, '_unfreeze_mask_decoder_heads'):
                    with patch('ml_engine.training.peft_utils.verify_freezing'):
                        model = SAMLoRA(
                            base_checkpoint='dummy.pth',
                            model_type='vit_h',
                            lora_config={'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj']}
                        )
        
        batch_size = 2
        num_objects = 3
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024
        
        outputs = model.forward(images, box_prompts=box_prompts, multimask_output=True)
        
        # With multimask, should return 3 masks per prompt
        assert outputs['pred_masks'].shape == (batch_size, num_objects, 3, 1024, 1024)
        assert outputs['iou_predictions'].shape == (batch_size, num_objects, 3)


class TestSegmentationLossCompatibility:
    """Test that SAMLoRA outputs are compatible with SegmentationLoss."""
    
    def test_loss_with_sam_output_format(self):
        """Test SegmentationLoss with SAMLoRA output format."""
        from ml_engine.training.losses import SegmentationLoss
        
        batch_size = 2
        num_objects = 3
        H, W = 1024, 1024
        
        # Mock SAMLoRA output
        predictions = {
            'pred_masks': torch.randn(batch_size, num_objects, H, W),
            'iou_predictions': torch.rand(batch_size, num_objects)
        }
        
        # Mock targets
        targets = {
            'masks': torch.randint(0, 2, (batch_size, num_objects, H, W)).float(),
            'valid_mask': torch.ones(batch_size, num_objects, dtype=torch.bool)
        }
        
        # Create loss and compute
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(predictions, targets)
        
        # Check loss output
        assert 'loss' in loss_dict
        assert 'loss_focal' in loss_dict
        assert 'loss_dice' in loss_dict
        assert 'loss_iou' in loss_dict
        
        # Check loss is valid (not NaN)
        assert not torch.isnan(loss_dict['loss'])
        assert loss_dict['loss'] >= 0
    
    def test_loss_with_padding_mask(self):
        """Test SegmentationLoss handles padding correctly."""
        from ml_engine.training.losses import SegmentationLoss
        
        batch_size = 2
        num_objects = 3
        H, W = 256, 256  # Use smaller for speed
        
        predictions = {
            'pred_masks': torch.randn(batch_size, num_objects, H, W),
        }
        
        # Only first 2 objects are valid
        valid_mask = torch.zeros(batch_size, num_objects, dtype=torch.bool)
        valid_mask[:, :2] = True
        
        targets = {
            'masks': torch.randint(0, 2, (batch_size, num_objects, H, W)).float(),
            'valid_mask': valid_mask
        }
        
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(predictions, targets)
        
        # Should not raise error and produce valid loss
        assert not torch.isnan(loss_dict['loss'])


class TestGradientFlow:
    """Test gradient flow through LoRA adapters."""
    
    def test_gradients_flow_through_mask_decoder(self, mock_sam):
        """Test that gradients flow through mask decoder during training."""
        from ml_engine.models.teacher.sam_lora import SAMLoRA
        from ml_engine.training.losses import SegmentationLoss
        
        with patch.object(SAMLoRA, '_load_base_model', return_value=mock_sam):
            with patch.object(SAMLoRA, '_apply_lora'):
                with patch.object(SAMLoRA, '_unfreeze_mask_decoder_heads'):
                    with patch('ml_engine.training.peft_utils.verify_freezing'):
                        model = SAMLoRA(
                            base_checkpoint='dummy.pth',
                            model_type='vit_h',
                            lora_config={'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj']}
                        )
        
        batch_size = 2
        num_objects = 2
        
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024
        
        # Enable gradients for mask decoder parameters (simulating LoRA training)
        for param in model._get_mask_decoder().parameters():
            param.requires_grad = True
        
        outputs = model.forward(images, box_prompts=box_prompts)
        
        targets = {
            'masks': torch.randint(0, 2, (batch_size, num_objects, 1024, 1024)).float(),
            'valid_mask': torch.ones(batch_size, num_objects, dtype=torch.bool)
        }
        
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(outputs, targets)
        
        # Backward pass
        loss_dict['loss'].backward()
        
        # Check that mask decoder parameters have gradients
        has_grad = False
        for param in model._get_mask_decoder().parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in mask decoder parameters"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

