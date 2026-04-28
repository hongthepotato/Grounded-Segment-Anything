"""
Unit tests for SAM-HQ LoRA fine-tuning pipeline.

Moved from tests/test_sam_lora.py (root-level, invisible to CI) to
tests/unit/ml_engine/ so CI catches regressions.

Failures in the original file were all stale tests (wrong expectations), not
production bugs. Corrections made:

1. test_lora_target_modules_format: SAM image encoder is a ViT with fused qkv
   projections — the correct module names are 'qkv' and 'proj', not the
   decoder-style 'q_proj'/'k_proj'/'v_proj'/'out_proj' the old test asserted.

2. test_forward_returns_expected_keys / test_forward_with_multimask_output:
   SAMHQLoRA.forward() returns masks at native decoder resolution (256x256),
   not 1024x1024. The docstring states this explicitly and upscale_masks() is
   provided for callers that need full resolution. Old tests asserted 1024x1024.

3. test_gradients_flow_through_mask_decoder: MockMaskDecoder.forward() created
   fresh random tensors with requires_grad=True that had no computational graph
   connection to self.parameters(), so gradients could never reach decoder
   params. Fixed by routing the mock output through iou_token / iou_prediction_head.

Adversarial tests added (TestSAMHQLoRAEdgeCases, TestUpscaleMasks,
TestSegmentationLossEdgeCases) after code-reading audit of sam_lora.py and
losses.py. Real bugs / limitations found:

A. forward() crashes on N=0 prompts — torch.cat([]) on an empty prompt list
   raises RuntimeError. Callers must ensure box_prompts has at least 1 object.

B. upscale_masks() only handles 4D input despite the docstring claiming
   [B, N, num_masks, H, W] support. F.interpolate(bilinear) requires exactly 4D.

C. SegmentationLoss ignores iou_predictions entirely — the quality head
   (iou_prediction_head in MaskDecoderHQ) receives no gradient from the mask
   loss. Callers that want to train the quality head need a separate regression
   loss against actual computed mask IoU.

D. Frozen image encoder must run under torch.no_grad(); LoRA encoder must not.
"""

from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


class MockImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, img_size: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.dummy = nn.Conv2d(3, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        features = torch.randn(B, self.embed_dim, 64, 64, device=x.device)
        interm_embeddings = [torch.randn(B, 64, 64, self.embed_dim, device=x.device) for _ in range(4)]
        return features, interm_embeddings


class MockPromptEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = (64, 64)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(4)])
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return torch.randn(1, self.embed_dim, 64, 64)

    def forward(self, points=None, boxes=None, masks=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes is not None:
            B = boxes.shape[0]
        elif points is not None:
            B = points[0].shape[0]
        else:
            B = 1
        device = boxes.device if boxes is not None else (points[0].device if points is not None else "cpu")
        sparse = torch.randn(B, 2, self.embed_dim, device=device)
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, 64, 64).to(device)
        return sparse, dense


class MockMaskDecoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.transformer = MockTwoWayTransformer(embed_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
        )
        self.iou_prediction_head = nn.Linear(embed_dim, 4)
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
        num_masks = 3 if multimask_output else 1

        # Route through real parameters so gradients reach self.parameters().
        # Previously this returned fresh random tensors with requires_grad=True
        # which had no graph connection to any parameter — gradients could never
        # flow back. Now iou_token.weight and iou_prediction_head.weight are in
        # the forward path.
        iou_tok = self.iou_token.weight.expand(B, -1)  # [B, 256]
        iou_pred = torch.sigmoid(self.iou_prediction_head(iou_tok))[:, :num_masks]  # [B, num_masks]
        # [B, num_masks, 256, 256]
        masks = iou_tok.mean(dim=-1).view(B, 1, 1, 1).expand(B, num_masks, 256, 256)
        return masks, iou_pred


class MockTwoWayTransformer(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([MockAttentionBlock(embed_dim) for _ in range(2)])

    def forward(self, x, pos, tokens):
        for layer in self.layers:
            x, tokens = layer(x, tokens)
        return tokens, x


class MockAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.self_attn = MockAttention(embed_dim)
        self.cross_attn_token_to_image = MockAttention(embed_dim)
        self.cross_attn_image_to_token = MockAttention(embed_dim)

    def forward(self, x, tokens):
        return x, tokens


class MockAttention(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.image_encoder = MockImageEncoder()
        self.prompt_encoder = MockPromptEncoder()
        self.mask_decoder = MockMaskDecoder()
        self.mask_threshold = 0.0
        self.image_format = "RGB"
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1))


@pytest.fixture
def mock_sam():
    return MockSAM()


class TestSAMHQLoRAConfig:
    def test_lora_target_modules_format(self):
        """LoRA target_modules must match ViT image-encoder layer names."""
        from core.constants import DEFAULT_SAM_LORA_CONFIG

        lora_config = DEFAULT_SAM_LORA_CONFIG["lora"]
        target_modules = lora_config["target_modules"]

        for module in target_modules:
            assert "*" not in module, f"Wildcard in target_module: {module}"

        # SAM image encoder is a ViT with fused qkv — module names are 'qkv'
        # and 'proj', NOT the decoder-style 'q_proj'/'k_proj'/'v_proj'.
        assert "qkv" in target_modules
        assert "proj" in target_modules


class TestSAMHQLoRAForwardPass:
    def test_forward_returns_expected_keys(self, mock_sam):
        """forward() returns pred_masks at native 256x256 decoder resolution."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "k_proj"]},
                    )

        batch_size, num_objects = 2, 3
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024

        outputs = model.forward(images, box_prompts=box_prompts)

        assert "pred_masks" in outputs
        assert "iou_predictions" in outputs
        # Native decoder resolution is 256x256; call upscale_masks() for 1024x1024.
        assert outputs["pred_masks"].shape == (batch_size, num_objects, 256, 256)
        assert outputs["iou_predictions"].shape == (batch_size, num_objects)

    def test_forward_with_multimask_output(self, mock_sam):
        """multimask_output=True yields 3 masks per prompt at 256x256."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        batch_size, num_objects = 2, 3
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024

        outputs = model.forward(images, box_prompts=box_prompts, multimask_output=True)

        assert outputs["pred_masks"].shape == (batch_size, num_objects, 3, 256, 256)
        assert outputs["iou_predictions"].shape == (batch_size, num_objects, 3)


class TestSegmentationLossCompatibility:
    def test_loss_with_sam_output_format(self):
        """SegmentationLoss accepts 256x256 pred_masks (native decoder resolution)."""
        from ml_engine.training.losses import SegmentationLoss

        batch_size, num_objects, H, W = 2, 3, 256, 256
        predictions = {
            "pred_masks": torch.randn(batch_size, num_objects, H, W),
            "iou_predictions": torch.rand(batch_size, num_objects),
        }
        targets = {
            "masks": torch.randint(0, 2, (batch_size, num_objects, H, W)).float(),
            "valid_mask": torch.ones(batch_size, num_objects, dtype=torch.bool),
        }
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(predictions, targets)

        assert "loss" in loss_dict
        assert "loss_focal" in loss_dict
        assert "loss_dice" in loss_dict
        assert "loss_iou" in loss_dict
        assert "loss_iou_quality" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"] >= 0

    def test_loss_with_padding_mask(self):
        """SegmentationLoss ignores padded (invalid) objects correctly."""
        from ml_engine.training.losses import SegmentationLoss

        batch_size, num_objects, H, W = 2, 3, 256, 256
        predictions = {"pred_masks": torch.randn(batch_size, num_objects, H, W)}
        valid_mask = torch.zeros(batch_size, num_objects, dtype=torch.bool)
        valid_mask[:, :2] = True
        targets = {
            "masks": torch.randint(0, 2, (batch_size, num_objects, H, W)).float(),
            "valid_mask": valid_mask,
        }
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(predictions, targets)
        assert not torch.isnan(loss_dict["loss"])


class TestSAMHQLoRAEdgeCases:
    """Edge-case contract tests — run first, then read failures."""

    def test_forward_minimal_batch_b1_n1(self, mock_sam):
        """Single image, single prompt: shapes must be [1,1,256,256] and [1,1]."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        outputs = model.forward(
            torch.randn(1, 3, 1024, 1024),
            box_prompts=torch.rand(1, 1, 4) * 1024,
        )
        assert outputs["pred_masks"].shape == (1, 1, 256, 256)
        assert outputs["iou_predictions"].shape == (1, 1)

    def test_forward_no_prompts_raises(self, mock_sam):
        """forward() with no box or point prompts must raise ValueError."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        with pytest.raises(ValueError):
            model.forward(torch.randn(2, 3, 1024, 1024))

    def test_forward_return_features_shape(self, mock_sam):
        """return_features=True must add 'features' key with shape [B, 256, 64, 64]."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        outputs = model.forward(
            torch.randn(2, 3, 1024, 1024),
            box_prompts=torch.rand(2, 3, 4) * 1024,
            return_features=True,
        )
        assert "features" in outputs
        assert outputs["features"].shape == (2, 256, 64, 64)

    def test_frozen_encoder_blocks_grad(self, mock_sam):
        """image_encoder_mode='frozen' must run the encoder inside torch.no_grad()."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                        image_encoder_mode="frozen",
                    )

        grad_enabled_during_encode: List[bool] = []
        original_forward = mock_sam.image_encoder.forward

        def spy(x):
            grad_enabled_during_encode.append(torch.is_grad_enabled())
            return original_forward(x)

        mock_sam.image_encoder.forward = spy
        model.forward(torch.randn(1, 3, 1024, 1024), box_prompts=torch.rand(1, 1, 4) * 1024)

        assert grad_enabled_during_encode, "image_encoder.forward was never called"
        assert not any(grad_enabled_during_encode), (
            "frozen encoder must run inside torch.no_grad() — grad was enabled"
        )

    def test_lora_encoder_allows_grad(self, mock_sam):
        """image_encoder_mode='lora' must NOT wrap the encoder in no_grad."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                        image_encoder_mode="lora",
                    )

        grad_enabled_during_encode: List[bool] = []
        original_forward = mock_sam.image_encoder.forward

        def spy(x):
            grad_enabled_during_encode.append(torch.is_grad_enabled())
            return original_forward(x)

        mock_sam.image_encoder.forward = spy
        model.forward(torch.randn(1, 3, 1024, 1024), box_prompts=torch.rand(1, 1, 4) * 1024)

        assert grad_enabled_during_encode, "image_encoder.forward was never called"
        assert all(grad_enabled_during_encode), (
            "lora encoder must run with grad enabled — no_grad was unexpectedly set"
        )

    def test_forward_empty_box_list(self, mock_sam):
        """box_prompts with N=0 objects: forward() should raise, not silently return garbage."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        # N=0 prompts per image
        with pytest.raises((RuntimeError, ValueError)):
            model.forward(
                torch.randn(2, 3, 1024, 1024),
                box_prompts=torch.zeros(2, 0, 4),
            )


class TestUpscaleMasks:
    """Contract tests for the upscale_masks() static utility."""

    def test_4d_single_mask_upscale(self):
        """[B, N, H, W] upscales to the target resolution."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        masks = torch.randn(2, 3, 256, 256)
        upscaled = SAMHQLoRA.upscale_masks(masks, (1024, 1024))
        assert upscaled.shape == (2, 3, 1024, 1024)

    def test_non_square_target(self):
        """Upscaling to non-square targets should preserve the requested H, W."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        masks = torch.randn(1, 2, 256, 256)
        upscaled = SAMHQLoRA.upscale_masks(masks, (512, 768))
        assert upscaled.shape == (1, 2, 512, 768)

    def test_5d_multimask_input(self):
        """upscale_masks() should handle [B, N, K, H, W] multimask tensors.

        The docstring claims to support [B, N, num_masks, H, W]. If it raises
        here the docstring is wrong and callers must flatten to 4D first.
        """
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        masks_5d = torch.randn(2, 3, 3, 256, 256)  # [B, N, K=3, H, W]
        upscaled = SAMHQLoRA.upscale_masks(masks_5d, (1024, 1024))
        assert upscaled.shape == (2, 3, 3, 1024, 1024)

    def test_upscale_output_bounded_by_input_range(self):
        """Bilinear interpolation is a weighted average — output must stay in [min, max] of input."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA

        torch.manual_seed(0)
        masks = torch.randn(2, 3, 256, 256)
        upscaled = SAMHQLoRA.upscale_masks(masks, (1024, 1024))
        assert upscaled.min() >= masks.min() - 1e-4
        assert upscaled.max() <= masks.max() + 1e-4


class TestSegmentationLossEdgeCases:
    """Numerical correctness, gradient flow, and edge inputs."""

    def test_all_invalid_valid_mask_gives_zero_loss(self):
        """All objects masked as invalid → loss must be 0.0, not NaN."""
        from ml_engine.training.losses import SegmentationLoss

        B, N, H, W = 2, 3, 256, 256
        predictions = {"pred_masks": torch.randn(B, N, H, W)}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": torch.zeros(B, N, dtype=torch.bool),
        }
        loss_dict = SegmentationLoss()(predictions, targets)
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() == pytest.approx(0.0)

    def test_iou_predictions_receives_gradient(self):
        """iou_predictions quality scores must receive gradient through SegmentationLoss.

        SegmentationLoss adds an MSE regression loss that trains the quality head
        to predict the actual mask IoU against GT. iou_predictions is consumed by
        the evaluator as `scores` for mAP ranking — uncalibrated scores degrade
        reported mAP even when mask pixels are correct.
        """
        from ml_engine.training.losses import SegmentationLoss

        B, N, H, W = 2, 3, 256, 256
        pred_masks = torch.randn(B, N, H, W, requires_grad=True)
        iou_preds = torch.rand(B, N, requires_grad=True)

        predictions = {"pred_masks": pred_masks, "iou_predictions": iou_preds}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": torch.ones(B, N, dtype=torch.bool),
        }

        loss_dict = SegmentationLoss()(predictions, targets)
        assert "loss_iou_quality" in loss_dict
        loss_dict["loss"].backward()

        assert iou_preds.grad is not None and iou_preds.grad.abs().sum() > 0

    def test_spatial_mismatch_raises(self):
        """Spatial size mismatch between pred_masks and target masks must raise."""
        from ml_engine.training.losses import SegmentationLoss

        B, N = 2, 3
        predictions = {"pred_masks": torch.randn(B, N, 256, 256)}
        targets = {
            "masks": torch.randint(0, 2, (B, N, 512, 512)).float(),
            "valid_mask": torch.ones(B, N, dtype=torch.bool),
        }
        with pytest.raises((RuntimeError, ValueError)):
            SegmentationLoss()(predictions, targets)

    def test_focal_zero_weight_removes_focal_from_total(self):
        """focal weight=0 → total loss equals dice + iou."""
        from ml_engine.training.losses import SegmentationLoss

        torch.manual_seed(42)
        B, N, H, W = 2, 3, 256, 256
        predictions = {"pred_masks": torch.randn(B, N, H, W)}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": torch.ones(B, N, dtype=torch.bool),
        }

        d = SegmentationLoss(loss_weights={"focal": 0.0, "dice": 1.0, "iou": 1.0})(predictions, targets)
        expected = (d["loss_dice"] + d["loss_iou"]).item()
        assert d["loss"].item() == pytest.approx(expected, rel=1e-5)

    def test_loss_non_negative_on_random_inputs(self):
        """All loss components must be >= 0 for arbitrary random predictions."""
        from ml_engine.training.losses import SegmentationLoss

        torch.manual_seed(7)
        B, N, H, W = 3, 4, 256, 256
        predictions = {"pred_masks": torch.randn(B, N, H, W) * 5}
        targets = {
            "masks": torch.randint(0, 2, (B, N, H, W)).float(),
            "valid_mask": torch.ones(B, N, dtype=torch.bool),
        }
        d = SegmentationLoss()(predictions, targets)
        for key in ("loss", "loss_focal", "loss_dice", "loss_iou"):
            assert d[key].item() >= 0, f"{key} was negative: {d[key].item()}"
            assert not torch.isnan(d[key]), f"{key} was NaN"


class TestGradientFlow:
    def test_gradients_flow_through_mask_decoder(self, mock_sam):
        """Backward through the loss reaches mask decoder parameters."""
        from ml_engine.models.teacher.sam_lora import SAMHQLoRA
        from ml_engine.training.losses import SegmentationLoss

        with patch.object(SAMHQLoRA, "_load_base_model", return_value=mock_sam):
            with patch.object(SAMHQLoRA, "_apply_training_modes"):
                with patch("ml_engine.training.peft_utils.verify_freezing"):
                    model = SAMHQLoRA(
                        base_checkpoint="dummy.pth",
                        model_type="vit_h",
                        lora_config={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
                    )

        batch_size, num_objects = 2, 2
        images = torch.randn(batch_size, 3, 1024, 1024)
        box_prompts = torch.rand(batch_size, num_objects, 4) * 1024

        for param in model._get_mask_decoder().parameters():
            param.requires_grad = True

        outputs = model.forward(images, box_prompts=box_prompts)

        # Targets match native 256x256 decoder output resolution.
        targets = {
            "masks": torch.randint(0, 2, (batch_size, num_objects, 256, 256)).float(),
            "valid_mask": torch.ones(batch_size, num_objects, dtype=torch.bool),
        }
        loss_fn = SegmentationLoss()
        loss_dict = loss_fn(outputs, targets)
        loss_dict["loss"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model._get_mask_decoder().parameters()
        )
        assert has_grad, "No gradients reached mask decoder parameters"
