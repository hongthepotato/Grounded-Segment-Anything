"""
SAM (Segment Anything Model) with LoRA integration.

This module provides a wrapper for SAM with:
- LoRA image_encoder
- Full fine-tuning on mask_decoder
- Completely frozen prompt_encoder
- Box and point prompt support
"""

from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import torch
from torch import nn
from segment_anything import sam_hq_model_registry  # Use SAM-HQ
from ml_engine.training.peft_utils import (
    apply_lora, freeze_module, unfreeze_module, load_lora_model, verify_freezing
)

logger = logging.getLogger(__name__)


class SAMHQLoRA(nn.Module):
    """
    SAM-HQ (High Quality) with configurable training modes per component.
    
    SAM-HQ improves mask quality using intermediate ViT features.
    Each component can be independently set to: "frozen" | "lora" | "full"
    
    Default Configuration (LoRA encoder + Full decoder):
    - image_encoder: "lora" - LoRA adapters trainable (~1.5M), base weights frozen
    - prompt_encoder: "frozen" - Completely frozen (~3.8M)
    - mask_decoder: "full" - All parameters trainable (~4.1M + HQ layers)
    
    CRITICAL: target_modules in lora_config determines WHERE LoRA is applied!
    - image_encoder layers: "attn.qkv", "attn.proj"
    - mask_decoder layers: "self_attn.q_proj", "self_attn.k_proj", etc.
    
    Example:
        >>> model = SAMHQLoRA(
        >>>     base_checkpoint='sam_vit_h_4b8939.pth',
        >>>     model_type='vit_h',
        >>>     lora_config={
        >>>         'r': 16, 'lora_alpha': 32,
        >>>         'target_modules': ['attn.qkv', 'attn.proj']
        >>>     },
        >>>     image_encoder_mode='lora',
        >>>     prompt_encoder_mode='frozen',
        >>>     mask_decoder_mode='full'
        >>> )
    """

    def __init__(
        self,
        base_checkpoint: str,
        model_type: str = 'vit_h',
        lora_config: Dict = None,
        image_encoder_mode: str = 'lora',    # "frozen" | "lora" | "full"
        prompt_encoder_mode: str = 'frozen', # "frozen" | "lora" | "full"
        mask_decoder_mode: str = 'full'      # "frozen" | "lora" | "full"
    ):
        """
        Args:
            base_checkpoint: Path to pretrained SAM checkpoint
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            lora_config: LoRA configuration dictionary. MUST include target_modules
                         that match the intended modules (e.g., 'attn.qkv' for image_encoder)
            image_encoder_mode: Training mode for image encoder
            prompt_encoder_mode: Training mode for prompt encoder
            mask_decoder_mode: Training mode for mask decoder
        
        Training modes:
            - "frozen": No training, all parameters frozen
            - "lora": LoRA adapters trainable, base weights frozen
            - "full": All parameters trainable (full fine-tuning)
        
        Example configurations:
            1. LoRA encoder + Full decoder (recommended):
               - image_encoder_mode="lora", mask_decoder_mode="full"
               - target_modules=['attn.qkv', 'attn.proj']
            2. LoRA both:
               - image_encoder_mode="lora", mask_decoder_mode="lora"
               - target_modules=['attn.qkv', 'attn.proj', 'self_attn.q_proj', ...]
        """
        super().__init__()

        self.model_type = model_type
        self.lora_config = lora_config or {}
        self.component_modes = {
            'image_encoder': image_encoder_mode,
            'prompt_encoder': prompt_encoder_mode,
            'mask_decoder': mask_decoder_mode
        }

        logger.info("Loading SAM-HQ (%s) from: %s", model_type, base_checkpoint)
        self.model = self._load_base_model(base_checkpoint, model_type)

        logger.info("Configuring training modes: %s", self.component_modes)
        self._apply_training_modes()

        # Verify freezing (non-strict because we intentionally unfreeze some modules)
        verify_freezing(self.model, strict=False)

        logger.info(" SAM-HQ with LoRA initialized")

    def _load_base_model(self, checkpoint_path: str, model_type: str) -> nn.Module:
        """
        Load pretrained SAM-HQ model.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint (.pth file)
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        
        Returns:
            Loaded SAM-HQ model
            
        Raises:
            FileNotFoundError: If checkpoint file not found
            ValueError: If model_type is invalid
        """
        # Validate model type
        valid_types = ['vit_h', 'vit_l', 'vit_b']
        if model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type: {model_type}\n"
                f"Valid types: {valid_types}"
            )

        # Check checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for SAM-HQ not found: {checkpoint_path}"
            )

        # Load model
        logger.info("Loading SAM-HQ %s model...", model_type)
        logger.info("Checkpoint: %s", checkpoint_path)

        sam = sam_hq_model_registry[model_type](checkpoint=str(checkpoint_path))

        logger.info("✓ SAM-HQ model loaded successfully")
        logger.info("  - Image encoder: %s", type(sam.image_encoder).__name__)
        logger.info("  - Prompt encoder: %s", type(sam.prompt_encoder).__name__)
        logger.info("  - Mask decoder: %s (HQ version)", type(sam.mask_decoder).__name__)
        return sam

    def _apply_training_modes(self):
        """
        Apply training modes to each component.
        
        Flow:
        1. Apply LoRA first (freezes ALL base weights, adds LoRA adapters)
        2. Then apply component-specific modes (frozen/full) by unfreezing as needed
        
        CRITICAL: target_modules in lora_config determines WHERE LoRA is applied!
        - image_encoder: 'attn.qkv', 'attn.proj'
        - mask_decoder: 'self_attn.q_proj', 'self_attn.k_proj', etc.
        """
        # Step 1: Apply LoRA if any component uses it (freezes all, adds LoRA adapters)
        if 'lora' in self.component_modes.values():
            self.model = apply_lora(self.model, self.lora_config)
            logger.info("Applied LoRA to target_modules: %s",
                        self.lora_config.get('target_modules', []))

        # Step 2: Apply each component's mode (frozen/full handled by freeze/unfreeze)
        for component, mode in self.component_modes.items():
            self._apply_component_mode(component, mode)

    def _apply_component_mode(self, component_name: str, mode: str):
        """
        Apply training mode to a specific component.

        Args:
            component_name: Name of the component ('image_encoder', 'prompt_encoder', 'mask_decoder')
            mode: Training mode ('frozen', 'lora', 'full')
        """
        base_model = self._get_base_model()
        component = getattr(base_model, component_name, None)

        if component is None:
            logger.warning("Component %s not found in model", component_name)
            return

        param_count = sum(p.numel() for p in component.parameters())
        if mode == 'frozen':
            freeze_module(component)
            logger.info("  %s: frozen (%d params)", component_name, param_count)
        elif mode == 'lora':
            # LoRA already applied - count only LoRA params
            lora_count = sum(p.numel() for n, p in component.named_parameters() 
                           if 'lora_' in n and p.requires_grad)
            logger.info("  %s: LoRA (%d trainable, %d frozen)",
                       component_name, lora_count, param_count - lora_count)
        elif mode == 'full':
            unfreeze_module(component)
            logger.info("  %s: full (%d params trainable)", component_name, param_count)
        else:
            raise ValueError(f"Invalid mode '{mode}' for {component_name}. "
                           f"Must be 'frozen', 'lora', or 'full'")


    def _get_base_model(self) -> nn.Module:
        """
        Get the base SAM model, unwrapping PEFT wrapper if present.
        
        Returns:
            The underlying SAM model with image_encoder, prompt_encoder, mask_decoder
        """
        if hasattr(self.model, 'base_model'):
            # PEFT-wrapped model: self.model.base_model.model
            return self.model.base_model.model
        return self.model

    def _get_image_encoder(self) -> nn.Module:
        """Get the image encoder (frozen during LoRA training)."""
        return self._get_base_model().image_encoder

    def _get_prompt_encoder(self) -> nn.Module:
        """Get the prompt encoder (frozen during LoRA training)."""
        return self._get_base_model().prompt_encoder

    def _get_mask_decoder(self) -> nn.Module:
        """Get the mask decoder (has LoRA adapters)."""
        return self._get_base_model().mask_decoder

    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode images using the image encoder.
        
        Note: Gradient flow is controlled by the caller based on image_encoder_mode.
        If mode='frozen', caller wraps with torch.no_grad().
        If mode='lora' or 'full', gradients flow through.
        
        Args:
            images: Input images [B, 3, H, W], already preprocessed (normalized, padded)
        
        Returns:
            Tuple of:
                - Image embeddings [B, 256, 64, 64]
                - Intermediate embeddings (list of 4 tensors, SAM-HQ specific) 
                  each [B, 64, 64, 1280] for vit_h
        """
        # SAM-HQ image encoder returns (features, interm_embeddings)
        image_embeddings, interm_embeddings = self._get_image_encoder()(images)

        return image_embeddings, interm_embeddings

    def _encode_prompts(
        self,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using the prompt encoder.
        
        Args:
            boxes: Box prompts [N, 4] in xyxy format for ONE image
            points: Tuple of (coords [N, num_points, 2], labels [N, num_points])
            masks: Optional mask inputs [N, 1, H, W]
        
        Returns:
            Tuple of (sparse_embeddings, dense_embeddings)
        """
        prompt_encoder = self._get_prompt_encoder()

        # Format points for prompt encoder
        point_inputs = None
        if points is not None:
            coords, labels = points
            point_inputs = (coords, labels)

        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=point_inputs,
            boxes=boxes,
            masks=masks
        )

        return sparse_embeddings, dense_embeddings

    def forward(
        self,
        images: torch.Tensor,
        box_prompts: Optional[torch.Tensor] = None,
        point_prompts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        multimask_output: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SAM-HQ 3-stage pipeline.

        SAM-HQ architecture:
        1. Image encoding (frozen): Encode images + extract 4 intermediate features
        2. Prompt encoding (frozen): Encode box/point prompts to embeddings
        3. Mask decoding (LoRA/Full): Predict high-quality masks using intermediate features
        
        Args:
            images: Input images [B, 3, 1024, 1024] (preprocessed by SAMPreprocessor)
            box_prompts: Box prompts [B, N, 4] in [x1, y1, x2, y2] format
            point_prompts: Tuple of (points [B, N, P, 2], labels [B, N, P])
            multimask_output: Whether to output multiple masks per prompt
            return_features: Whether to return intermediate features
        
        Returns:
            Dict with:
                - pred_masks: [B, N, 256, 256] predicted masks
                - iou_predictions: [B, N] IoU quality predictions
                - features: [B, 256, 64, 64] optional image features
        
        Note:
            - Images should already be preprocessed (resized, normalized, padded)
            - Box coordinates should be in the preprocessed image space
            - For training, use multimask_output=False (single best mask)
            - SAM-HQ processes prompts individually due to interm_embeddings constraints
            - Use upscale_masks() to upscale to full resolution for inference
        """
        batch_size = images.shape[0]

        # Get base model components
        mask_decoder = self._get_mask_decoder()
        prompt_encoder = self._get_prompt_encoder()

        # === Stage 1: Image Encoding ===
        # Gradient flow depends on image_encoder_mode:
        # - "frozen": no gradients (wrapped in no_grad)
        # - "lora" or "full": gradients flow through
        encoder_mode = self.component_modes.get('image_encoder', 'frozen')
        if encoder_mode == 'frozen':
            with torch.no_grad():
                image_embeddings, interm_embeddings = self._encode_images(images)
        else:
            # "lora" or "full" - allow gradients
            image_embeddings, interm_embeddings = self._encode_images(images)

        # Get positional encoding for dense features
        image_pe = prompt_encoder.get_dense_pe()  # [1, 256, 64, 64]

        # Validate prompts
        if box_prompts is None and point_prompts is None:
            raise ValueError("Either box_prompts or point_prompts must be provided")

        # Allocate output tensors
        all_masks = []
        all_iou_predictions = []

        # === Stage 2 & 3: Process each image in batch ===
        for b in range(batch_size):
            # Get image embedding for this batch element
            curr_embedding = image_embeddings[b:b+1]  # [1, 256, 64, 64]

            # Get prompts for this image
            if box_prompts is not None:
                # boxes: [N, 4] for this image
                boxes = box_prompts[b]  # [N, 4]
            else:
                boxes = None

            if point_prompts is not None:
                # points: [N, P, 2], labels: [N, P]
                points = (point_prompts[0][b], point_prompts[1][b])
            else:
                points = None

            # === Stage 2: Prompt Encoding (frozen) ===
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self._encode_prompts(
                    boxes=boxes,
                    points=points
                )
            # sparse_embeddings: [N, num_tokens, 256]
            # dense_embeddings: [N, 256, 64, 64]

            num_prompts = sparse_embeddings.shape[0]

            # Get intermediate embeddings for this batch element (SAM-HQ specific)
            curr_interm_embeddings = [emb[b:b+1] for emb in interm_embeddings]

            # === Stage 3: Mask Decoding (with LoRA, gradients flow here) ===
            # SAM-HQ's MaskDecoderHQ doesn't handle batched prompts well with interm_embeddings
            # due to internal feature fusion operations. Process each prompt individually.
            prompt_masks = []
            prompt_ious = []

            for p in range(num_prompts):
                # Get single prompt embeddings
                sparse_single = sparse_embeddings[p:p+1]  # [1, num_tokens, 256]
                dense_single = dense_embeddings[p:p+1]    # [1, 256, 64, 64]

                low_res_mask, iou_pred = mask_decoder(
                    image_embeddings=curr_embedding,  # [1, 256, 64, 64]
                    image_pe=image_pe,                # [1, 256, 64, 64]
                    sparse_prompt_embeddings=sparse_single,
                    dense_prompt_embeddings=dense_single,
                    multimask_output=multimask_output,
                    hq_token_only=False,
                    interm_embeddings=curr_interm_embeddings,
                )
                # low_res_mask: [1, num_masks, 256, 256]
                # iou_pred: [1, num_masks]

                prompt_masks.append(low_res_mask)
                prompt_ious.append(iou_pred)

            # Stack results: [N, num_masks, 256, 256]
            low_res_masks = torch.cat(prompt_masks, dim=0)
            iou_predictions = torch.cat(prompt_ious, dim=0)

            # Select best mask if multimask (based on IoU prediction)
            if multimask_output:
                best_masks = low_res_masks  # [N, 3, 256, 256]
                best_iou = iou_predictions  # [N, 3]
            else:
                best_masks = low_res_masks[:, 0:1, :, :]  # [N, 1, 256, 256]
                best_iou = iou_predictions[:, 0:1]  # [N, 1]

            # Use upscale_masks() post-processing for inference if full resolution needed
            all_masks.append(best_masks)
            all_iou_predictions.append(best_iou)

        # For now, assume all images have same number of prompts (padded in dataloader)
        pred_masks = torch.stack(all_masks, dim=0)  # [B, N, num_masks, 256, 256]
        iou_predictions = torch.stack(all_iou_predictions, dim=0)  # [B, N, num_masks]

        # Squeeze mask dimension if single mask output
        if not multimask_output:
            pred_masks = pred_masks.squeeze(2)  # [B, N, 256, 256]
            iou_predictions = iou_predictions.squeeze(2)  # [B, N]

        # Note: pred_masks are at native 256x256 resolution
        # Use upscale_masks() for inference if full resolution needed
        outputs = {
            'pred_masks': pred_masks,
            'iou_predictions': iou_predictions
        }

        if return_features:
            outputs['features'] = image_embeddings

        return outputs

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA adapters only)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def upscale_masks(
        masks: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upscale masks from native 256x256 to target resolution.
        
        This is a post-processing utility for inference. The forward() method
        returns masks at native decoder resolution (256x256) for memory efficiency.
        Use this method to upscale to full resolution for visualization or output.
        
        Args:
            masks: Predicted masks [B, N, 256, 256] or [B, N, num_masks, 256, 256]
            target_size: Target (height, width), e.g., (1024, 1024) or original image size
        
        Returns:
            Upscaled masks at target resolution
        
        Example:
            >>> outputs = model(images, box_prompts=boxes)
            >>> pred_masks_256 = outputs['pred_masks']  # [B, N, 256, 256]
            >>> pred_masks_full = SAMHQLoRA.upscale_masks(pred_masks_256, (1024, 1024))
        """
        return torch.nn.functional.interpolate(
            masks,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

    def save_lora_adapters(self, output_dir: str):
        """Save only LoRA adapters."""
        from ml_engine.training.peft_utils import save_lora_adapters
        save_lora_adapters(self.model, output_dir)

    @classmethod
    def from_lora_checkpoint(
        cls,
        base_checkpoint: str,
        lora_adapter_path: str,
        model_type: str = 'vit_h',
        lora_config: Dict = None,
        image_encoder_mode: str = 'lora',
        prompt_encoder_mode: str = 'frozen',
        mask_decoder_mode: str = 'full',
        merge: bool = False
    ) -> 'SAMHQLoRA':
        """
        Load model with LoRA adapters from checkpoint.
        
        Args:
            base_checkpoint: Path to base pretrained checkpoint
            lora_adapter_path: Path to LoRA adapter directory
            model_type: SAM model type
            lora_config: LoRA configuration
            image_encoder_mode: "frozen" | "lora" | "full"
            prompt_encoder_mode: "frozen" | "lora" | "full"
            mask_decoder_mode: "frozen" | "lora" | "full"
            merge: Whether to merge LoRA into base weights
        
        Returns:
            Model with LoRA adapters loaded
        """
        # Create model with specified modes
        model = cls(
            base_checkpoint=base_checkpoint,
            model_type=model_type,
            lora_config=lora_config or {},
            image_encoder_mode=image_encoder_mode,
            prompt_encoder_mode=prompt_encoder_mode,
            mask_decoder_mode=mask_decoder_mode
        )

        model.model = load_lora_model(
            model.model,
            lora_adapter_path,
            merge=merge
        )

        return model


def load_sam_hq_with_lora(
    base_checkpoint: str,
    lora_adapter_path: Optional[str] = None,
    model_type: str = 'vit_h',
    lora_config: Optional[Dict] = None,
    image_encoder_mode: str = 'lora',
    prompt_encoder_mode: str = 'frozen',
    mask_decoder_mode: str = 'full',
    merge: bool = False
) -> SAMHQLoRA:
    """
    Factory function to load SAM-HQ with configurable training modes.
    
    Args:
        base_checkpoint: Path to pretrained checkpoint
        lora_adapter_path: Optional path to LoRA adapters
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        lora_config: LoRA configuration (r, lora_alpha, target_modules, etc.)
        image_encoder_mode: "frozen" | "lora" | "full"
        prompt_encoder_mode: "frozen" | "lora" | "full"
        mask_decoder_mode: "frozen" | "lora" | "full"
        merge: Whether to merge LoRA adapters into base model
    
    Returns:
        SAMHQLoRA instance
    
    Example:
        >>> # LoRA encoder + Full decoder (recommended)
        >>> model = load_sam_hq_with_lora(
        >>>     base_checkpoint='pretrained/sam_vit_h_4b8939.pth',
        >>>     model_type='vit_h',
        >>>     lora_config={'r': 16, 'lora_alpha': 32, 'target_modules': ['attn.qkv', 'attn.proj']},
        >>>     image_encoder_mode='lora',
        >>>     prompt_encoder_mode='frozen',
        >>>     mask_decoder_mode='full'
        >>> )
    """
    if lora_adapter_path:
        return SAMHQLoRA.from_lora_checkpoint(
            base_checkpoint=base_checkpoint,
            lora_adapter_path=lora_adapter_path,
            model_type=model_type,
            lora_config=lora_config or {},
            merge=merge
        )
    if lora_config is None:
        from core.constants import DEFAULT_SAM_LORA_CONFIG
        lora_config = DEFAULT_SAM_LORA_CONFIG['lora']

    return SAMHQLoRA(
        base_checkpoint=base_checkpoint,
        model_type=model_type,
        lora_config=lora_config,
        image_encoder_mode=image_encoder_mode,
        prompt_encoder_mode=prompt_encoder_mode,
        mask_decoder_mode=mask_decoder_mode
    )


class GroundedSAM(nn.Module):
    """
    Combined Grounded SAM (Grounding DINO + SAM) for teacher model.
    
    This wraps both models for convenient usage during distillation.
    Grounding DINO is open-vocabulary, so NO num_classes parameter needed.
    
    Example:
        >>> teacher = GroundedSAM(
        >>>     grounding_dino_base='pretrained/groundingdino.pth',
        >>>     grounding_dino_lora='teachers/dino_lora/',
        >>>     sam_base='pretrained/sam_vit_h.pth',
        >>>     sam_lora='teachers/sam_lora/',
        >>>     use_merged=True
        >>> )
        >>> 
        >>> # Get predictions - pass class names
        >>> dino_out = teacher.detect(images, class_names=['dog', 'cat', 'car'])
        >>> sam_out = teacher.segment(images, dino_out['pred_boxes'])
    """
    
    def __init__(
        self,
        grounding_dino_base: str,
        grounding_dino_lora: Optional[str] = None,
        sam_base: str = None,
        sam_lora: Optional[str] = None,
        use_merged: bool = False,
        bert_model_path: Optional[str] = None
    ):
        """
        Args:
            grounding_dino_base: Path to base Grounding DINO checkpoint
            grounding_dino_lora: Optional path to DINO LoRA adapters
            sam_base: Path to base SAM checkpoint
            sam_lora: Optional path to SAM LoRA adapters
            use_merged: Whether to merge LoRA for faster inference
            bert_model_path: Optional path to local BERT model (for offline use)
        """
        super().__init__()
        
        # Load Grounding DINO (open-vocabulary, no num_classes needed)
        from .grounding_dino_lora import load_grounding_dino_with_lora
        self.grounding_dino = load_grounding_dino_with_lora(
            base_checkpoint=grounding_dino_base,
            lora_adapter_path=grounding_dino_lora,
            merge=use_merged,
            bert_model_path=bert_model_path
        )
        
        # Load SAM
        self.sam = None
        if sam_base:
            self.sam = load_sam_hq_with_lora(
                base_checkpoint=sam_base,
                lora_adapter_path=sam_lora,
                merge=use_merged
            )
        
        logger.info("✓ Grounded SAM teacher initialized")
    
    def detect(
        self,
        images: torch.Tensor,
        class_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Run detection with Grounding DINO (open-vocabulary)."""
        return self.grounding_dino(images, class_names=class_names)
    
    def segment(
        self,
        images: torch.Tensor,
        box_prompts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run segmentation with SAM."""
        if self.sam is None:
            raise ValueError("SAM model not initialized")
        return self.sam(images, box_prompts)
    
    def forward(
        self,
        images: torch.Tensor,
        text_prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Run full Grounded SAM pipeline."""
        # Stage 1: Detection
        dino_outputs = self.detect(images, text_prompts)
        
        # Stage 2: Segmentation
        sam_outputs = {}
        if self.sam:
            sam_outputs = self.segment(images, dino_outputs['pred_boxes'])
        
        # Combine outputs
        return {
            **dino_outputs,
            **sam_outputs
        }
