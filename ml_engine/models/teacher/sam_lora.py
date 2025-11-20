"""
SAM (Segment Anything Model) with LoRA integration.

This module provides a wrapper for SAM with:
- LoRA parameter-efficient fine-tuning
- Partial freezing strategy (freeze encoder, train decoder with LoRA)
- Box and point prompt support
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Union
from pathlib import Path
import logging
from ml_engine.training.peft_utils import load_lora_model

logger = logging.getLogger(__name__)


class SAMLoRA(nn.Module):
    """
    SAM with LoRA fine-tuning.
    
    Freezing Strategy:
    1. PEFT automatically freezes ALL parameters when applying LoRA
    2. PEFT unfreezes ONLY LoRA adapters in target_modules (mask decoder transformer layers)
    3. We manually UNFREEZE mask decoder prediction heads (IoU head, output layers)
    4. Image encoder and prompt encoder remain frozen by default
    
    Final Architecture:
    - Image encoder (ViT): ❄️  FROZEN (default, ~308M params)
    - Prompt encoder: ❄️  FROZEN (default, ~3.8M params)
    - Mask decoder transformer: ✅ LoRA adapters TRAINABLE (~0.3M params)
    - Mask decoder heads (IoU, output): ✅ TRAINABLE (~0.1M params)
    
    This reduces trainable parameters from 316M to ~0.4M (0.13%) while keeping
    prediction heads trainable for better domain adaptation.
    
    Example:
        >>> model = SAMLoRA(
        >>>     base_checkpoint='data/models/pretrained/sam_vit_h_4b8939.pth',
        >>>     model_type='vit_h',
        >>>     lora_config={'r': 8, 'lora_alpha': 16},
        >>>     freeze_image_encoder=True  # Keep encoder frozen (efficient LoRA training)
        >>> )
        >>> 
        >>> # Forward pass
        >>> outputs = model(
        >>>     images=images,
        >>>     box_prompts=boxes
        >>> )
    """
    
    def __init__(
        self,
        base_checkpoint: str,
        model_type: str = 'vit_h',
        lora_config: Dict = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True
    ):
        """
        Args:
            base_checkpoint: Path to pretrained SAM checkpoint
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            lora_config: LoRA configuration dictionary
            freeze_image_encoder: Whether to freeze image encoder (default=True)
            freeze_prompt_encoder: Whether to freeze prompt encoder (default=True)
        
        Note:
            - The mask decoder prediction heads are ALWAYS trainable
            - LoRA adapters in the mask decoder transformer are also trainable
            - Image/prompt encoders remain frozen by default for efficient training
        """
        super().__init__()
        
        self.model_type = model_type
        self.lora_config = lora_config or {}
        
        # Load base SAM model
        logger.info(f"Loading SAM ({model_type}) from: {base_checkpoint}")
        self.model = self._load_base_model(base_checkpoint, model_type)
        
        # Apply LoRA (PEFT automatically freezes ALL parameters)
        logger.info("Applying LoRA to SAM mask decoder...")
        self._apply_lora()
        
        # Unfreeze mask decoder prediction heads (required for better adaptation)
        # PEFT froze everything, but we need these trainable
        self._unfreeze_mask_decoder_heads()
        
        # Optionally unfreeze encoders (if not using LoRA-only training)
        if not freeze_image_encoder:
            self._unfreeze_image_encoder()
        if not freeze_prompt_encoder:
            self._unfreeze_prompt_encoder()
        
        # Verify freezing
        from ml_engine.training.peft_utils import verify_freezing
        verify_freezing(self.model, strict=False)
        
        logger.info("✓ SAM with LoRA initialized")
    
    def _load_base_model(self, checkpoint_path: str, model_type: str) -> nn.Module:
        """
        Load pretrained SAM model (PRODUCTION).
        
        This method loads the official Segment Anything Model (SAM).
        Requires segment_anything library to be installed.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint (.pth file)
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        
        Returns:
            Loaded SAM model
            
        Raises:
            ImportError: If segment_anything library is not installed
            FileNotFoundError: If checkpoint file not found
            ValueError: If model_type is invalid
            RuntimeError: If model loading fails
        """
        import sys
        
        # Add segment_anything to path
        sam_path = Path(__file__).parent.parent.parent.parent / 'segment_anything'
        if not sam_path.exists():
            raise FileNotFoundError(
                f"segment_anything library not found at {sam_path}\n"
                f"Please clone it: git clone https://github.com/facebookresearch/segment-anything.git segment_anything\n"
                f"Then install: cd segment_anything && pip install -e ."
            )
        
        sys.path.insert(0, str(sam_path))
        
        # Import SAM utilities
        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(
                f"Failed to import segment_anything library: {e}\n"
                f"Make sure segment_anything is properly installed:\n"
                f"  cd {sam_path}\n"
                f"  pip install -e ."
            ) from e
        
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
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download pretrained SAM weights from:\n"
                f"  https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                f"Available models:\n"
                f"  - ViT-H: sam_vit_h_4b8939.pth (2.4GB)\n"
                f"  - ViT-L: sam_vit_l_0b3195.pth (1.2GB)\n"
                f"  - ViT-B: sam_vit_b_01ec64.pth (375MB)"
            )
        
        # Load model
        logger.info("Loading SAM %s model...", model_type)
        logger.info("Checkpoint: %s", checkpoint_path)
        
        try:
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM model: {e}\n"
                f"Model type: {model_type}\n"
                f"Checkpoint: {checkpoint_path}"
            ) from e
        
        # Verify model structure
        required_attrs = ['image_encoder', 'prompt_encoder', 'mask_decoder']
        for attr in required_attrs:
            if not hasattr(sam, attr):
                raise RuntimeError(
                    f"Loaded SAM model is missing required attribute: {attr}\n"
                    f"The checkpoint may be corrupted or incompatible."
                )
        
        logger.info("✓ SAM model loaded successfully")
        logger.info("  - Image encoder: %s", type(sam.image_encoder).__name__)
        logger.info("  - Prompt encoder: %s", type(sam.prompt_encoder).__name__)
        logger.info("  - Mask decoder: %s", type(sam.mask_decoder).__name__)
        
        return sam
    
    def _apply_lora(self):
        """
        Apply LoRA adapters to mask decoder transformer layers.
        
        Note: partial_freeze_for_lora() calls PEFT's get_peft_model() which:
        1. Freezes ALL parameters
        2. Unfreezes ONLY LoRA adapters in target_modules
        """
        from ml_engine.training.peft_utils import partial_freeze_for_lora
        
        # Apply LoRA only to mask decoder
        # Note: This will freeze everything except LoRA adapters
        self.model = partial_freeze_for_lora(
            self.model,
            freeze_modules=['image_encoder', 'prompt_encoder'],
            lora_config=self.lora_config,
            lora_modules=['mask_decoder']
        )
    
    def _unfreeze_mask_decoder_heads(self):
        """
        Unfreeze mask decoder prediction heads for better domain adaptation.
        
        PEFT freezes ALL parameters by default, but mask decoder prediction heads
        (IoU prediction, output upscaling) should be trainable for better adaptation
        to custom datasets and domains.
        """
        # Access the base model (PEFT wraps it)
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        
        trainable_count = 0
        
        # Unfreeze mask decoder components
        if hasattr(base_model, 'mask_decoder'):
            mask_decoder = base_model.mask_decoder
            
            # Unfreeze IoU prediction head
            if hasattr(mask_decoder, 'iou_prediction_head'):
                for param in mask_decoder.iou_prediction_head.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
                logger.info("🔓 Unfrozen IoU prediction head")
            
            # Unfreeze output upscaling layers
            if hasattr(mask_decoder, 'output_upscaling'):
                for param in mask_decoder.output_upscaling.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
                logger.info("🔓 Unfrozen output upscaling layers")
            
            # Unfreeze mask tokens (if present)
            if hasattr(mask_decoder, 'mask_tokens'):
                mask_decoder.mask_tokens.requires_grad = True
                trainable_count += mask_decoder.mask_tokens.numel()
                logger.info("🔓 Unfrozen mask tokens")
            
            # Unfreeze IoU token (if present)
            if hasattr(mask_decoder, 'iou_token'):
                mask_decoder.iou_token.requires_grad = True
                trainable_count += mask_decoder.iou_token.numel()
                logger.info("🔓 Unfrozen IoU token")
        
        if trainable_count == 0:
            logger.warning("⚠️  No mask decoder prediction heads found to unfreeze!")
        else:
            logger.info(f"✓ Unfroze {trainable_count:,} mask decoder head parameters")
    
    def _unfreeze_image_encoder(self):
        """Unfreeze image encoder (if not using LoRA-only training)."""
        # Access the base model (PEFT wraps it)
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        
        if hasattr(base_model, 'image_encoder'):
            for param in base_model.image_encoder.parameters():
                param.requires_grad = True
            logger.info("🔓 Unfrozen image encoder (full fine-tuning mode)")
        else:
            logger.warning("⚠️  Image encoder not found in model structure")
    
    def _unfreeze_prompt_encoder(self):
        """Unfreeze prompt encoder (if not using LoRA-only training)."""
        # Access the base model (PEFT wraps it)
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        
        if hasattr(base_model, 'prompt_encoder'):
            for param in base_model.prompt_encoder.parameters():
                param.requires_grad = True
            logger.info("🔓 Unfrozen prompt encoder (full fine-tuning mode)")
        else:
            logger.warning("⚠️  Prompt encoder not found in model structure")
    
    def forward(
        self,
        images: torch.Tensor,
        box_prompts: Optional[torch.Tensor] = None,
        point_prompts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        multimask_output: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, 3, H, W]
            box_prompts: Box prompts [B, N, 4] in [x1, y1, x2, y2] format
            point_prompts: Tuple of (points [B, N, 2], labels [B, N])
            multimask_output: Whether to output multiple masks
            return_features: Whether to return intermediate features
        
        Returns:
            Dict with:
                - pred_masks: [B, N, H, W] or [B, 3, N, H, W] if multimask
                - iou_predictions: [B, N] or [B, 3, N] if multimask
                - features: Optional intermediate features
        """
        outputs = self.model(
            images,
            box_prompts=box_prompts,
            point_prompts=point_prompts
        )
        
        if return_features and hasattr(self.model, 'get_features'):
            outputs['features'] = self.model.get_features()
        
        return outputs
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA adapters only)."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
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
        merge: bool = False
    ) -> 'SAMLoRA':
        """
        Load model with LoRA adapters.
        
        Args:
            base_checkpoint: Path to base pretrained checkpoint
            lora_adapter_path: Path to LoRA adapter directory
            model_type: SAM model type
            lora_config: LoRA configuration
        
        Returns:
            Model with LoRA adapters loaded
        """
        # Create model
        model = cls(
            base_checkpoint=base_checkpoint,
            model_type=model_type,
            lora_config=lora_config or {}
        )
        
        model.model = load_lora_model(
            model.model,
            lora_adapter_path,
            merge=merge
        )
        
        return model


def load_sam_with_lora(
    base_checkpoint: str,
    lora_adapter_path: Optional[str] = None,
    model_type: str = 'vit_h',
    lora_config: Optional[Dict] = None,
    merge: bool = False
) -> SAMLoRA:
    """
    Factory function to load SAM with optional LoRA.
    
    Args:
        base_checkpoint: Path to pretrained checkpoint
        lora_adapter_path: Optional path to LoRA adapters
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        lora_config: LoRA configuration
        merge: Whether to merge LoRA adapters into base model
    
    Returns:
        SAMLoRA instance
    
    Example:
        >>> # Load pretrained SAM and apply LoRA
        >>> model = load_sam_with_lora(
        >>>     base_checkpoint='pretrained/sam_vit_h_4b8939.pth',
        >>>     model_type='vit_h',
        >>>     lora_config={'r': 8, 'lora_alpha': 16}
        >>> )
        >>> 
        >>> # Load fine-tuned LoRA adapters
        >>> model = load_sam_with_lora(
        >>>     base_checkpoint='pretrained/sam_vit_h_4b8939.pth',
        >>>     lora_adapter_path='experiments/exp1/teachers/sam_lora/',
        >>>     model_type='vit_h',
        >>>     merge=True  # For distillation
        >>> )
    """
    if lora_adapter_path:
        return SAMLoRA.from_lora_checkpoint(
            base_checkpoint=base_checkpoint,
            lora_adapter_path=lora_adapter_path,
            model_type=model_type,
            lora_config=lora_config or {},
            merge=merge
        )
    if lora_config is None:
        from core.constants import DEFAULT_SAM_LORA_CONFIG
        lora_config = DEFAULT_SAM_LORA_CONFIG['lora']
    
    return SAMLoRA(
        base_checkpoint=base_checkpoint,
        model_type=model_type,
        lora_config=lora_config
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
            self.sam = load_sam_with_lora(
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


