"""
Grounding DINO model with LoRA integration.

This module provides a wrapper for Grounding DINO with:
- LoRA parameter-efficient fine-tuning
- Partial freezing strategy (freeze backbone, train decoder with LoRA)
- Training and inference modes
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from torch import nn

from core.constants import DEFAULT_DINO_LORA_CONFIG
from ml_engine.training.peft_utils import apply_lora, save_lora_adapters, verify_freezing

logger = logging.getLogger(__name__)


class GroundingDINOLoRA(nn.Module):
    """
    Grounding DINO with LoRA fine-tuning.

    Grounding DINO is an OPEN-VOCABULARY detector - it doesn't have fixed classes.
    Instead, it uses contrastive learning between visual features and text tokens.

    Freezing Strategy:
    1. PEFT automatically freezes ALL parameters when applying LoRA
    2. PEFT unfreezes ONLY LoRA adapters in target_modules (self-attention layers)
    3. ContrastiveEmbed has NO learnable parameters (just dot product)
    4. Backbone remains frozen by default (set freeze_backbone=False to unfreeze)

    Final Architecture:
    - Swin Transformer backbone: FROZEN (default) or UNFROZEN (if freeze_backbone=False)
    - Transformer decoder: LoRA adapters TRAINABLE (~2M params)
    - ContrastiveEmbed (class_embed): NO PARAMETERS (just dot product)
    - bbox_embed: Can optionally be made trainable

    Example:
        >>> model = GroundingDINOLoRA(
        >>>     base_checkpoint='data/models/pretrained/groundingdino_swint_ogc.pth',
        >>>     lora_config={'r': 16, 'lora_alpha': 32},
        >>>     freeze_backbone=True
        >>> )
        >>>
        >>> # Forward pass (training) - pass class names, they'll be formatted as "class1 . class2 . class3"
        >>> outputs = model(
        >>>     images=images,
        >>>     class_names=['dog', 'cat', 'car']
        >>> )
        >>> # Output: pred_logits shape = [B, num_queries, num_tokens]
    """

    def __init__(
        self,
        base_checkpoint: str,
        lora_config: Optional[Dict] = None,
        freeze_backbone: bool = True,
        freeze_bbox_embed: bool = False,
        bert_model_path: Optional[str] = None,
        _skip_lora_setup: bool = False,
    ):
        """
        Args:
            base_checkpoint: Path to pretrained Grounding DINO checkpoint
            lora_config: LoRA configuration dictionary
            freeze_backbone: Whether to freeze Swin Transformer backbone
            freeze_bbox_embed: Whether to freeze bbox prediction head
            bert_model_path: Optional path to local BERT model directory

        Note:
            - ContrastiveEmbed has NO parameters to train
            - LoRA adapters in the transformer decoder are trainable
            - bbox_embed can optionally be unfrozen for better box regression
        """
        super().__init__()

        self.lora_config = lora_config
        self.bert_model_path = bert_model_path
        self._text_token_mask_cache: Optional[torch.Tensor] = None
        self._text_token_mask_caption: Optional[str] = None
        self._positive_map_cache: Optional[torch.Tensor] = None
        self._positive_map_class_key: Optional[str] = None

        # `Any` annotation: self.model is a PEFT-wrapped GroundingDINO that
        # exposes attributes (.tokenizer, .base_model, .get_features, ...) via
        # nn.Module.__getattr__ — mypy resolves those through the Module stub
        # to Tensor/Module and rejects every callsite. Same boundary pattern
        # as model_trainers/base.py self.model: Any (TODO #13 / Step 2.4.3).
        logger.info("Loading Grounding DINO from: %s", base_checkpoint)
        self.model: Any = self._load_base_model(base_checkpoint)

        if not _skip_lora_setup:
            # Apply LoRA
            logger.info("Applying LoRA to Grounding DINO...")
            self._apply_lora()

            # Optionally unfreeze bbox prediction head
            if not freeze_bbox_embed:
                self._unfreeze_bbox_embed()

            # Optionally unfreeze backbone
            if not freeze_backbone:
                self._unfreeze_backbone()

            # Verify freezing
            verify_freezing(self.model, strict=False)

        logger.info("✓ Grounding DINO with LoRA initialized")

    def _load_base_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load pretrained Grounding DINO model.

        This method loads the official GroundingDINO model WITHOUT modification.
        Grounding DINO is an open-vocabulary model - it has NO fixed num_classes.

        Args:
            checkpoint_path: Path to pretrained checkpoint (.pth file)

        Returns:
            Loaded GroundingDINO model

        Raises:
            FileNotFoundError: If checkpoint or config file not found
            RuntimeError: If model building or loading fails
        """
        # Add GroundingDINO to path
        groundingdino_path = Path(__file__).parent.parent.parent.parent / "GroundingDINO"

        sys.path.insert(0, str(groundingdino_path))

        # Load config
        config_file = groundingdino_path / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file of Grounding DINO not found: {config_file}")

        logger.info("Loading config from: %s", config_file)
        args = SLConfig.fromfile(str(config_file))

        # Configure BERT path (for offline environments)
        if self.bert_model_path:
            bert_path = Path(self.bert_model_path)
            if not bert_path.exists():
                raise FileNotFoundError(f"Local BERT model not found: {bert_path}\n")
            logger.info("Using local BERT model from: %s", bert_path)
            # Set bert_base_uncased_path, NOT text_encoder_type!
            args.bert_base_uncased_path = str(bert_path.absolute())
        else:
            logger.info("BERT will be downloaded from HuggingFace: %s", args.text_encoder_type)
            logger.warning("No local BERT path provided. If offline, download BERT first!")
            # Ensure bert_base_uncased_path is None for online mode
            if not hasattr(args, "bert_base_uncased_path"):
                args.bert_base_uncased_path = None

        # Enable auxiliary losses for DETR-style training
        # This enables:
        # 1. Auxiliary outputs from all 6 decoder layers
        # 2. Encoder outputs for auxiliary loss
        args.aux_loss = True
        logger.info("Auxiliary losses enabled for DETR-style training")

        # Build model
        logger.info("Building Grounding DINO model...")
        try:
            model = build_model(args)
        except Exception as e:
            raise RuntimeError(f"Failed to build Grounding DINO model: {e}") from e

        # Load pretrained checkpoint. Use a fresh `path: Path` instead of
        # rebinding the str-typed `checkpoint_path` parameter (same shadowing
        # pattern as core/config.py and checkpoint_manager.py).
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Download pretrained weights from:\n"
                f"  https://github.com/IDEA-Research/GroundingDINO/releases"
            )

        logger.info("Loading checkpoint: %s", path)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

        # Extract state dict (handle different formats)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                logger.info("Checkpoint contains: epoch=%s", checkpoint.get("epoch", "N/A"))
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Clean and load state dict
        try:
            state_dict = clean_state_dict(state_dict)
            msg = model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}") from e

        # Report loading results
        if msg.missing_keys:
            logger.warning("Missing keys (%d): %s...", len(msg.missing_keys), msg.missing_keys[:5])

        if msg.unexpected_keys:
            logger.warning("Unexpected keys (%d): %s...", len(msg.unexpected_keys), msg.unexpected_keys[:5])

        if not msg.missing_keys and not msg.unexpected_keys:
            logger.info("All keys matched perfectly")

        # Verify model is in correct mode
        model.eval()  # Start in eval mode (will be set to train by trainer)

        logger.info("Grounding DINO model loaded successfully")
        return model

    def _apply_lora(self):
        """Apply LoRA adapters to transformer decoder."""
        # Apply LoRA to the model
        self.model = apply_lora(
            self.model,
            self.lora_config,
            target_modules=self.lora_config.get(
                "target_modules",
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"],
            ),
        )

    def _unfreeze_bbox_embed(self):
        """
        Unfreeze bbox prediction head for better box regression on custom datasets.

        Note: ContrastiveEmbed (class_embed) has NO parameters - it's just a dot product!
        Only bbox_embed has learnable parameters that can be unfrozen.
        """
        # Access the base model (PEFT wraps it)
        base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model

        trainable_count = 0
        # Unfreeze bbox_embed (box regression head)
        if hasattr(base_model, "bbox_embed"):
            for module in base_model.bbox_embed:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_count += param.numel()
            logger.info(" Unfrozen bbox_embed (box regression head): %d parameters", trainable_count)
        else:
            logger.warning(" bbox_embed not found in model structure")

    def _unfreeze_backbone(self):
        """Unfreeze Swin Transformer backbone (if not using LoRA-only training)."""
        # Access the base model (PEFT wraps it)
        base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model

        if hasattr(base_model, "backbone"):
            for param in base_model.backbone.parameters():
                param.requires_grad = True
            logger.info(" Unfrozen backbone (full fine-tuning mode)")
        else:
            logger.warning(" Backbone not found in model structure")

    def forward(
        self,
        images: torch.Tensor,
        class_names: Optional[List[str]] = None,
        captions: Optional[List[str]] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Grounding DINO.

        Args:
            images: Input images [B, 3, H, W] or NestedTensor
            class_names: List of class names (e.g., ['dog', 'cat', 'car'])
                        Will be formatted as "dog . cat . car" automatically
            captions: Alternative: provide pre-formatted captions directly
                     (one caption per batch item)
            return_features: Whether to return intermediate features

        Returns:
            Dict with:
                - pred_logits: [B, N, num_tokens] - similarity scores with text tokens
                - pred_boxes: [B, N, 4] - normalized boxes in [cx, cy, w, h] format
                - text_token_mask: [B, num_valid_tokens] - boolean mask for valid tokens
                - features: Optional intermediate features

        Note:
            Output shape is [B, N, num_tokens], NOT [B, N, num_classes]!
            You need to map high-scoring tokens back to class names.
        """
        # Format captions properly
        if captions is None:
            if class_names is None:
                raise ValueError("Must provide either class_names or captions")
            # Format as "class1 . class2 . class3"
            caption, _ = build_captions_and_token_span(class_names, force_lowercase=False)
            captions = [caption] * (
                images.tensors.shape[0] if hasattr(images, "tensors") else images.shape[0]
            )

        # Must use 'samples' parameter name to match original GroundingDINO signature
        outputs = self.model(samples=images, captions=captions)

        # Add text_token_mask for loss computation (needed to filter -inf padding).
        # All batch items share the same caption (class names don't change), so we
        # tokenize once and cache — the mask is identical for every forward call.
        assert captions is not None  # guaranteed: either passed in or built from class_names above
        caption_key = captions[0]
        if self._text_token_mask_cache is None or self._text_token_mask_caption != caption_key:
            base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
            tokenized = base_model.tokenizer(captions[:1], padding="longest", return_tensors="pt")
            self._text_token_mask_cache = tokenized.attention_mask.bool()  # [1, num_valid_tokens]
            self._text_token_mask_caption = caption_key

        # Expand cached mask to match current batch size and move to correct device
        assert self._text_token_mask_cache is not None
        batch_size = outputs["pred_logits"].shape[0]
        text_token_mask = self._text_token_mask_cache.expand(batch_size, -1).to(outputs["pred_logits"].device)
        outputs["text_token_mask"] = text_token_mask

        if return_features and hasattr(self.model, "get_features"):
            outputs["features"] = self.model.get_features()

        return outputs

    @property
    def tokenizer(self):
        """
        Access the BERT tokenizer from the wrapped GroundingDINO model.

        Returns:
            The BERT tokenizer used by Grounding DINO
        """
        # Access the base model through PEFT wrapper
        base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        return base_model.tokenizer

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, class_names: List[str], confidence_threshold: float = 0.3
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return detections in standard format.

        This handles all the Grounding DINO token-to-class conversion internally,
        returning clean (boxes, scores, labels) per image.

        Args:
            images: Input images [B, 3, H, W] or NestedTensor
            class_names: List of class names
            confidence_threshold: Minimum confidence to keep detections

        Returns:
            List of dicts (one per image), each with:
                - boxes: [N, 4] in normalized cxcywh format
                - scores: [N] confidence scores
                - labels: [N] class indices (0-based)
        """
        # Forward pass
        outputs = self.forward(images, class_names=class_names)
        pred_boxes = outputs["pred_boxes"]  # [B, num_queries, 4]
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_tokens]

        batch_size = pred_logits.shape[0]
        device = pred_logits.device

        # Build positive_map for token -> class mapping (same for all batch items).
        # class_names never change during evaluation, so cache after first build.
        class_key = ",".join(class_names)
        if self._positive_map_cache is None or self._positive_map_class_key != class_key:
            caption, cat2tokenspan = build_captions_and_token_span(class_names, force_lowercase=False)
            tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
            token_span_per_class = []
            for name in class_names:
                if name not in cat2tokenspan:
                    raise ValueError(
                        f"Class '{name}' not found in cat2tokenspan during predict()!\n"
                        f"Available classes: {list(cat2tokenspan.keys())}\n"
                        f"This indicates a mismatch between class_names and caption tokenization."
                    )
                token_span_per_class.append(cat2tokenspan[name])
            self._positive_map_cache = create_positive_map_from_span(
                tokenized, token_span_per_class, max_text_len=pred_logits.shape[-1]
            )  # [num_classes, num_tokens]
            self._positive_map_class_key = class_key

        assert self._positive_map_cache is not None
        positive_map = self._positive_map_cache.to(device)  # [num_classes, num_tokens]

        # Convert to per-image predictions
        results = []
        num_classes = len(class_names)

        for b in range(batch_size):
            # Convert token logits to class scores
            token_probs = pred_logits[b].sigmoid()  # [num_queries, num_tokens]
            class_probs = torch.zeros(token_probs.shape[0], num_classes, device=device)

            for c in range(num_classes):
                token_mask = positive_map[c] > 0
                if token_mask.sum() > 0:
                    class_probs[:, c] = token_probs[:, token_mask].mean(dim=-1)

            # Get best class per query
            scores, labels = class_probs.max(dim=-1)

            # Filter by confidence
            keep = scores > confidence_threshold

            results.append({"boxes": pred_boxes[b][keep], "scores": scores[keep], "labels": labels[keep]})

        return results

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA adapters only)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_lora_adapters(self, output_dir: str):
        """Save only LoRA adapters."""

        save_lora_adapters(self.model, output_dir)

    @classmethod
    def from_lora_checkpoint(
        cls,
        base_checkpoint: str,
        lora_adapter_path: str,
        merge: bool = False,
        bert_model_path: Optional[str] = None,
    ) -> "GroundingDINOLoRA":
        """
        Load model with LoRA adapters.

        Args:
            base_checkpoint: Path to base pretrained checkpoint
            lora_adapter_path: Path to LoRA adapter directory
            merge: Whether to merge LoRA weights into base model
            bert_model_path: Optional path to local BERT model

        Returns:
            Model with LoRA adapters loaded
        """
        instance = cls(
            base_checkpoint=base_checkpoint, bert_model_path=bert_model_path, _skip_lora_setup=True
        )
        from peft import PeftModel

        instance.model = PeftModel.from_pretrained(instance.model, lora_adapter_path)

        if merge:
            instance.model = instance.model.merge_and_unload()

        return instance


def load_grounding_dino_with_lora(
    base_checkpoint: str,
    lora_adapter_path: Optional[str] = None,
    lora_config: Optional[Dict] = None,
    freeze_backbone: bool = True,
    freeze_bbox_embed: bool = False,
    bert_model_path: Optional[str] = None,
    merge: bool = False,
) -> GroundingDINOLoRA:
    """
    Factory function to load Grounding DINO with optional LoRA.

    Args:
        base_checkpoint: Path to pretrained checkpoint
        lora_adapter_path: Optional path to LoRA adapters
        lora_config: LoRA configuration
        freeze_backbone: Whether to freeze Swin Transformer backbone
        freeze_bbox_embed: Whether to freeze bbox prediction head
        bert_model_path: Optional path to local BERT model directory
        merge: Whether to merge LoRA weights (for distillation)

    Returns:
        GroundingDINOLoRA instance

    Example:
        >>> # Load pretrained model and apply LoRA (online)
        >>> model = load_grounding_dino_with_lora(
        >>>     base_checkpoint='pretrained/groundingdino.pth',
        >>>     lora_config={'r': 16, 'lora_alpha': 32},
        >>>     freeze_backbone=True,
        >>>     freeze_bbox_embed=False,
        >>>     bert_model_path=None,
        >>>     merge=False
        >>> )
        >>>
        >>> # Load with local BERT
        >>> model = load_grounding_dino_with_lora(
        >>>     base_checkpoint='pretrained/groundingdino.pth',
        >>>     lora_config={'r': 16, 'lora_alpha': 32},
        >>>     bert_model_path='data/models/pretrained/bert-base-uncased'
        >>> )
        >>>
        >>> # Forward pass with class names
        >>> outputs = model(samples, class_names=['dog', 'cat', 'car'])
    """
    if lora_adapter_path:
        return GroundingDINOLoRA.from_lora_checkpoint(
            base_checkpoint=base_checkpoint,
            lora_adapter_path=lora_adapter_path,
            merge=merge,
            bert_model_path=bert_model_path,
        )

    if lora_config is None:
        lora_config = DEFAULT_DINO_LORA_CONFIG["lora"]

    return GroundingDINOLoRA(
        base_checkpoint=base_checkpoint,
        lora_config=lora_config,
        freeze_backbone=freeze_backbone,
        freeze_bbox_embed=freeze_bbox_embed,
        bert_model_path=bert_model_path,
    )
