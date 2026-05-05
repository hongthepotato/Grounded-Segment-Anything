"""
LoRA weight merging utilities.

This module provides functions to merge LoRA adapters into base model weights
for standalone inference without PEFT dependency.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adapters into base model for standalone inference.

    This uses PEFT's merge_and_unload() to:
    1. Merge LoRA weights (A @ B) into original weights
    2. Remove PEFT wrapper, returning clean model

    Args:
        model: GroundingDINOLoRA model with PEFT wrapper

    Returns:
        Merged model without PEFT wrapper (standard GroundingDINO)

    Example:
        >>> lora_model = GroundingDINOLoRA(...)
        >>> merged = merge_lora_weights(lora_model)
        >>> # merged is now a standard model, no PEFT dependency
    """
    # `Any` annotation note: PEFT's `PeftModel` exposes `merge_and_unload`
    # via __getattr__ delegation to `self.base_model` (a `LoraModel`). mypy
    # cannot follow that delegation, so the call would fail static analysis
    # even with `peft_model: PeftModel = ...`. The hasattr() checks above are
    # the runtime contract; `Any` tells mypy "trust the duck-typing here."

    # Check if model has PEFT wrapper
    if hasattr(model, "model") and hasattr(model.model, "merge_and_unload"):
        # GroundingDINOLoRA wraps PEFT model in self.model
        logger.info("Merging LoRA weights into base model...")

        peft_model: Any = model.model
        merged_model = peft_model.merge_and_unload()

        logger.info("LoRA weights merged successfully")
        return merged_model

    if hasattr(model, "merge_and_unload"):
        # Direct PEFT model
        logger.info("Merging LoRA weights (direct PEFT model)...")
        direct_peft: Any = model
        merged_model = direct_peft.merge_and_unload()
        logger.info("LoRA weights merged successfully")
        return merged_model

    logger.warning("Model does not have LoRA adapters, returning as-is")
    return model


def save_merged_model(
    model: nn.Module,
    output_path: Path,
    class_names: Optional[list] = None,
    training_info: Optional[Dict[str, Any]] = None,
    model_name: str = "grounding_dino",
) -> Path:
    """
    Save merged model weights with metadata.

    Saves a checkpoint that can be loaded without PEFT:
    - model_state_dict: Model weights
    - class_names: Classes the model was trained on
    - metadata: Framework integrity fields (format, peft_merged, requires_peft)
    - training_info: Caller-supplied training provenance (epochs, metrics, etc.)

    Args:
        model: Merged model (after merge_lora_weights)
        output_path: Path to save .pth file
        class_names: List of class names used in training
        training_info: Caller-supplied training provenance; stored under a
            separate top-level checkpoint key so it cannot interfere with
            framework integrity fields in ``metadata``.

    Returns:
        Path to saved checkpoint

    Example:
        >>> merged = merge_lora_weights(lora_model)
        >>> path = save_merged_model(merged, Path("model.pth"), ["dog", "cat"])
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # `metadata` holds only the three framework integrity fields read by
    # load_merged_model. Caller provenance lives in the separate
    # `training_info` key so the two namespaces cannot interfere.
    # Explicit `Dict[str, Any]` annotation prevents mypy from widening the
    # literal's value type to `Collection[Any]`.
    metadata: Dict[str, Any] = {
        "format": f"merged_{model_name}",
        "peft_merged": True,
        "requires_peft": False,
    }

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names or [],
        "metadata": metadata,
    }
    if training_info:
        checkpoint["training_info"] = dict(training_info)

    # Save
    torch.save(checkpoint, output_path)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved merged model to: %s (%.1f MB)", output_path, size_mb)

    return output_path


def load_merged_model(checkpoint_path: Path, model: nn.Module, strict: bool = True) -> nn.Module:
    """
    Load merged model weights.

    Args:
        checkpoint_path: Path to merged checkpoint
        model: Model instance to load weights into
        strict: Whether to require exact key match

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Malformed merged-model checkpoint: missing 'model_state_dict' key in {checkpoint_path}"
        )

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Malformed merged-model checkpoint: 'metadata' is "
            f"{type(metadata).__name__}, expected dict in {checkpoint_path}"
        )

    fmt = metadata.get("format", "")
    if not fmt.startswith("merged_"):
        logger.warning("Checkpoint may not be a merged model format (got: %s)", fmt)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    logger.info("Loaded merged model from: %s", checkpoint_path)

    return model
