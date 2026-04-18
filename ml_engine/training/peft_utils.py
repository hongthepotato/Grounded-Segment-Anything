"""
PEFT (Parameter-Efficient Fine-Tuning) utilities for LoRA integration.

This module provides utilities for:
- Applying LoRA to models
- Verifying freezing status
- Loading and saving LoRA adapters
"""

import os
import logging
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    lora_config: Dict,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Apply LoRA to a model.

    This automatically freezes all base model parameters and adds
    trainable LoRA adapters to specified modules.

    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration dictionary with keys:
            - r: LoRA rank
            - lora_alpha: LoRA scaling factor
            - lora_dropout: Dropout probability
            - target_modules: List of module names to apply LoRA
        target_modules: Optional override for target modules

    Returns:
        Model with LoRA adapters applied

    Example:
        >>> model = load_grounding_dino()
        >>> lora_config = {
        >>>     'r': 16,
        >>>     'lora_alpha': 32,
        >>>     'lora_dropout': 0.1,
        >>>     'target_modules': ['self_attn.q_proj', 'self_attn.v_proj']
        >>> }
        >>> model = apply_lora(model, lora_config)
    """
    # Override target modules if provided
    if target_modules is not None:
        lora_config = lora_config.copy()
        lora_config['target_modules'] = target_modules

    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'FEATURE_EXTRACTION')
    )

    model = get_peft_model(model, peft_config)

    logger.info("Applied LoRA with rank %s", peft_config.r)
    logger.info("Target modules: %s", peft_config.target_modules)
    logger.info("All base model parameters frozen except LoRA adapters")

    return model


def verify_freezing(model: nn.Module, strict: bool = True) -> Dict[str, int]:
    """
    Verify that base model is frozen and only LoRA adapters are trainable.

    Args:
        model: Model to verify
        strict: If True, raises error if non-LoRA parameters are trainable.
                If False, only logs warnings for non-LoRA trainable parameters
                (use when you intentionally unfreeze some parameters like
                prediction heads).
                Note: A frozen LoRA parameter always raises regardless of strict,
                as this indicates a misconfiguration that would prevent training.

    Returns:
        Dictionary with parameter statistics:
            - frozen_params: Number of frozen parameters
            - trainable_params: Number of trainable parameters
            - lora_params: Number of trainable LoRA parameters
            - total_params: Total parameter count
            - trainable_ratio: Percentage of trainable parameters

    Raises:
        AssertionError: If strict=True and non-LoRA parameters are trainable,
                        or if any LoRA parameter is frozen (regardless of strict).

    Example:
        >>> # Strict mode (LoRA-only training)
        >>> model = apply_lora(base_model, lora_config)
        >>> stats = verify_freezing(model, strict=True)
        >>>
        >>> # Non-strict mode (LoRA + prediction heads)
        >>> model = apply_lora(base_model, lora_config)
        >>> unfreeze_prediction_heads(model)
        >>> stats = verify_freezing(model, strict=False)  # Won't raise for unfrozen heads
    """
    frozen_params = 0
    trainable_params = 0
    lora_params = 0
    non_lora_trainable = []

    for name, param in model.named_parameters():
        param_count = param.numel()

        if param.requires_grad:
            trainable_params += param_count

            # Check if this is a LoRA parameter
            if 'lora' in name.lower():
                lora_params += param_count
            else:
                non_lora_trainable.append(name)
                if strict:
                    raise AssertionError(
                        f"Non-LoRA param is trainable: {name}\n"
                        f"This defeats the purpose of LoRA! Only LoRA adapters should be trainable."
                    )
        else:
            frozen_params += param_count

            # A frozen LoRA parameter is always a misconfiguration — it would
            # silently prevent the adapter from training regardless of strict mode.
            if 'lora' in name.lower():
                raise AssertionError(
                    f"LoRA param is frozen: {name}\n"
                    f"LoRA adapters should be trainable!"
                )

    total_params = frozen_params + trainable_params
    trainable_ratio = 100 * trainable_params / total_params if total_params > 0 else 0

    stats = {
        'frozen_params': frozen_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'total_params': total_params,
        'trainable_ratio': trainable_ratio
    }

    logger.info("=" * 60)
    logger.info("LoRA Freezing Verification")
    logger.info("=" * 60)
    logger.info(" Frozen parameters:    %s (%sM)", frozen_params, frozen_params / 1e6)
    logger.info(" Trainable parameters: %s (%sM)", trainable_params, trainable_params / 1e6)
    logger.info(" LoRA parameters:      %s (%sM)", lora_params, lora_params / 1e6)
    logger.info(" Trainable ratio:      %s%%", trainable_ratio)
    logger.info("=" * 60)

    if trainable_ratio > 5.0:
        msg = " Warning: Trainable ratio (%.1f%%) is high for LoRA! Expected < 5%%" % trainable_ratio
        if strict:
            raise AssertionError(msg)
        logger.warning(msg)

    if non_lora_trainable:
        logger.warning(" Non-LoRA trainable parameters found: %s...", non_lora_trainable[:5])

    return stats


def save_lora_adapters(
    model: nn.Module,
    output_dir: str,
    safe_serialization: bool = True
) -> None:
    """
    Save only LoRA adapters (not the full model).

    Args:
        model: Model with LoRA adapters (must be a PEFT model with save_pretrained)
        output_dir: Directory to save adapters
        safe_serialization: Use safe tensors format

    Raises:
        ValueError: If model is not a PEFT model (no save_pretrained method)

    Example:
        >>> model = apply_lora(base_model, lora_config)
        >>> # ... training ...
        >>> save_lora_adapters(model, 'experiments/exp1/teachers/dino_lora/')
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization
        )
        logger.info("Saved LoRA adapters to: %s", output_dir)
    else:
        raise ValueError(
            "Model has no save_pretrained method — is it a PEFT model? "
            "Got: %s. Apply LoRA via apply_lora() before saving." % type(model).__name__
        )


def freeze_module(module: nn.Module) -> None:
    """
    Freeze all parameters in a module.

    Args:
        module: Module to freeze

    Example:
        >>> # Freeze image encoder
        >>> freeze_module(model.image_encoder)
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    Unfreeze all parameters in a module.

    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
