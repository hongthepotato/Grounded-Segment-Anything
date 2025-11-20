"""
PEFT (Parameter-Efficient Fine-Tuning) utilities for LoRA integration.

This module provides utilities for:
- Applying LoRA to models
- Verifying freezing status
- Computing trainable parameter statistics
- Loading and saving LoRA adapters
"""

import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, List, Optional, Tuple
import logging

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
                If False, only logs warnings (use when you intentionally unfreeze
                some parameters like prediction heads).
    
    Returns:
        Dictionary with parameter statistics:
            - frozen_params: Number of frozen parameters
            - trainable_params: Number of trainable parameters
            - lora_params: Number of trainable LoRA parameters
            - trainable_ratio: Percentage of trainable parameters
    
    Raises:
        AssertionError: If strict=True and non-LoRA parameters are trainable
    
    Example:
        >>> # Strict mode (LoRA-only training)
        >>> model = apply_lora(base_model, lora_config)
        >>> stats = verify_freezing(model, strict=True)
        >>> 
        >>> # Non-strict mode (LoRA + prediction heads)
        >>> model = apply_lora(base_model, lora_config)
        >>> unfreeze_prediction_heads(model)
        >>> stats = verify_freezing(model, strict=False)  # Won't raise error
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
                        f"❌ Non-LoRA param is trainable: {name}\n"
                        f"This defeats the purpose of LoRA! Only LoRA adapters should be trainable."
                    )
        else:
            frozen_params += param_count

            # Check if LoRA parameter is frozen
            if 'lora' in name.lower():
                raise AssertionError(
                    f"❌ LoRA param is frozen: {name}\n"
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
    logger.info(" Frozen parameters:    %s (%sM)", frozen_params, frozen_params/1e6)
    logger.info(" Trainable parameters: %s (%sM)", trainable_params, trainable_params/1e6)
    logger.info(" LoRA parameters:      %s (%sM)", lora_params, lora_params/1e6)
    logger.info(" Trainable ratio:      %s%%", trainable_ratio)
    logger.info("=" * 60)

    if trainable_ratio > 5.0:
        msg = " Warning: Trainable ratio (%s%%) is high for LoRA! Expected < 5%%", trainable_ratio
        if strict:
            raise AssertionError(msg)
        logger.warning(msg)

    if non_lora_trainable:
        logger.warning(" Non-LoRA trainable parameters found: %s...", non_lora_trainable[:5])

    return stats


def get_trainable_parameters_summary(model: nn.Module) -> str:
    """
    Get a formatted summary of trainable parameters.
    
    Args:
        model: Model to summarize
    
    Returns:
        Formatted string with parameter summary
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable / total if total > 0 else 0
    
    return (
        f"Trainable params: {trainable:,} || "
        f"All params: {total:,} || "
        f"Trainable%: {ratio:.2f}%"
    )


def print_trainable_parameters(model: nn.Module) -> None:
    """Print trainable parameters information."""
    print(get_trainable_parameters_summary(model))


def load_lora_model(
    base_model: nn.Module,
    lora_adapter_path: str,
    merge: bool = False
) -> nn.Module:
    """
    Load a LoRA-adapted model from base model and adapter weights.
    
    Args:
        base_model: Base pretrained model
        lora_adapter_path: Path to LoRA adapter directory
        merge: If True, merge adapter into base model for faster inference
    
    Returns:
        Model with LoRA adapters loaded
    
    Example:
        >>> base_model = load_grounding_dino('pretrained/groundingdino.pth')
        >>> model = load_lora_model(
        >>>     base_model,
        >>>     lora_adapter_path='experiments/exp1/teachers/dino_lora/',
        >>>     merge=True  # For distillation
        >>> )
    """
    logger.info(f"Loading LoRA adapters from: {lora_adapter_path}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    logger.info("✓ LoRA adapters loaded successfully")
    
    # Merge if requested
    if merge:
        logger.info("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        logger.info("✓ LoRA adapters merged. Model is now a single fine-tuned model.")
    
    return model


def save_lora_adapters(
    model: nn.Module,
    output_dir: str,
    safe_serialization: bool = True
) -> None:
    """
    Save only LoRA adapters (not the full model).
    
    Args:
        model: Model with LoRA adapters
        output_dir: Directory to save adapters
        safe_serialization: Use safe tensors format
    
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
        logger.info(f"✓ Saved LoRA adapters to: {output_dir}")
    else:
        logger.warning("Model does not have save_pretrained method. Saving full state dict.")
        torch.save(model.state_dict(), os.path.join(output_dir, 'adapter_model.bin'))


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


def partial_freeze_for_lora(
    model: nn.Module,
    freeze_modules: List[str],
    lora_config: Dict,
    lora_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Apply partial freeze strategy + LoRA.
    
    This is the recommended approach:
    1. Freeze specified modules (e.g., image encoder)
    2. Apply LoRA to remaining modules (e.g., decoder)
    
    Args:
        model: Base model
        freeze_modules: List of module names to freeze
        lora_config: LoRA configuration
        lora_modules: Optional list of module names to apply LoRA to
    
    Returns:
        Model with partial freeze + LoRA applied
    
    Example:
        >>> # For SAM
        >>> model = load_sam()
        >>> model = partial_freeze_for_lora(
        >>>     model,
        >>>     freeze_modules=['image_encoder', 'prompt_encoder'],
        >>>     lora_config=sam_lora_config,
        >>>     lora_modules=['mask_decoder']
        >>> )
    """
    # Freeze specified modules
    for name, module in model.named_children():
        if name in freeze_modules:
            freeze_module(module)
            logger.info(f"❄️  Frozen module: {name}")
    
    # Apply LoRA
    if lora_modules:
        # Filter target modules to only those in lora_modules
        lora_config = lora_config.copy()
        lora_config['target_modules'] = [
            tm for tm in lora_config['target_modules']
            if any(lm in tm for lm in lora_modules)
        ]
    
    model = apply_lora(model, lora_config)
    
    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count LoRA-specific parameters.
    
    Args:
        model: Model with LoRA
    
    Returns:
        Dictionary with LoRA parameter counts
    """
    lora_a_params = 0
    lora_b_params = 0
    other_lora_params = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            if 'lora_a' in name.lower():
                lora_a_params += param.numel()
            elif 'lora_b' in name.lower():
                lora_b_params += param.numel()
            else:
                other_lora_params += param.numel()
    
    total_lora = lora_a_params + lora_b_params + other_lora_params
    
    return {
        'lora_a_params': lora_a_params,
        'lora_b_params': lora_b_params,
        'other_lora_params': other_lora_params,
        'total_lora_params': total_lora
    }


def get_lora_rank(model: nn.Module) -> Optional[int]:
    """
    Get LoRA rank from model.
    
    Args:
        model: Model with LoRA
    
    Returns:
        LoRA rank if found, else None
    """
    for name, param in model.named_parameters():
        if 'lora_a' in name.lower():
            # LoRA A has shape [r, input_dim]
            return param.shape[0]
    return None


def enable_lora_dropout(model: nn.Module, dropout_rate: float = 0.1) -> None:
    """
    Enable dropout in LoRA layers during training.
    
    Args:
        model: Model with LoRA
        dropout_rate: Dropout rate to set
    """
    for name, module in model.named_modules():
        if 'lora' in name.lower() and isinstance(module, nn.Dropout):
            module.p = dropout_rate
            logger.info(f"Set dropout rate to {dropout_rate} for: {name}")


def disable_lora_dropout(model: nn.Module) -> None:
    """Disable dropout in LoRA layers for inference."""
    enable_lora_dropout(model, dropout_rate=0.0)
