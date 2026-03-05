"""Training module with LoRA support and training utilities."""

# Core utilities (always available)
from .training_manager import TrainingManager
from .checkpoint_manager import CheckpointManager
from .losses import SegmentationLoss, CombinedTeacherLoss, build_criterion
from .peft_utils import (
    apply_lora,
    verify_freezing,
    load_lora_model,
    save_lora_adapters,
    freeze_module,
    unfreeze_module,
    partial_freeze_for_lora
)

# Lazy imports for heavy model-dependent modules
# These are only loaded when explicitly imported
def __getattr__(name):
    """Lazy import for heavy modules to avoid loading all model dependencies at import time."""
    if name == 'Trainer':
        from .trainer import Trainer
        return Trainer
    elif name == 'TeacherTrainer':
        # Backward compat alias
        from .trainer import Trainer
        return Trainer
    elif name == 'TrainingCancelledException':
        from .trainer import TrainingCancelledException
        return TrainingCancelledException
    elif name == 'BaseModelTrainer':
        from .model_trainers.base import BaseModelTrainer
        return BaseModelTrainer
    elif name == 'GroundingDINOTrainer':
        from .model_trainers.grounding_dino import GroundingDINOTrainer
        return GroundingDINOTrainer
    elif name == 'SAMTrainer':
        from .model_trainers.sam import SAMTrainer
        return SAMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main trainers (lazy loaded)
    'Trainer',
    'TeacherTrainer',  # Backward compat
    'TrainingCancelledException',
    # Model trainers (lazy loaded)
    'BaseModelTrainer',
    'GroundingDINOTrainer',
    'SAMTrainer',
    # Training utilities (always available)
    'TrainingManager',
    'CheckpointManager',
    # Losses
    'SegmentationLoss',
    'CombinedTeacherLoss',
    'build_criterion',
    # LoRA utilities
    'apply_lora',
    'verify_freezing',
    'load_lora_model',
    'save_lora_adapters',
    'freeze_module',
    'unfreeze_module',
    'partial_freeze_for_lora'
]
