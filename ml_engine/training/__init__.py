"""Training module with LoRA support and training utilities."""

# Core utilities (no circular dependencies)
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

# Lazy import for TeacherTrainer to avoid circular dependency
# TeacherTrainer imports from models.teacher which imports from training.peft_utils
def __getattr__(name):
    """Lazy import to break circular dependencies."""
    if name == 'TeacherTrainer':
        from .teacher_trainer import TeacherTrainer
        return TeacherTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Trainers
    'TeacherTrainer',
    # Training utilities
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

